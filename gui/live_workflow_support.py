"""GUI ライブ文字起こしの補助部品(LiveWorkflow 本体から切り出した純粋な関数・データクラス)

バナー種別・バナー文言・開始前の検証・ダウンロード進捗の整形、差し替え可能な依存(LiveDeps)と
コールバック群のデータクラス、エンジン直列化ラッパーを置く。Tkinter には依存しない。
重い依存(faster_whisper / huggingface_hub / mlx / openai)は各関数内で遅延 import されるため、
このモジュールの import は軽い。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import VoiceNoteConfig
from live import engines, local_llm, model_fetch, vad
from live.bootstrap import BackendCheck
from live.context import LiveContext
from live.level_meter import ChunkSink, InputLevelAssessment, LevelMeter
from live.model_fetch import ProgressCallback, RepoSpec
from live.session import LiveSession
from live.session_models import LiveSettings
from note_writer import save_live_note
from recorder import ThreadedRecorder

from .devices import parse_device_id
from .live_state import LiveViewState

_logger = logging.getLogger("voicenote")

PROGRESS_INTERVAL_SEC = 0.25  # ダウンロード進捗バナーの更新間隔
ENGINE_LOCK_TIMEOUT_SEC = 30.0  # load() が前回の実行の transcribe 完了を待つ上限

BANNER_MODEL_DOWNLOAD = "model_download"
BANNER_ENGINE_FALLBACK = "engine_fallback"
BANNER_OLLAMA = "ollama"
BANNER_VAD_FALLBACK = "vad_fallback"
BANNER_EXTERNAL_SEND = "external_send"
BANNER_LOW_INPUT = "low_input"


def validate_live_start(
    config: VoiceNoteConfig, device_label: str, *, mic_busy: bool, stream_closing: bool = False
) -> tuple[str, str] | None:
    """ライブ開始前の検証。問題なければ None、あれば (title, message)

    Args:
        config: 現在の設定
        device_label: デバイス選択欄のラベル('[3] マイク' 形式。無ければ 'デバイスなし')
        mic_busy: 通常の録音・文字起こしなど、マイクや処理を使う操作が実行中か
        stream_closing: 通常録音で停止がタイムアウトし放棄したストリームが、まだ解放中か
    """
    if mic_busy:
        return ("実行中", "他の録音・文字起こしが実行中です。終了してから開始してください")
    if stream_closing:
        return (
            "ストリーム解放中",
            "前回の録音のストリームがまだ解放されていません。しばらく待ってから再試行してください",
        )
    if not config.save_folder:
        return (
            "設定が必要",
            "先に 設定から文字起こし保存フォルダ(ライブノートの保存先)を設定してください",
        )
    if parse_device_id(device_label) is None:
        return ("デバイスなし", "入力デバイスが見つかりません。マイクを接続してください")
    try:
        LiveSettings.from_config(config)
    except ValueError as e:
        return ("ライブ設定が不正", f"ライブ文字起こしの設定が不正です: {e}")
    return None


def external_send_banner_text(transcription_mode: str, format_mode: str) -> str | None:
    """GUI 向けの外部送信バナー文言。OpenAI へ何も送らない設定なら None

    ライブ文字起こし自体はローカルで完結する。送信が起きるのは「文字起こしだけする」モードでの
    再文字起こしで、transcription_mode=openai なら音声、format_mode=llm なら本文が送られる。
    (live.note.external_send_warning は CLI 向けの文言なので GUI では使わない)
    """
    sends_audio = transcription_mode == "openai"
    sends_text = format_mode == "llm"
    if sends_audio and sends_text:
        what = "音声と文字起こし本文が"
    elif sends_audio:
        what = "音声が"
    elif sends_text:
        what = "文字起こし本文が"
    else:
        return None
    return (
        "ライブ文字起こし自体はこのPCの中だけで処理されます。"
        "ただし現在の設定では「文字起こしだけする」モードでの高精度な再文字起こしで、"
        f"{what} OpenAI に送信されます（設定ダイアログで変更できます）"
    )


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.1f} GB"
    return f"{num_bytes / 1_000_000:.0f} MB"


def format_download_progress(model: str, done: int, total: int) -> str:
    """モデル取得バナーの文言。total=0 は総量不明"""
    head = f"音声認識モデル {model} をダウンロードしています"
    if total <= 0:
        return f"{head}...（{_format_bytes(done)}）"
    percent = min(100, int(done * 100 / total))
    return f"{head}... {_format_bytes(done)} / {_format_bytes(total)}（{percent}%）"


def ollama_banner_text(check: BackendCheck) -> str:
    """要約なしで続行する旨のバナー(`ollama serve` / `ollama pull` の案内は message に含む)"""
    sep = "。" if check.status == "invalid_url" else " "
    return f"{check.message}{sep}要約なしで続行します。"


def dropped_chunks(session) -> int:
    """セッションの破棄チャンク数(属性が無ければ 0、負値は 0 に丸める)"""
    return max(0, int(getattr(session, "dropped_chunks", 0)))


def _default_recorder_factory(device_id: int | None, dest: Path, sink: ChunkSink):
    return ThreadedRecorder(device_id, dest, chunk_sink=sink)


@dataclass(frozen=True)
class ModelFetcher:
    """Whisper モデルの取得手段(live.model_fetch と同じシグネチャ)"""

    resolve_repo: Callable[[str, str], RepoSpec | None] = model_fetch.resolve_repo
    is_cached: Callable[[RepoSpec], bool] = model_fetch.is_cached
    fetch: Callable[[RepoSpec, ProgressCallback | None], str] = model_fetch.fetch


@dataclass(frozen=True)
class LiveDeps:
    """差し替え可能な依存。既定は本番実装(重い依存は各関数内で遅延 import される)"""

    recorder_factory: Callable[[int | None, Path, ChunkSink], Any] = _default_recorder_factory
    session_factory: Callable[..., Any] = LiveSession
    create_engine: Callable[..., Any] = engines.create_engine
    unavailable_error: type[Exception] = engines.EngineUnavailableError
    create_vad: Callable[[str], Any] = vad.create_vad
    vad_kind: Callable[[Any], str | None] = vad.vad_kind
    client_factory: Callable[..., Any] = local_llm.LocalLLMClient
    probe_fn: Callable[[Any, str], Any] = local_llm.probe
    model_fetcher: ModelFetcher = field(default_factory=ModelFetcher)
    meter_factory: Callable[[], LevelMeter] = LevelMeter
    note_saver: Callable[..., Path] = save_live_note


@dataclass(frozen=True)
class LiveWorkflowCallbacks:
    """LiveWorkflow から呼ばれる UI 更新コールバック群(App 側で submit 経由にすること)"""

    on_state: Callable[[LiveViewState], None]
    on_log: Callable[[str], None]
    on_recording_started: Callable[[], None]
    on_wav_saved: Callable[[Path], None]
    on_done: Callable[[Path, Path], None]  # (wav_path, note_path)
    on_error: Callable[[str], None]


@dataclass(frozen=True)
class MeterReading:
    """poll_meter() の読み出し結果"""

    level_db: float
    backlog_sec: float
    dropped_chunks: int
    input_level: InputLevelAssessment


@dataclass(frozen=True)
class Run:
    """1回のライブ実行の文脈。設定は開始時点のものを使い続ける"""

    gen: int
    config: VoiceNoteConfig
    device_id: int | None
    device_label: str
    dest: Path
    cancel: threading.Event


class Cancelled(Exception):
    """PREPARING 中に取り消された"""


class EngineBusyError(RuntimeError):
    """前回の実行の asr スレッドがエンジンを使い続けていて load() できない"""


class SerializedEngine:
    """load / transcribe を workflow 共有のロック下で実行するラッパー

    session.stop() がタイムアウトしても asr スレッドは処理中の区間を終えるまで動き続ける。
    次の実行の asr スレッドと同時に mlx(Metal)を使わないよう、実行をまたいで直列化する。
    放棄した asr スレッドが transcribe で固まった場合に load() が無通知で待ち続けないよう、
    load() のロック取得は load_timeout 秒で諦めて EngineBusyError を送出する
    (LiveSession が on_warning で知らせ、録音は継続する)。
    """

    def __init__(
        self, inner, lock: threading.Lock, *, load_timeout: float = ENGINE_LOCK_TIMEOUT_SEC
    ):
        self._inner = inner
        self._lock = lock
        self._load_timeout = load_timeout

    def load(self) -> None:
        if not self._lock.acquire(timeout=self._load_timeout):
            raise EngineBusyError(
                "前回の処理が終わっていません。しばらくしてからもう一度お試しください"
            )
        try:
            self._inner.load()
        finally:
            self._lock.release()

    def transcribe(self, samples, language, prompt=None):
        with self._lock:
            return self._inner.transcribe(samples, language, prompt=prompt)


class DownloadProgress:
    """fetch の進捗を間引いてバナー文言に変換する(総量が変わった時・完了時は必ず通知)"""

    def __init__(self, model: str, clock: Callable[[], float], report: Callable[[str], None]):
        self._model = model
        self._clock = clock
        self._report = report
        self._last_at: float | None = None
        self.total = 0

    def __call__(self, done: int, total: int) -> None:
        now = self._clock()
        changed = total != self.total
        finished = total > 0 and done >= total
        self.total = total
        recent = self._last_at is not None and now - self._last_at < PROGRESS_INTERVAL_SEC
        if recent and not changed and not finished:
            return
        self._last_at = now
        self._report(format_download_progress(self._model, done, total))


def coerce_context(context: LiveContext | None) -> LiveContext:
    """update_context の引数を検証する。None は空の参考情報として扱う"""
    if context is None:
        return LiveContext()
    if not isinstance(context, LiveContext):
        raise TypeError(f"context は LiveContext が必要です: {type(context).__name__}")
    return context


def push_context(session, context: LiveContext) -> None:
    """セッションへ参考情報を反映する。失敗してもライブは止めない(本文はログに出さない)"""
    try:
        session.set_context(context)
    except Exception:
        _logger.exception("ライブ要約の参考情報(参加者・メモ)を反映できませんでした")


def always_allowed() -> bool:
    """can_start の既定値(常に開始を許可する)"""
    return True
