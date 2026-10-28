"""停止後の高精度再文字起こし(CLI と GUI で共有する・Tk 非依存)

ライブ録音の停止・ドレイン・速報版ノート保存の後に、確定済みの WAV 全体を高精度モデル
(既定 large-v3, mlx)で文字起こしし直す。WAV は読むだけで、停止順序の不変条件
「WAV確定 → ストリーム停止 → ドレイン」には関与しない。

メモリの制約: mlx の ModelHolder は1枠で、新モデルを読み込んでから旧モデルを置き換える。
2モデルを同時に載せないよう、リアルタイム用エンジンを解放してから高精度モデルを読み込み、
終わったら高精度モデルも解放する。ロード・文字起こし・解放は同じスレッドで行う。

キャンセル・期限は窓(約60秒)の間で確認する。窓の途中では止まらない。
文字起こし本文・ヒント文はログや例外メッセージに出さない。
"""

from __future__ import annotations

import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from live.asr_hint import HINT_MIN_SEGMENT_SEC, is_hint_echo
from live.hallucination import is_hallucination
from live.note import format_timestamp
from live.session_models import LiveSegment, LiveSettings

_logger = logging.getLogger("voicenote")

REFINE_ENGINE = "mlx"
REFINE_MAX_RTF = 1.0  # 期限 = 録音長 × これ + 読み込みの猶予
REFINE_LOAD_GRACE_SEC = 300.0
REFINE_SAMPLE_RATE = 16000
WINDOW_TARGET_SEC = 60.0  # 1窓の目安。キャンセル・進捗の粒度になる
WINDOW_SEARCH_SEC = 5.0  # 切れ目を目安の前後この範囲で探す
QUIET_PROBE_SEC = 0.1  # 静かさを測る単位
_INT16_SCALE = 32768.0

ProgressCallback = Callable[[float, float, "float | None"], None]


class CancelFlag(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class RefineDecision:
    """再文字起こしを行うか。行わない理由(利用者向け)は reason。無効設定なら空文字"""

    run: bool
    reason: str


@dataclass(frozen=True)
class RefineResult:
    """再文字起こしの結果。complete が False ならキャンセル・期限超過で途中までしかない"""

    segments: tuple[LiveSegment, ...]
    complete: bool
    elapsed: float


def decide_refine(settings: LiveSettings, effective_engine: str, is_cached: bool) -> RefineDecision:
    """設定・実際に動いたエンジン・モデルの取得状況から、再文字起こしを行うか決める"""
    if not settings.refine_enabled:
        return RefineDecision(False, "")
    if effective_engine != REFINE_ENGINE:
        return RefineDecision(
            False,
            "高精度の再文字起こしは mlx エンジン使用時のみ行います"
            f"（今回は {effective_engine} で動作したためスキップしました）",
        )
    if not is_cached:
        return RefineDecision(
            False,
            f"高精度モデル {settings.refine_model} が未取得のため、再文字起こしをスキップしました",
        )
    return RefineDecision(True, "")


def rms_reader(samples: np.ndarray) -> Callable[[int, int], float]:
    """samples[start:end] の RMS を返す関数を作る(plan_windows 用)"""

    def read_rms(start: int, end: int) -> float:
        part = np.asarray(samples[start:end], dtype=np.float64)
        return float(np.sqrt(np.mean(part * part))) if part.size else 0.0

    return read_rms


def _quietest_cut(
    start: int, target: int, search: int, probe: int, read_rms: Callable[[int, int], float]
) -> int:
    """目安位置 start+target の前後 search 内で最も静かな probe 区間の中央を返す"""
    center = start + target
    positions = range(center - search, center + search - probe + 1, probe)
    best = min(positions, key=lambda p: (read_rms(p, p + probe), abs(p + probe // 2 - center)))
    return best + probe // 2


def plan_windows(
    num_samples: int,
    read_rms: Callable[[int, int], float],
    sample_rate: int,
    target_sec: float = WINDOW_TARGET_SEC,
    search_sec: float = WINDOW_SEARCH_SEC,
) -> tuple[tuple[int, int], ...]:
    """音声を約 target_sec 秒の窓に分ける(純関数)。切れ目は前後 search_sec 内で最も静かな位置

    残りが target_sec + search_sec 以下なら最後の窓にまとめる。窓は隙間なく連続する。
    """
    target = int(target_sec * sample_rate)
    search = int(search_sec * sample_rate)
    probe = max(1, int(QUIET_PROBE_SEC * sample_rate))
    windows: list[tuple[int, int]] = []
    start = 0
    while num_samples - start > target + search:
        cut = _quietest_cut(start, target, search, probe, read_rms)
        windows.append((start, cut))
        start = cut
    if start < num_samples:
        windows.append((start, num_samples))
    return tuple(windows)


def _read_chunks(path: Path) -> tuple[bytes | None, int | None, int | None]:
    """RIFF チャンクを辿り (fmt 本体, data の開始位置, data の宣言サイズ) を返す"""
    fmt = None
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("WAV ファイルではありません")
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                return fmt, None, None
            chunk_id, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if chunk_id == b"data":
                return fmt, f.tell(), size
            body = f.read(size + (size & 1))
            if chunk_id == b"fmt ":
                fmt = body[:size]


def load_wav_mmap(path: Path) -> np.ndarray:
    """16kHz mono int16 の WAV を memmap で開く(全体をメモリに読まない)

    data の宣言サイズが実ファイルより大きい場合(中断した録音)は実サイズに丸める。

    Raises:
        ValueError: WAV でない・形式が違う・音声が空
    """
    path = Path(path)
    fmt, offset, declared = _read_chunks(path)
    if fmt is None or len(fmt) < 16 or offset is None:
        raise ValueError("WAV の形式情報または音声データが見つかりません")
    audio_format, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if audio_format not in (1, 0xFFFE) or (channels, rate, bits) != (1, REFINE_SAMPLE_RATE, 16):
        raise ValueError(
            f"16kHz mono 16bit の WAV が必要です（{rate}Hz / {channels}ch / {bits}bit）"
        )
    available = max(0, path.stat().st_size - offset)
    frames = min(declared, available) // 2
    if frames == 0:
        raise ValueError("WAV に音声データがありません")
    return np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=(frames,))


def refine_deadline(duration_sec: float, now: float) -> float:
    """録音長から再文字起こしの期限(clock 基準の絶対時刻)を求める"""
    return now + duration_sec * REFINE_MAX_RTF + REFINE_LOAD_GRACE_SEC


def _safe_progress(on_progress: ProgressCallback, done: float, total: float, eta) -> None:
    try:
        on_progress(done, total, eta)
    except Exception:
        _logger.exception("再文字起こしの進捗通知に失敗しました")


def _window_segments(engine, audio: np.ndarray, start: int, language, hint, first_index: int):
    """1窓を文字起こしし、幻覚・ヒント反響を除いて録音先頭からの時刻に直す"""
    offset = start / REFINE_SAMPLE_RATE
    duration = len(audio) / REFINE_SAMPLE_RATE
    window_hint = hint if duration >= HINT_MIN_SEGMENT_SEC else None
    pieces = engine.transcribe_timed(audio, language, prompt=window_hint)
    kept = [
        p
        for p in pieces
        if p.text
        and not is_hallucination(p.text, p.no_speech_prob, p.avg_logprob)
        and not is_hint_echo(p.text, window_hint)
    ]
    return tuple(
        LiveSegment(
            first_index + i,
            offset + max(0.0, p.start),
            offset + min(duration, max(p.start, p.end)),
            p.text,
        )
        for i, p in enumerate(kept)
    )


@dataclass(frozen=True)
class _Job:
    samples: np.ndarray
    windows: tuple[tuple[int, int], ...]
    language: str | None
    hint: str | None
    cancel: CancelFlag
    on_progress: ProgressCallback
    deadline: float
    clock: Callable[[], float]


def _transcribe_windows(engine, job: _Job) -> tuple[tuple[LiveSegment, ...], bool]:
    """窓を順に文字起こしする。窓の間で cancel・期限を確認し、止まれば complete=False"""
    total = len(job.samples) / REFINE_SAMPLE_RATE
    segments: tuple[LiveSegment, ...] = ()
    began = job.clock()
    for start, end in job.windows:
        if job.cancel.is_set() or job.clock() >= job.deadline:
            return segments, False
        audio = np.asarray(job.samples[start:end], dtype=np.float32) / _INT16_SCALE
        found = _window_segments(engine, audio, start, job.language, job.hint, len(segments))
        segments = (*segments, *found)
        done = end / REFINE_SAMPLE_RATE
        spent = job.clock() - began
        eta = spent / done * (total - done) if done > 0 else None
        _safe_progress(job.on_progress, done, total, eta)
    return segments, True


def _unload(engine) -> None:
    try:
        engine.unload()
    except Exception as e:
        _logger.warning("高精度モデルの解放に失敗しました: %s", type(e).__name__)


def refine_wav(
    wav: Path,
    make_engine: Callable[[], object],
    release_live: Callable[[], None],
    *,
    language: str | None,
    hint: str | None,
    cancel: CancelFlag,
    on_progress: ProgressCallback,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RefineResult:
    """WAV 全体を高精度モデルで文字起こしし直す

    順序: WAV を検証 → リアルタイム用エンジン解放 → 高精度モデル生成・ロード → 窓ごとに
    文字起こし → 高精度モデル解放。deadline(clock 基準)の既定は録音長 × REFINE_MAX_RTF +
    REFINE_LOAD_GRACE_SEC。on_progress(処理済み秒, 全体秒, 残り秒の見込み|None) を呼ぶ。

    Raises:
        ValueError: WAV が 16kHz mono int16 でない・空(リアルタイム用エンジンは解放しない)
        Exception: エンジンのロード・推論の失敗はそのまま送出する(高精度モデルは解放済み)
    """
    started = clock()
    samples = load_wav_mmap(wav)
    total = len(samples) / REFINE_SAMPLE_RATE
    job = _Job(
        samples=samples,
        windows=plan_windows(len(samples), rms_reader(samples), REFINE_SAMPLE_RATE),
        language=language,
        hint=hint,
        cancel=cancel,
        on_progress=on_progress,
        deadline=refine_deadline(total, started) if deadline is None else deadline,
        clock=clock,
    )
    _safe_progress(on_progress, 0.0, total, None)
    release_live()
    if cancel.is_set():
        return RefineResult((), False, clock() - started)
    engine = make_engine()
    try:
        engine.load()
        segments, complete = _transcribe_windows(engine, job)
    finally:
        _unload(engine)
    return RefineResult(segments, complete, clock() - started)


def _format_eta(eta_sec: float) -> str:
    if eta_sec < 60:
        return f"残り約{max(0, int(round(eta_sec)))}秒"
    return f"残り約{int(round(eta_sec / 60))}分"


def format_refine_progress(done_sec: float, total_sec: float, eta_sec: float | None) -> str:
    """進捗の文言(GUI バナー・CLI 共通)。例: 高精度で再文字起こし中... 02:00 / 10:00（20%）"""
    percent = 0 if total_sec <= 0 else min(100, int(done_sec * 100 / total_sec))
    text = (
        f"高精度で再文字起こし中... {format_timestamp(done_sec)} / "
        f"{format_timestamp(total_sec)}（{percent}%）"
    )
    if eta_sec is None or done_sec <= 0:
        return text
    return f"{text} {_format_eta(eta_sec)}"
