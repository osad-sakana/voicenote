"""GUI ライブ文字起こしのライフサイクルとスレッド管理(Tkinter 非依存)

状態遷移: IDLE → PREPARING → RECORDING → FINALIZING → DRAINING → SAVING → IDLE
(PREPARING の失敗・取り消しと、DRAINING の中断はそれぞれ IDLE / SAVING へ分岐する)
停止後の高精度再文字起こしを行う場合は、速報版ノートの保存後に SAVING → REFINING → SAVING → IDLE
(REFINING の中断は待たずに IDLE。on_done は最後に1回だけ、速報版または上書き後のノートで呼ぶ)

停止順序の不変条件「WAV確定 → ストリーム停止 → ライブのドレイン」: finalize_recording() の後で
stop_with_timeout()、session.stop() はさらに別スレッドで実行する。ドレインはコア側でキャンセル
できないため、中断時は待ちを放棄し GUI 側が集めた区間と要約から部分ノート(complete=False)を保存する。
GUI は record_audio() を使えない(SIGINT ハンドラはメインスレッド専用)ため ThreadedRecorder を直接使う。

表示状態(LiveViewState)はロック下で gui.live_state の純関数により更新し、on_state で通知する。
コールバックはワーカースレッドから(ロック保持中にも)呼ばれるため、App は各コールバックを
ThreadSafeUIQueue.submit でラップして渡すこと(RecordingWorkflow と同じ方式)。
放棄した実行から遅れて届く結果・通知は、実行ごとの世代番号で捨てる。
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from config import VoiceNoteConfig
from live.bootstrap import LiveSetupError, build_engine, build_vad, check_backend
from live.context import LiveContext
from live.level_meter import InputLevelAssessment, LevelMeter, assess_input_level, tee
from live.model_fetch import describe_fetch_error
from live.note import render_live_note
from live.refine_fetch import BackgroundModelFetch
from live.session_models import LiveCallbacks, LiveResult, LiveSettings
from recorder import STREAM_STOP_TIMEOUT_SEC

from .live_state import (
    LivePhase,
    LiveViewState,
    add_banner,
    apply_segment,
    apply_status,
    apply_summary,
    apply_warning,
    clear_banner,
    set_phase,
    to_live_result,
)
from .live_workflow_refine import RefineLifecycle
from .live_workflow_support import (
    BANNER_ENGINE_FALLBACK,
    BANNER_EXTERNAL_SEND,
    BANNER_LOW_INPUT,
    BANNER_MODEL_DOWNLOAD,
    BANNER_OLLAMA,
    BANNER_REFINE,
    BANNER_VAD_FALLBACK,
    ENGINE_LOCK_TIMEOUT_SEC,
    PROGRESS_INTERVAL_SEC,
    LiveDeps,
    LiveWorkflowCallbacks,
    MeterReading,
    ModelFetcher,
    external_send_banner_text,
    format_download_progress,
    format_refine_progress,
    validate_live_start,
)
from .live_workflow_support import Cancelled as _Cancelled
from .live_workflow_support import DownloadProgress as _DownloadProgress
from .live_workflow_support import Run as _Run
from .live_workflow_support import SerializedEngine as _SerializedEngine
from .live_workflow_support import always_allowed as _always_allowed
from .live_workflow_support import coerce_context as _coerce_context
from .live_workflow_support import dropped_chunks as _dropped_chunks
from .live_workflow_support import ollama_banner_text as _ollama_banner_text
from .live_workflow_support import push_context as _push_context

__all__ = [
    "BANNER_ENGINE_FALLBACK",
    "BANNER_EXTERNAL_SEND",
    "BANNER_LOW_INPUT",
    "BANNER_MODEL_DOWNLOAD",
    "BANNER_OLLAMA",
    "BANNER_REFINE",
    "BANNER_VAD_FALLBACK",
    "PROGRESS_INTERVAL_SEC",
    "LiveDeps",
    "LiveWorkflow",
    "LiveWorkflowCallbacks",
    "MeterReading",
    "ModelFetcher",
    "external_send_banner_text",
    "format_download_progress",
    "format_refine_progress",
    "validate_live_start",
]

_logger = logging.getLogger("voicenote")

SESSION_STOP_TIMEOUT_SEC = (
    120.0  # ドレイン(残り区間の文字起こし)の上限。最終要約は量に比例した別上限
)
ERROR_STOP_TIMEOUT_SEC = 2.0  # 失敗・取り消し時にセッションを解放する際の上限
SHUTDOWN_FINALIZE_TIMEOUT_SEC = 1.0  # ウィンドウを閉じる時の WAV 確定の上限
SHUTDOWN_JOIN_SEC = 1.0  # ウィンドウを閉じる時にストリーム停止を待つ猶予
PROBE_TIMEOUT_SEC = 3.0
STREAM_CLOSE_GRACE_FACTOR = 3  # 放棄ストリームの解放を stop_timeout の何倍まで待つか
MLX_INSTALL_HINT = "uv sync --extra live-mlx"


class LiveWorkflow(RefineLifecycle):
    """ライブ録音の準備 → 録音 → WAV確定 → ドレイン → ノート保存を管理する

    停止後の高精度再文字起こし(REFINING)の段階は RefineLifecycle(live_workflow_refine)にある。
    """

    def __init__(
        self,
        config: VoiceNoteConfig,
        callbacks: LiveWorkflowCallbacks,
        deps: LiveDeps | None = None,
        *,
        can_start: Callable[[], bool] = _always_allowed,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
        clock: Callable[[], float] = time.monotonic,
        stop_timeout: float = STREAM_STOP_TIMEOUT_SEC,
        session_stop_timeout: float = SESSION_STOP_TIMEOUT_SEC,
        engine_lock_timeout: float = ENGINE_LOCK_TIMEOUT_SEC,
    ):
        """config は開始時点の値をその実行の間使い続ける。can_start はマイク排他の判定
        (App の busy・通常録音の状態)で False なら開始を拒否する。thread_factory は
        start / join / is_alive を持つ threading.Thread 互換。clock は警告・進捗の間引きと
        放棄ストリームの経過時間に使う。engine_lock_timeout は load() が前回の実行の
        transcribe 完了を待つ上限(超えたら load 失敗として警告し、録音は継続する)。"""
        self._config = config
        self._callbacks = callbacks
        self._deps = deps or LiveDeps()
        self._can_start = can_start
        self._thread_factory = thread_factory
        self._clock = clock
        self._stop_timeout = stop_timeout
        self._session_stop_timeout = session_stop_timeout
        self._engine_lock_timeout = engine_lock_timeout
        self._lock = threading.RLock()
        self._engine_lock = threading.Lock()  # 実行をまたいで asr を直列化する
        # 高精度モデルのバックグラウンド取得(実行をまたいで二重取得を避ける)
        self._refine_fetch = BackgroundModelFetch(
            thread_factory=lambda **kw: self._thread_factory(**kw)
        )
        self._state = LiveViewState()
        self._gen = 0
        self._run: _Run | None = None
        self._session = None
        self._recorder = None
        self._meter: LevelMeter | None = None
        self._drain_wav: Path | None = None
        self._drain_thread = None
        self._abandoned_drain = None
        self._pending_close_recorder = None
        self._pending_close_since: float | None = None
        self._closed = False
        # 停止後の再文字起こしに使う、実行中の設定・エンジン・要約バックエンドと保存済みのノート
        self._live_settings: LiveSettings | None = None
        self._live_engine = None
        self._backend = None
        self._refine_note: tuple[Path, Path] | None = None  # (wav, 速報版ノート)
        # 参加者・メモ。実行をまたいで保持し(IDLE でも次回の開始で使う)、保存時は最新の値を使う
        self._context = LiveContext()

    @property
    def state(self) -> LiveViewState:
        """現在の表示状態(イミュータブル)"""
        with self._lock:
            return self._state

    @property
    def phase(self) -> LivePhase:
        with self._lock:
            return self._state.phase

    @property
    def is_active(self) -> bool:
        """ライブ処理中、または放棄したドレインのスレッドがまだ生きていれば True"""
        with self._lock:
            return self._state.phase is not LivePhase.IDLE or self._abandoned_alive()

    @property
    def context(self) -> LiveContext:
        """現在の参考情報(参加者・メモ)"""
        with self._lock:
            return self._context

    def update_context(self, context: LiveContext | None) -> None:
        """参考情報(参加者・メモ)を差し替える。どの段階で呼んでも安全

        セッションがあれば次回の要約から反映し(変更だけでは要約を起動しない)、
        ノートには保存時点の最新の値を書く。IDLE 中の値は次回の開始時に使う。None は空。
        """
        validated = _coerce_context(context)
        with self._lock:
            self._context = validated
            session = self._session
            # 呼び出し順のまま反映されるようロック下で渡す(set_context は短時間で戻る)
            if session is not None:
                _push_context(session, validated)

    def update_config(self, config: VoiceNoteConfig) -> None:
        """設定を差し替える(実行中のライブには影響せず、次回の開始から使う)"""
        with self._lock:
            self._config = config

    def start(self, device_id: int | None, device_label: str, rec_dest: Path) -> str | None:
        """ライブ録音の準備をバックグラウンドで始める。開始できなければエラーメッセージを返す"""
        with self._lock:
            error = self._start_rejection()
            if error is not None:
                return error
            self._gen += 1
            run = _Run(
                self._gen, self._config, device_id, device_label, Path(rec_dest), threading.Event()
            )
            self._run = run
            self._state = LiveViewState()
            self._update(set_phase, LivePhase.PREPARING)
            warning = external_send_banner_text(
                run.config.transcription_mode, run.config.format_mode
            )
            if warning is not None:
                self._update(add_banner, BANNER_EXTERNAL_SEND, warning)
        try:
            self._spawn(self._prepare_worker, run, name="live-prepare")
        except Exception:
            _logger.error("ライブ準備ワーカーの起動に失敗:\n%s", traceback.format_exc())
            self._finish_run(run)
            return "ライブ文字起こしを開始できませんでした（準備処理を起動できません）"
        return None

    def stop(self) -> None:
        """録音中なら停止処理をバックグラウンドで始める。準備中なら開始を取り消す"""
        with self._lock:
            run = self._run
            phase = self._state.phase
            if run is None:
                return
            if phase is LivePhase.PREPARING:
                run.cancel.set()
            elif phase is LivePhase.RECORDING:
                recorder, session = self._recorder, self._session
                self._recorder = None
                self._update(set_phase, LivePhase.FINALIZING)
            else:
                return
        if phase is LivePhase.PREPARING:
            self._safe(
                self._callbacks.on_log,
                "ライブ文字起こしの開始を取り消しています（モデル取得中は取得完了後に止まります）",
            )
            return
        self._safe(self._callbacks.on_log, "録音停止 → 録音データを確定しています...")
        try:
            self._spawn(self._stop_worker, run, recorder, session, name="live-stop")
        except Exception:
            _logger.error("ライブ停止ワーカーの起動に失敗:\n%s", traceback.format_exc())
            self._emergency_stop(run, recorder, session)

    def abort_drain(self) -> bool:
        """ドレインの待ちを放棄し、ここまでの区間と要約で部分ノートを保存する

        REFINING 中なら再文字起こしを取り消し、待たずに速報版ノートで完了する。

        Returns:
            DRAINING / REFINING 中で中断を受け付けたら True
        """
        with self._lock:
            run = self._run
            if run is None:
                return False
            if self._state.phase is LivePhase.REFINING:
                wav, note = self._take_over_refine()
                refining = True
            elif self._state.phase is LivePhase.DRAINING:
                wav, session, state = self._take_over_drain()
                refining = False
            else:
                return False
        if refining:
            self._safe(
                self._callbacks.on_log, "高精度の再文字起こしを中断しました（速報版を残します）"
            )
            self._finish_run(run)
            self._safe(self._callbacks.on_done, wav, note)
            return True
        self._safe(self._callbacks.on_log, "ドレインを中断し、ここまでの内容でノートを保存します")
        result = to_live_result(state, dropped_chunks=_dropped_chunks(session), complete=False)
        try:
            self._spawn(self._save_note, run, result, wav, name="live-save")
        except Exception:
            self._save_note(run, result, wav)
        return True

    def poll_meter(self) -> MeterReading | None:
        """録音中の入力レベル・文字起こしの遅れ・破棄チャンク数を読む(App が定期的に呼ぶ)

        入力が小さい判定になれば low_input バナーを出し、回復したら消す。録音中以外は None
        """
        with self._lock:
            run, meter, session = self._run, self._meter, self._session
            if self._state.phase is not LivePhase.RECORDING or meter is None or session is None:
                return None
        reading = MeterReading(
            level_db=meter.level_db,
            backlog_sec=float(session.backlog_sec),
            dropped_chunks=_dropped_chunks(session),
            input_level=assess_input_level(meter.history_db),
        )
        self._sync_low_input_banner(run, reading.input_level)
        return reading

    def shutdown(self) -> None:
        """ウィンドウを閉じる時に呼ぶ。WAV を確定し、集めた分で部分ノートを同期保存する

        ストリームとセッションの停止は daemon スレッドに任せ、短い猶予だけ待って戻る。
        submit 経由でまだ UI キューに残っている通知は部分ノートに含まれない(best-effort)。
        """
        with self._lock:
            self._closed = True
            run = self._run
            if run is None:
                return
            run.cancel.set()
            phase = self._state.phase
            if phase is LivePhase.RECORDING:
                recorder, session = self._recorder, self._session
                self._recorder = None
                self._gen += 1  # 以降のセッション通知は捨てる
                state = self._state
            elif phase is LivePhase.DRAINING or (
                # FINALIZING でも WAV 確定後なら(ストリーム停止のハング中など)部分ノートを残す
                phase is LivePhase.FINALIZING and self._drain_wav is not None
            ):
                wav, session, state = self._take_over_drain()
            elif phase is LivePhase.REFINING:
                self._take_over_refine()  # 速報版ノートは保存済み。再文字起こしは取り消す
            else:
                return  # PREPARING は取り消し済み、それ以外はワーカーが完走する
        if phase is LivePhase.RECORDING:
            self._shutdown_recording(run, recorder, session, state)
        elif phase is not LivePhase.REFINING:
            self._write_partial_note(run, state, session, wav)
        self._finish_run(run)

    def _guarded(self, what: str, fn: Callable, *args):
        """fn を呼んで結果を返す。例外は型名だけをログに残して None を返す(録音・保存を続ける)"""
        try:
            return fn(*args)
        except Exception as e:
            _logger.error("%s: %s", what, type(e).__name__)
            self._safe(self._callbacks.on_log, f"{what}（{type(e).__name__}）")
            return None

    def _safe(self, fn: Callable, *args) -> None:
        """コールバックを呼ぶ。UI 側の例外でワーカーを止めない"""
        try:
            fn(*args)
        except Exception:
            _logger.exception("ライブ文字起こしの UI コールバックでエラーが発生しました")

    def _update(self, reducer: Callable[..., LiveViewState], *args) -> None:
        """純関数で状態を更新し、変化があれば on_state で通知する"""
        with self._lock:
            new_state = reducer(self._state, *args)
            if new_state is self._state:
                return
            self._state = new_state
            self._safe(self._callbacks.on_state, new_state)

    def _update_for(self, run: _Run, reducer: Callable[..., LiveViewState], *args) -> None:
        """run が現行の実行である場合だけ状態を更新する(放棄した実行の通知を捨てる)"""
        with self._lock:
            if run.gen != self._gen:
                return
            self._update(reducer, *args)

    def _banner(self, run: _Run, kind: str, text: str) -> None:
        self._update_for(run, add_banner, kind, text)

    def _spawn(self, target: Callable, *args, name: str):
        thread = self._thread_factory(target=target, args=args, daemon=True, name=name)
        thread.start()
        return thread

    def _abandoned_alive(self) -> bool:
        thread = self._abandoned_drain
        if thread is None:
            return False
        if thread.is_alive():
            return True
        self._abandoned_drain = None
        return False

    def _start_rejection(self) -> str | None:
        """開始を拒否する理由(ロック保持中に呼ぶ)"""
        if self._closed:
            return "アプリを終了しています"
        if self._state.phase is not LivePhase.IDLE:
            return "ライブ文字起こしは既に実行中です"
        if self._abandoned_alive():
            return (
                "前回のライブ文字起こしの後処理がまだ終わっていません。"
                "しばらく待ってから再試行してください"
            )
        if not self._can_start():
            return "他の録音・文字起こしが実行中です。終了してから開始してください"
        return self._pending_close_rejection()

    def has_pending_close(self) -> bool:
        """タイムアウトで放棄したストリームが猶予内でまだ解放中なら True(状態は変えない)

        通常録音側の開始判定から参照する(ライブ自身の開始判定は _pending_close_rejection)。
        """
        with self._lock:
            recorder = self._pending_close_recorder
            if recorder is None:
                return False
            return recorder.is_closing() and not self._close_grace_expired()

    def _close_grace_expired(self) -> bool:
        """放棄ストリームの猶予を過ぎたか(IDLE に戻る前は経過0秒とみなす。ロック保持中に呼ぶ)"""
        return self._pending_close_elapsed() >= self._stop_timeout * STREAM_CLOSE_GRACE_FACTOR

    def _pending_close_elapsed(self) -> float:
        since = self._pending_close_since
        return 0.0 if since is None else self._clock() - since

    def _pending_close_rejection(self) -> str | None:
        """タイムアウトで放棄したストリームがまだ解放中なら拒否する(猶予を過ぎたら諦めて許可)"""
        recorder = self._pending_close_recorder
        if recorder is None:
            return None
        elapsed = self._pending_close_elapsed()
        closing = recorder.is_closing()
        if closing and not self._close_grace_expired():
            return "前回のストリームがまだ解放されていません。しばらく待ってから再試行してください"
        if closing:
            _logger.warning("放棄したストリームが %.1f 秒経っても解放されません", elapsed)
            self._safe(
                self._callbacks.on_log,
                f"前回のストリームが {elapsed:.0f}秒経っても解放されないため、録音を再開します",
            )
        self._pending_close_recorder = None
        self._pending_close_since = None
        return None

    def _finish_run(self, run: _Run) -> None:
        """実行を終えて IDLE に戻す(別の実行に切り替わっていれば何もしない)"""
        with self._lock:
            if self._run is not run:
                return
            self._run = None
            self._session = None
            self._recorder = None
            self._meter = None
            self._drain_wav = None
            self._drain_thread = None
            self._live_settings = self._live_engine = self._backend = None
            self._refine_note = None
            self._update(clear_banner, BANNER_REFINE)
            if self._pending_close_recorder is not None and self._pending_close_since is None:
                self._pending_close_since = self._clock()
            self._update(set_phase, LivePhase.IDLE)

    def _take_over_drain(self):
        """ドレインを放棄して SAVING に移る(ロック保持中に呼ぶ)。(wav, session, state) を返す"""
        wav, session = self._drain_wav, self._session
        self._abandoned_drain = self._drain_thread
        self._drain_thread = None
        self._drain_wav = None
        self._gen += 1  # 放棄したドレインの結果・通知を捨てる
        self._update(set_phase, LivePhase.SAVING)
        return wav, session, self._state

    def _sync_low_input_banner(self, run: _Run | None, assessment: InputLevelAssessment) -> None:
        if run is None:
            return
        shown = any(b.kind == BANNER_LOW_INPUT for b in self.state.banners)
        if assessment.is_low and not shown:
            self._banner(run, BANNER_LOW_INPUT, assessment.message)
        elif assessment.status == "ok" and shown:
            self._update_for(run, clear_banner, BANNER_LOW_INPUT)

    def _prepare_worker(self, run: _Run) -> None:
        try:
            self._prepare(run)
        except _Cancelled:
            self._fail_prepare(run, None)
        except LiveSetupError as e:
            self._fail_prepare(run, str(e))
        except Exception as e:
            _logger.error("ライブ準備で未捕捉の例外:\n%s", traceback.format_exc())
            self._fail_prepare(run, f"ライブ文字起こしを開始できませんでした: {e}")

    def _prepare(self, run: _Run) -> None:
        """エンジン → モデル取得 → LLM → VAD → セッション → 録音 の順に準備する

        モデルは build_engine のフォールバック後の(実際に使う)エンジン・モデルについて取得する。
        """
        try:
            settings = LiveSettings.from_config(run.config)
        except ValueError as e:
            raise LiveSetupError(f"ライブ文字起こしの設定が不正です: {e}") from e
        engine, settings = self._build_engine(settings, run)
        self._check_cancel(run)
        self._ensure_model(settings.engine, settings.model, run)
        self._check_cancel(run)
        backend = self._prepare_backend(settings, run)
        detector = build_vad(
            settings,
            create_vad=self._deps.create_vad,
            vad_kind=self._deps.vad_kind,
            warn=lambda msg: self._banner(run, BANNER_VAD_FALLBACK, msg),
        )
        self._check_cancel(run)
        context = self.context
        serialized = _SerializedEngine(
            engine, self._engine_lock, load_timeout=self._engine_lock_timeout
        )
        session = self._deps.session_factory(
            settings,
            engine=serialized,
            vad=detector,
            backend=backend,
            callbacks=self._session_callbacks(run),
            context=context,
        )
        meter = self._deps.meter_factory()
        with self._lock:
            self._session = session
            self._meter = meter
            self._live_settings, self._live_engine, self._backend = settings, serialized, backend
            if self._context is not context:  # セッション生成中に更新された
                _push_context(session, self._context)
        # engine.load() は session の asr スレッドで行う(mlx は load と transcribe を同じスレッドで)
        session.start()
        self._check_cancel(run)
        sink = tee(meter.feed, session.feed)
        self._start_recorder(run, self._deps.recorder_factory(run.device_id, run.dest, sink))
        self._safe(self._callbacks.on_recording_started)
        # 録音開始後に取得する(録音の開始を待たせない)。ここで例外を出すと録音中のマイクを
        # 開いたまま準備失敗扱いになるため、取得の失敗は必ずここで止める
        self._guarded("高精度モデルの取得を開始できません", self._start_refine_fetch, settings)
        self._safe(self._callbacks.on_log, f"ライブ録音開始 (デバイス: {run.device_label})")

    @staticmethod
    def _check_cancel(run: _Run) -> None:
        if run.cancel.is_set():
            raise _Cancelled()

    def _start_recorder(self, run: _Run, recorder) -> None:
        """取り消されていなければ録音を始める。shutdown と競合しないようロック下で行う"""
        with self._lock:
            self._check_cancel(run)
            recorder.start()
            self._recorder = recorder
            self._update(set_phase, LivePhase.RECORDING)

    def _fail_prepare(self, run: _Run, message: str | None) -> None:
        """準備の失敗・取り消し。起動済みセッションを解放して IDLE に戻す

        session.stop() がタイムアウトしてもセッションのスレッドは動き続けるため、世代を進めて
        遅れて届く on_status / on_warning などを捨てる(_take_over_drain と同じ方式)。
        """
        with self._lock:
            current = self._run is run
            session = self._session if current else None
            if current:
                self._gen += 1
        if session is not None:
            with contextlib.suppress(Exception):
                session.stop(timeout=ERROR_STOP_TIMEOUT_SEC)
        self._finish_run(run)
        if message is None:
            self._safe(self._callbacks.on_log, "ライブ文字起こしの開始を取り消しました")
        else:
            self._safe(self._callbacks.on_error, message)

    def _build_engine(self, settings: LiveSettings, run: _Run):
        return build_engine(
            settings,
            create_engine=self._deps.create_engine,
            unavailable_error=self._deps.unavailable_error,
            notify=lambda msg: self._safe(self._callbacks.on_log, msg),
            warn=lambda msg: self._banner(run, BANNER_ENGINE_FALLBACK, msg),
            install_hint=MLX_INSTALL_HINT,
        )

    def _ensure_model(self, engine_name: str, model: str, run: _Run) -> None:
        """エンジン・モデルが未取得ならダウンロードする。失敗は LiveSetupError"""
        fetcher = self._deps.model_fetcher
        try:
            repo = fetcher.resolve_repo(engine_name, model)
        except ValueError as e:
            raise LiveSetupError(str(e)) from e
        if repo is None or fetcher.is_cached(repo):
            return
        self._banner(run, BANNER_MODEL_DOWNLOAD, format_download_progress(model, 0, 0))
        progress = _DownloadProgress(
            model, self._clock, lambda text: self._banner(run, BANNER_MODEL_DOWNLOAD, text)
        )
        try:
            fetcher.fetch(repo, progress)
        except Exception as e:
            _logger.error("モデルの取得に失敗:\n%s", traceback.format_exc())
            info = describe_fetch_error(e, required_bytes=progress.total or None)
            raise LiveSetupError(info.message) from e
        finally:
            self._update_for(run, clear_banner, BANNER_MODEL_DOWNLOAD)
        self._safe(self._callbacks.on_log, f"音声認識モデル {model} を取得しました")

    def _prepare_backend(self, settings: LiveSettings, run: _Run):
        """ローカルLLM の疎通を確認する。使えなければバナーを出し、要約なしで続行する"""
        check = check_backend(
            settings,
            client_factory=self._deps.client_factory,
            probe_fn=self._deps.probe_fn,
            probe_timeout=PROBE_TIMEOUT_SEC,
        )
        if check.status == "disabled":
            return None
        if check.status == "ok":
            self._safe(self._callbacks.on_log, check.message)
            return check.backend
        self._banner(run, BANNER_OLLAMA, _ollama_banner_text(check))
        return None

    def _session_callbacks(self, run: _Run) -> LiveCallbacks:
        return LiveCallbacks(
            on_segment=lambda index, start, end, text: self._update_for(
                run, apply_segment, index, start, end, text
            ),
            on_summary=lambda text: self._update_for(run, apply_summary, text),
            on_status=lambda msg: self._update_for(run, apply_status, msg),
            on_warning=lambda msg: self._update_for(run, apply_warning, msg, self._clock()),
        )

    def _stop_worker(self, run: _Run, recorder, session) -> None:
        """WAV 確定 → ストリーム停止 → ドレイン開始。WAV 確定はライブの待ちより必ず前"""
        try:
            wav = recorder.finalize_recording()
            if wav is None:
                _logger.warning("ライブ録音: 録音データがありません")
                self._close_stream(recorder)
                with contextlib.suppress(Exception):
                    session.stop(timeout=ERROR_STOP_TIMEOUT_SEC)
                self._finish_run(run)
                self._safe(self._callbacks.on_error, "エラー: 録音データがありません")
                return
            with self._lock:
                if self._run is run:
                    self._drain_wav = wav  # shutdown がストリーム停止中でも部分ノートを残せるよう
            self._safe(self._callbacks.on_wav_saved, wav)
            self._safe(self._callbacks.on_log, f"音声ファイルを保存: {wav.name}")
            self._close_stream(recorder)
            self._begin_drain(run, session, wav)
        except Exception:
            _logger.error("ライブ停止ワーカーで未捕捉の例外:\n%s", traceback.format_exc())
            self._finish_run(run)
            self._safe(self._callbacks.on_error, "予期せぬエラーが発生しました（停止処理）")

    def _close_stream(self, recorder) -> None:
        """ストリームを止める。タイムアウトしたら放棄ストリームとして追跡する"""
        if recorder.stop_with_timeout(self._stop_timeout):
            return
        _logger.warning("ストリームの停止がタイムアウトしました（%s秒）", self._stop_timeout)
        self._safe(
            self._callbacks.on_log,
            "ストリームの停止がタイムアウトしました。録音データは既に保存済みです",
        )
        with self._lock:
            self._pending_close_since = None  # IDLE に戻った時点から猶予を数える
            self._pending_close_recorder = recorder

    def _begin_drain(self, run: _Run, session, wav: Path) -> None:
        with self._lock:
            if self._run is not run or run.gen != self._gen:
                return  # shutdown が引き取った
            self._drain_wav = wav
            self._update(set_phase, LivePhase.DRAINING)
            thread = self._thread_factory(
                target=self._drain_worker, args=(run, session, wav), daemon=True, name="live-drain"
            )
            self._drain_thread = thread
        self._safe(self._callbacks.on_log, "残りの区間を文字起こしし、最終要約を作成しています...")
        thread.start()

    def _drain_worker(self, run: _Run, session, wav: Path) -> None:
        """session.stop() でドレインし、ノートを保存する。中断されていれば結果を捨てる

        再文字起こしを行うかは、ドレイン後(実際に動いたエンジンが確定した後)に決める。
        """
        try:
            # 全文要約まで作る(stop の中で終わるか打ち切る。上限は量に比例、「中断して保存」で放棄可)
            result: LiveResult | None = session.stop(timeout=self._session_stop_timeout)
        except Exception:
            _logger.error("session.stop で例外:\n%s", traceback.format_exc())
            result = None
        with self._lock:
            if run.gen != self._gen or self._state.phase is not LivePhase.DRAINING:
                return  # 中断済み(部分ノートは保存済み)
            self._drain_thread = None
            self._drain_wav = None
            self._update(set_phase, LivePhase.SAVING)
            state = self._state
        if result is None:
            result = to_live_result(state, dropped_chunks=_dropped_chunks(session), complete=False)
        # 判定に失敗しても SAVING に残さず、速報版の保存へ進む
        refine = self._guarded("再文字起こしの判定に失敗しました", self._plan_refine) is True
        self._save_note(run, result, wav, refine=refine)

    def _render_and_save(self, run: _Run, result: LiveResult, wav: Path) -> Path:
        config = run.config
        warning = external_send_banner_text(config.transcription_mode, config.format_mode)
        context = self.context  # ドレイン中・終了直前の更新も含め、保存時点の最新の値
        body = render_live_note(result, wav, send_warning=warning, context=context)
        folder = Path(config.save_folder).expanduser()
        return self._deps.note_saver(
            folder, body, audio_file=wav, participants=context.participants
        )

    def _save_note(self, run: _Run, result: LiveResult, wav: Path, refine: bool = False) -> None:
        """ライブノートを保存して IDLE に戻り、on_done / on_error を通知する

        refine なら保存した速報版ノートの高精度な上書きへ進む(on_done はその後に1回だけ)。
        """
        try:
            note = self._render_and_save(run, result, wav)
        except Exception as e:
            _logger.error("ライブノートの保存に失敗:\n%s", traceback.format_exc())
            self._finish_run(run)
            self._safe(
                self._callbacks.on_error,
                f"ライブノートの保存に失敗しました: {e}（録音は {wav} に保存済みです）",
            )
            return
        self._safe(self._callbacks.on_log, f"ライブノートを保存: {note}")
        if refine and self._begin_refine(run, result, wav, note):
            return
        self._finish_run(run)
        self._safe(self._callbacks.on_done, wav, note)

    def _emergency_stop(self, run: _Run, recorder, session) -> None:
        """停止ワーカーを起動できない場合の最終手段。WAV 確定とマイク停止を同期で行う"""
        wav = None
        with contextlib.suppress(Exception):
            wav = recorder.finalize_recording()
        with contextlib.suppress(Exception):
            recorder.stop_with_timeout(self._stop_timeout)
        with contextlib.suppress(Exception):
            session.stop(timeout=0.0)
        self._finish_run(run)
        saved = f"（録音は {wav} に保存済みです）" if wav is not None else ""
        self._safe(self._callbacks.on_error, f"予期せぬエラーが発生しました（停止処理）{saved}")

    def _shutdown_recording(self, run: _Run, recorder, session, state: LiveViewState) -> None:
        wav = None
        with contextlib.suppress(Exception):
            wav = recorder.finalize_recording(timeout=SHUTDOWN_FINALIZE_TIMEOUT_SEC)
        if wav is not None:
            _logger.info("終了時にライブ録音を確定しました: %s", wav)
            self._write_partial_note(run, state, session, wav)

        def _stop() -> None:
            with contextlib.suppress(Exception):
                recorder.stop_with_timeout(self._stop_timeout)
            with contextlib.suppress(Exception):
                session.stop(timeout=ERROR_STOP_TIMEOUT_SEC)

        with contextlib.suppress(Exception):
            stopper = self._thread_factory(target=_stop, daemon=True, name="live-shutdown")
            stopper.start()
            stopper.join(SHUTDOWN_JOIN_SEC)

    def _write_partial_note(self, run: _Run, state: LiveViewState, session, wav) -> None:
        """終了時の部分ノートを同期保存する(失敗はログのみ)"""
        if wav is None:
            return
        result = to_live_result(state, dropped_chunks=_dropped_chunks(session), complete=False)
        try:
            note = self._render_and_save(run, result, wav)
        except Exception:
            _logger.error("終了時のライブノート保存に失敗:\n%s", traceback.format_exc())
            return
        _logger.info("終了時にライブノートを保存しました: %s", note)
