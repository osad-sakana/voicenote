"""GUI ライブの停止後の高精度再文字起こし(Tkinter 非依存)

LiveWorkflow が速報版ノートを保存した後、REFINING 段階のスレッドで RefineRunner.run を呼ぶ。
RefineRunner は状態を持たず、ファイルへも書かない。
エンジンロックを取り(取れなければスキップ)、その下でリアルタイム用エンジンを解放してから
refine_wav を実行する。続いて高精度版の全文から要約を作り直し、上書き用の本文を返す。
上書きと状態遷移は、世代番号を確かめたうえで呼び出し側(LiveWorkflow)が行う。

文字起こし本文・ヒント・参加者・メモはログやメッセージに出さない(例外は型名だけ)。
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from live.asr_hint import build_asr_hint
from live.bootstrap import LiveSetupError, effective_engine_name
from live.context import LiveContext
from live.note import render_live_note
from live.refine import REFINE_ENGINE, RefineDecision, decide_refine
from live.session_models import LiveResult, LiveSettings, RefinedTranscript

from .live_state import LivePhase, set_phase
from .live_workflow_support import (
    BANNER_REFINE,
    PROGRESS_INTERVAL_SEC,
    external_send_banner_text,
    format_refine_progress,
)

_logger = logging.getLogger("voicenote")

REFINE_SUMMARY_TIMEOUT_SEC = 180.0  # 高精度版からの要約の上限
SKIP_HINT = "完了後に「高精度で文字起こし」から再文字起こしできます"


@dataclass(frozen=True)
class RefineJob:
    """1回の再文字起こしに必要な材料(速報版の結果と保存先を含む)"""

    settings: LiveSettings
    engine: Any  # SerializedEngine(unload_locked でリアルタイム用エンジンを解放する)
    backend: Any | None
    wav: Path
    note: Path
    result: LiveResult
    send_warning: str | None


@dataclass(frozen=True)
class RefineOutcome:
    """body があれば上書きする。message は利用者向けのログ(本文は含まない)"""

    body: str | None
    participants: tuple[str, ...]
    message: str


def plan_refine(settings: LiveSettings, engine, fetcher) -> RefineDecision:
    """停止前に、再文字起こしを行うか決める(実際に動いたエンジン・モデルの取得状況から)"""
    cached = settings.refine_enabled and _is_cached(settings.refine_model, fetcher)
    return decide_refine(settings, effective_engine_name(engine, settings), cached)


def _is_cached(model: str, fetcher) -> bool:
    try:
        repo = fetcher.resolve_repo(REFINE_ENGINE, model)
        return repo is None or fetcher.is_cached(repo)
    except Exception as e:
        _logger.warning("高精度モデルの取得状況を確認できません: %s", type(e).__name__)
        return False


class ProgressThrottle:
    """進捗を間引いてバナー文言にする(完了時は必ず通知)"""

    def __init__(self, clock: Callable[[], float], report: Callable[[str], None]):
        self._clock = clock
        self._report = report
        self._last_at: float | None = None

    def __call__(self, done: float, total: float, eta: float | None) -> None:
        now = self._clock()
        recent = self._last_at is not None and now - self._last_at < PROGRESS_INTERVAL_SEC
        if recent and done < total:
            return
        self._last_at = now
        self._report(format_refine_progress(done, total, eta))


class RefineRunner:
    """エンジンロック下で再文字起こし → 全文要約 → 上書き本文の作成を行う"""

    def __init__(
        self,
        deps,
        engine_lock: threading.Lock,
        *,
        lock_timeout: float,
        notify: Callable[[str], None],
    ):
        self._deps = deps
        self._engine_lock = engine_lock
        self._lock_timeout = lock_timeout
        self._notify = notify

    def run(
        self,
        job: RefineJob,
        cancel: threading.Event,
        on_progress: Callable[[float, float, float | None], None],
        get_context: Callable[[], LiveContext],
    ) -> RefineOutcome:
        """再文字起こしして上書き本文を返す。中断・失敗・時間切れなら body=None"""
        context = get_context()
        if not self._engine_lock.acquire(timeout=self._lock_timeout):
            return _keep(
                "前回の処理がエンジンを使用中のため、高精度の再文字起こしをスキップしました"
            )
        try:
            result = self._refine(job, context, cancel, on_progress)
        except Exception as e:
            # 例外のメッセージは文字起こし本文を含みうるため型名だけを出す
            _logger.error("高精度の再文字起こしに失敗しました: %s", type(e).__name__)
            return _keep(f"高精度の再文字起こしに失敗しました（{type(e).__name__}）")
        finally:
            self._engine_lock.release()
        if cancel.is_set():
            return _keep("高精度の再文字起こしを中断しました")
        if not result.complete:
            return _keep("高精度の再文字起こしが時間内に終わらなかったため、速報版のままにします")
        refined = RefinedTranscript(result.segments, job.settings.refine_model)
        return self._outcome(job, refined, cancel, get_context)

    def _refine(self, job: RefineJob, context: LiveContext, cancel, on_progress):
        """ロード・文字起こし・解放は同じスレッド(このスレッド)で行う"""
        settings = job.settings
        return self._deps.refine_fn(
            job.wav,
            make_engine=lambda: self._deps.create_engine(
                REFINE_ENGINE, settings.refine_model, notify=self._notify
            ),
            release_live=job.engine.unload_locked,
            language=settings.language,
            hint=build_asr_hint(context),
            cancel=cancel,
            on_progress=on_progress,
        )

    def _outcome(self, job: RefineJob, refined, cancel, get_context) -> RefineOutcome:
        summary = self._summarize(job, refined, cancel, get_context())
        context = get_context()  # 再文字起こし中の編集も含め、上書き時点の最新の値
        body = render_live_note(
            replace(job.result, final_summary=summary),
            job.wav,
            send_warning=job.send_warning,
            context=context,
            refined=refined,
        )
        return RefineOutcome(body, context.participants, "高精度版でライブノートを上書きしました")

    def _summarize(self, job: RefineJob, refined, cancel, context: LiveContext) -> str:
        """高精度版の全文から要約を作り直す。作れなければ速報版の要約"""
        if job.backend is None or not refined.segments:
            return job.result.final_summary
        try:
            summary = self._deps.full_summarizer(
                job.backend,
                tuple(s.text for s in refined.segments),
                context,
                deadline=time.monotonic() + REFINE_SUMMARY_TIMEOUT_SEC,
                cancel=cancel,
            )
        except Exception as e:
            _logger.warning("高精度版からの要約に失敗しました: %s", type(e).__name__)
            summary = None
        return summary or job.result.final_summary


def _keep(message: str) -> RefineOutcome:
    """速報版のノートを残す結果"""
    return RefineOutcome(None, (), f"{message}（速報版のノートを残しています。{SKIP_HINT}）")


class RefineLifecycle:
    """LiveWorkflow の REFINING 段階(LiveWorkflow が継承する mixin)

    LiveWorkflow の _lock / _state / _gen / _run / _closed / _deps / _callbacks / _clock /
    _engine_lock / _engine_lock_timeout / _drain_thread / _abandoned_drain / _live_settings /
    _live_engine / _backend / _refine_note と、_safe / _update / _banner / _finish_run /
    _ensure_model / context を使う。共有状態の読み書きはすべて _lock 下で行う。
    """

    def _ensure_refine_model(self, settings: LiveSettings, run) -> LiveSettings:
        """再文字起こしを予定していれば高精度モデルも取得する。失敗しても録音は続ける"""
        if not settings.refine_enabled or settings.engine != REFINE_ENGINE:
            return settings
        try:
            self._ensure_model(REFINE_ENGINE, settings.refine_model, run)
        except LiveSetupError as e:
            self._banner(run, BANNER_REFINE, f"{e} 高精度の再文字起こしなしで続行します。")
            return replace(settings, refine_enabled=False)
        return settings

    def _plan_refine(self) -> bool:
        """停止前に再文字起こしを行うか決める。行わない理由があればログで案内する"""
        with self._lock:
            settings, engine = self._live_settings, self._live_engine
        if settings is None:
            return False
        decision = plan_refine(settings, engine, self._deps.model_fetcher)
        if decision.reason:
            self._safe(self._callbacks.on_log, f"{decision.reason}。{SKIP_HINT}")
        return decision.run

    def _take_over_refine(self) -> tuple[Path, Path]:
        """再文字起こしを取り消して放棄する(ロック保持中に呼ぶ)。(wav, 速報版ノート) を返す

        処理中のスレッドは窓の区切りで止まるまで動き続けるため、_abandoned_drain として追跡し、
        終わるまで次の開始を拒否する(2モデルを同時に載せない)。
        """
        self._run.cancel.set()
        self._abandoned_drain = self._drain_thread
        self._drain_thread = None
        self._gen += 1  # 遅れて届く進捗・結果を捨てる
        return self._refine_note

    def _begin_refine(self, run, result: LiveResult, wav: Path, note: Path) -> bool:
        """REFINING に移って再文字起こしのスレッドを起動する。起動しなければ False"""
        config = run.config
        with self._lock:
            if self._closed or self._run is not run or run.gen != self._gen:
                return False
            job = RefineJob(
                self._live_settings,
                self._live_engine,
                self._backend,
                wav,
                note,
                result,
                external_send_banner_text(config.transcription_mode, config.format_mode),
            )
            try:
                thread = self._thread_factory(
                    target=self._refine_worker, args=(run, job), daemon=True, name="live-refine"
                )
            except Exception:
                _logger.error("再文字起こしの起動に失敗:\n%s", traceback.format_exc())
                return False
            self._refine_note = (wav, note)
            self._drain_thread = thread
            self._update(set_phase, LivePhase.REFINING)
        self._safe(self._callbacks.on_log, "高精度モデルで録音全体を再文字起こししています...")
        thread.start()
        return True

    def _refine_worker(self, run, job: RefineJob) -> None:
        """再文字起こし → 世代を確かめて SAVING に移る → 上書き → IDLE → on_done"""
        runner = RefineRunner(
            self._deps,
            self._engine_lock,
            lock_timeout=self._engine_lock_timeout,
            notify=lambda msg: self._safe(self._callbacks.on_log, msg),
        )
        progress = ProgressThrottle(self._clock, lambda t: self._banner(run, BANNER_REFINE, t))
        outcome = runner.run(job, run.cancel, progress, lambda: self.context)
        with self._lock:
            if run.gen != self._gen or self._state.phase is not LivePhase.REFINING:
                return  # 中断済み(速報版で完了通知済み)
            self._drain_thread = None
            self._update(set_phase, LivePhase.SAVING)  # 以降は中断を受け付けない
        message = self._overwrite(job, outcome)
        self._safe(self._callbacks.on_log, message)
        self._finish_run(run)
        self._safe(self._callbacks.on_done, job.wav, job.note)

    def _overwrite(self, job: RefineJob, outcome: RefineOutcome) -> str:
        """上書き本文があれば速報版ノートを原子的に上書きする。ログ文言を返す"""
        if outcome.body is None:
            return outcome.message
        try:
            self._deps.note_overwriter(
                job.note, outcome.body, participants=outcome.participants, refined=True
            )
        except Exception as e:
            _logger.error("高精度版での上書きに失敗: %s", type(e).__name__)
            return f"高精度版でノートを上書きできませんでした（速報版は残っています）: {e}"
        return outcome.message
