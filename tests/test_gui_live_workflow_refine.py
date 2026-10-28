"""gui/live_workflow.py のテスト: 停止後の高精度再文字起こし(REFINING)

Fake 群と Harness は tests/live_workflow_fakes.py にある。refine_wav・全文要約・ノート上書きも
Fake に差し替え、DeferredThread で REFINING 段階を止めて検証する。
"""

from __future__ import annotations

import threading

import pytest

from config import VoiceNoteConfig
from gui.live_state import LivePhase
from gui.live_workflow import BANNER_REFINE
from tests.live_workflow_fakes import (
    DeferredThread,
    FakeFetcher,
    FakeRefine,
    Harness,
    ImmediateThread,
    reset_deferred_threads,
    run_named,
)


@pytest.fixture(autouse=True)
def _reset_deferred():
    reset_deferred_threads()
    yield
    reset_deferred_threads()


def _config(tmp_path, **overrides) -> VoiceNoteConfig:
    base = dict(
        save_folder=str(tmp_path / "notes"),
        live_engine="mlx",
        live_whisper_model="large-v3-turbo",
        live_vad="silero",
        live_refine_enabled=True,
    )
    return VoiceNoteConfig(**{**base, **overrides})


def _harness(tmp_path, **kwargs) -> Harness:
    config = kwargs.pop("config", None) or _config(tmp_path)
    return Harness(tmp_path, config=config, **kwargs)


def _phases(states) -> list[LivePhase]:
    phases: list[LivePhase] = []
    for state in states:
        if not phases or phases[-1] is not state.phase:
            phases.append(state.phase)
    return phases


def _stop_until_refining(h: Harness) -> DeferredThread:
    """停止 → ドレイン → 速報版保存まで進め、REFINING のスレッドを返す(未実行)"""
    h.wf._thread_factory = DeferredThread
    h.wf.stop()
    DeferredThread.instances[0].run()  # 停止ワーカー
    DeferredThread.instances[1].run()  # ドレイン(速報版を保存し、再文字起こしを起動)
    assert h.state.phase is LivePhase.REFINING
    return DeferredThread.instances[2]


def _fast_note(h: Harness):
    return h.tmp_path / "notes" / "note0_live.md"


class TestRefineFlow:
    def test_transitions_and_on_done_once(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        h.spy.states.clear()
        h.wf.stop()
        phases = _phases(h.spy.states)
        assert phases[:2] == [LivePhase.FINALIZING, LivePhase.DRAINING]
        assert LivePhase.REFINING in phases
        assert phases.index(LivePhase.REFINING) > phases.index(LivePhase.DRAINING)
        assert phases[-1] is LivePhase.IDLE
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert h.log.count("on_done") == 1
        assert h.wf.is_active is False

    def test_fast_note_saved_before_refine_and_overwritten(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        h.wf.stop()
        assert h.log.index("save") < h.log.index("refine.release_live")
        assert h.log.index("overwrite") < h.log.index("on_done")
        path, body, _participants, refined = h.overwrites[0]
        assert path == _fast_note(h)
        assert refined is True
        assert "## 文字起こし（高精度: large-v3）" in body
        assert "高精度の発話" in body
        assert "## ライブ文字起こし（速報）" in body
        assert "こんにちは" in body
        assert "高精度の要約" in body
        assert h.summaries == [("高精度の発話",)]

    def test_waits_for_asr_idle_before_refine(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        h.wf.stop()
        assert h.log.index("session.wait_asr_idle") < h.log.index("refine.release_live")

    def test_asr_not_idle_skips_refine(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        h.sessions[0].asr_idle = False
        h.wf.stop()
        assert h.refine.calls == []
        assert h.overwrites == []
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert any("スキップ" in m for m in h.spy.logs)

    def test_live_engine_released_before_refine_engine_loads(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        h.wf.stop()
        api_log = h.engine_api.log
        assert api_log == [
            "unload:mlx:large-v3-turbo",
            "load:mlx:large-v3",
            "unload:mlx:large-v3",
        ]

    def test_refine_uses_settings_language_and_context_hint(self, tmp_path):
        from live.context import LiveContext

        h = _harness(tmp_path, config=_config(tmp_path, live_language="ja"))
        h.wf.update_context(LiveContext.from_inputs("山田、佐藤", ""))
        h.start()
        h.wf.stop()
        call = h.refine.calls[0]
        assert call["wav"] == h.wav
        assert call["language"] == "ja"
        assert "山田" in call["hint"]
        assert h.overwrites[0][2] == ("山田", "佐藤")

    def test_progress_banner_shown_and_cleared(self, tmp_path):
        seen: list[str | None] = []
        h = _harness(tmp_path)

        def gate(_cancel):
            banner = h.banner(BANNER_REFINE)
            seen.append(None if banner is None else banner.text)

        h.refine.gate = gate
        h.start()
        h.wf.stop()
        assert seen and seen[0] is not None and "0%" in seen[0]
        assert h.banner(BANNER_REFINE) is None

    def test_summary_failure_keeps_live_summary(self, tmp_path):
        h = _harness(tmp_path, full_summary=lambda *a, **k: None)
        h.start()
        h.wf.stop()
        body = h.overwrites[0][1]
        assert "要約" in body and "高精度の要約" not in body

    def test_overwrite_failure_reports_and_keeps_fast_note(self, tmp_path):
        h = _harness(tmp_path)

        def broken(*_a, **_k):
            raise RuntimeError("保存エラー: disk")

        h.wf._deps = type(h.wf._deps)(**{**h.wf._deps.__dict__, "note_overwriter": broken})
        h.start()
        h.wf.stop()
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert any("上書き" in m for m in h.spy.logs)


class TestRefineSkips:
    def test_disabled_does_not_refine(self, tmp_path):
        h = _harness(tmp_path, config=_config(tmp_path, live_refine_enabled=False))
        h.start()
        h.spy.states.clear()
        h.wf.stop()
        assert LivePhase.REFINING not in _phases(h.spy.states)
        assert h.refine.calls == []
        assert h.spy.done == [(h.wav, _fast_note(h))]

    def test_faster_whisper_skips_with_hint(self, tmp_path):
        h = _harness(tmp_path, config=_config(tmp_path, live_engine="faster-whisper"))
        h.start()
        h.wf.stop()
        assert h.refine.calls == []
        assert any("スキップ" in m and "高精度で文字起こし" in m for m in h.spy.logs)
        assert h.log.count("on_done") == 1

    def test_load_time_fallback_skips(self, tmp_path):
        h = _harness(tmp_path, load_fails={"mlx"})
        h.start()
        h.sessions[0].engine.load()  # asr スレッドでの load を模す(faster-whisper に切替)
        h.wf.stop()
        assert h.refine.calls == []
        assert any("faster-whisper" in m for m in h.spy.logs)

    def test_engine_lock_busy_skips(self, tmp_path):
        h = _harness(tmp_path, engine_lock_timeout=0.01)
        h.start()
        refine_thread = _stop_until_refining(h)
        h.wf._engine_lock.acquire()  # 放棄した asr スレッドが使用中
        try:
            refine_thread.run()
        finally:
            h.wf._engine_lock.release()
        assert h.refine.calls == []
        assert h.overwrites == []
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert h.state.phase is LivePhase.IDLE

    def test_refine_error_keeps_fast_note(self, tmp_path):
        h = _harness(tmp_path, refine=None)
        h.refine.error = RuntimeError("本文を含む秘密")
        h.start()
        h.wf.stop()
        assert h.overwrites == []
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert not any("本文を含む秘密" in m for m in h.spy.logs)
        assert h.engine_api.log[-1] == "unload:mlx:large-v3"

    def test_incomplete_refine_keeps_fast_note(self, tmp_path):
        from live.refine import RefineResult

        h = _harness(tmp_path)
        h.refine.result = RefineResult((), False, 1.0)
        h.start()
        h.wf.stop()
        assert h.overwrites == []
        assert h.spy.done == [(h.wav, _fast_note(h))]


class TestRefineModelFetch:
    def test_refine_model_fetched_in_background_after_recording_starts(self, tmp_path):
        fetcher = FakeFetcher()
        fetcher.uncached = frozenset({"repo/large-v3"})
        h = _harness(tmp_path, fetcher=fetcher)
        h.wf._thread_factory = DeferredThread
        h.start()
        run_named("live-prepare")
        assert h.state.phase is LivePhase.RECORDING  # 取得を待たずに録音を始める
        assert fetcher.fetched == []
        fetch_thread = next(t for t in DeferredThread.instances if t.name == "refine-model-fetch")
        assert fetch_thread.daemon is True
        fetch_thread.run()
        assert [r.repo_id for r in fetcher.fetched] == ["repo/large-v3"]

    def test_still_fetching_at_stop_skips_refine(self, tmp_path):
        fetcher = FakeFetcher()
        fetcher.uncached = frozenset({"repo/large-v3"})
        h = _harness(tmp_path, fetcher=fetcher)
        h.wf._thread_factory = DeferredThread
        h.start()
        run_named("live-prepare")
        h.wf.stop()
        run_named("live-stop")
        run_named("live-drain")
        assert h.refine.calls == []
        assert h.state.phase is LivePhase.IDLE
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert any("取得中" in m and "高精度で文字起こし" in m for m in h.spy.logs)

    def test_no_second_fetch_while_previous_is_running(self, tmp_path):
        fetcher = FakeFetcher()
        fetcher.uncached = frozenset({"repo/large-v3"})
        h = _harness(tmp_path, fetcher=fetcher)
        h.wf._thread_factory = DeferredThread
        h.start()
        run_named("live-prepare")
        h.wf.stop()
        run_named("live-stop")
        run_named("live-drain")
        h.start()  # 2回目の実行(前回の取得はまだ動いている)
        run_named("live-prepare")
        fetches = [t for t in DeferredThread.instances if t.name == "refine-model-fetch"]
        assert len(fetches) == 1

    def test_fetch_failure_only_logs_and_recording_continues(self, tmp_path):
        class FailingFetcher(FakeFetcher):
            def fetch(self, repo, on_progress):
                if repo.repo_id == "repo/large-v3":
                    raise OSError("secret /Users/x")
                return super().fetch(repo, on_progress)

        fetcher = FailingFetcher()
        fetcher.uncached = frozenset({"repo/large-v3"})
        h = _harness(tmp_path, fetcher=fetcher)
        h.start()
        assert h.state.phase is LivePhase.RECORDING
        assert any("取得できませんでした" in m and "OSError" in m for m in h.spy.logs)
        assert not any("secret" in m for m in h.spy.logs)
        h.wf.stop()
        assert h.refine.calls == []

    def test_faster_whisper_does_not_fetch_refine_model(self, tmp_path):
        fetcher = FakeFetcher()
        fetcher.uncached = frozenset({"repo/large-v3"})
        h = _harness(
            tmp_path, fetcher=fetcher, config=_config(tmp_path, live_engine="faster-whisper")
        )
        h.start()
        assert fetcher.fetched == []


class TestRefineAbort:
    def test_abort_returns_to_idle_without_waiting(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        refine_thread = _stop_until_refining(h)
        assert h.wf.abort_drain() is True
        assert h.state.phase is LivePhase.IDLE
        assert h.spy.done == [(h.wav, _fast_note(h))]
        assert h.banner(BANNER_REFINE) is None
        # 放棄したスレッドが生きている間は開始を拒否する
        assert h.wf.is_active is True
        assert "後処理" in h.start()
        refine_thread.run()  # 取り消されたので結果は捨てられる
        assert h.overwrites == []
        assert h.log.count("on_done") == 1
        assert h.wf.is_active is False

    def test_abort_sets_cancel_seen_by_refine(self, tmp_path):
        cancelled: list[bool] = []
        h = _harness(tmp_path)
        h.refine.gate = lambda cancel: cancelled.append(cancel.is_set())
        h.start()
        refine_thread = _stop_until_refining(h)
        h.wf.abort_drain()
        refine_thread.run()
        assert cancelled == [True]

    def test_late_result_from_previous_generation_is_discarded(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        refine_thread = _stop_until_refining(h)
        h.wf.abort_drain()
        reset_deferred_threads()
        h.wf._thread_factory = DeferredThread
        refine_thread.run()  # 新しい実行を始める前に終える
        assert h.start() is None
        for t in list(DeferredThread.instances):
            t.run()
        assert h.state.phase is LivePhase.RECORDING
        assert h.overwrites == []

    def test_abort_after_commit_is_refused(self, tmp_path):
        results: list[bool] = []
        h = _harness(tmp_path)
        original = h._overwrite_note

        def overwrite(*args, **kwargs):
            results.append(h.wf.abort_drain())
            return original(*args, **kwargs)

        h.wf._deps = type(h.wf._deps)(**{**h.wf._deps.__dict__, "note_overwriter": overwrite})
        h.start()
        h.wf.stop()
        assert results == [False]
        assert h.log.count("on_done") == 1

    def test_shutdown_during_refine_keeps_fast_note(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        refine_thread = _stop_until_refining(h)
        saves_before = len(h.saved_notes)
        h.wf.shutdown()
        assert h.state.phase is LivePhase.IDLE
        assert len(h.saved_notes) == saves_before  # 速報版は保存済みなので追加の保存はしない
        refine_thread.run()
        assert h.overwrites == []
        assert h.spy.done == []

    def test_abandoned_refine_does_not_log(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        refine_thread = _stop_until_refining(h)
        h.wf.abort_drain()
        h.spy.logs.clear()
        refine_thread.run()  # 高精度モデルの読み込み通知(notify)は捨てる
        assert not any("large-v3" in m for m in h.spy.logs)

    def test_refine_notify_logged_while_current(self, tmp_path):
        h = _harness(tmp_path)
        h.start()
        h.wf.stop()
        assert any("large-v3 を読み込みました" in m for m in h.spy.logs)

    def test_abort_not_accepted_when_idle(self, tmp_path):
        h = _harness(tmp_path)
        assert h.wf.abort_drain() is False


def test_refine_banner_constant_exported():
    assert BANNER_REFINE == "refine"


def test_engine_lock_is_released_after_refine(tmp_path):
    h = _harness(tmp_path)
    h.start()
    h.wf.stop()
    assert h.wf._engine_lock.acquire(timeout=0.01)
    h.wf._engine_lock.release()


def test_refine_runs_on_dedicated_thread(tmp_path):
    names: list[str] = []
    h = _harness(tmp_path, refine=FakeRefine([]))
    original = h.wf._thread_factory

    def factory(*args, **kwargs):
        names.append(kwargs.get("name"))
        return original(*args, **kwargs)

    h.wf._thread_factory = factory
    h.start()
    h.wf.stop()
    assert "live-refine" in names
    assert threading.active_count() >= 1


def test_unexpected_refine_result_still_finishes_once(tmp_path):
    """想定外の戻り値・例外でも REFINING に取り残さず、速報版で on_done を1回だけ呼ぶ"""
    h = _harness(tmp_path)
    h.refine.result = object()  # complete 属性を持たない
    h.start()
    h.wf.stop()
    assert h.state.phase is LivePhase.IDLE
    assert h.spy.done == [(h.wav, _fast_note(h))]
    assert h.overwrites == []
    assert h.wf._engine_lock.acquire(timeout=0.01)
    h.wf._engine_lock.release()


# ──────────────── レビュー指摘(2回目): 取得確認の例外でマイクを開いたまま戻らない ────────────────


class _RaisingFetcher(FakeFetcher):
    def __init__(self, error):
        super().__init__()
        self.error_to_raise = error

    def resolve_repo(self, engine, model):
        if model == "large-v3":
            raise self.error_to_raise
        return super().resolve_repo(engine, model)


@pytest.mark.parametrize(
    "error",
    [RuntimeError("Could not determine home directory."), PermissionError("secret /Users/x")],
    ids=["runtime", "permission"],
)
def test_fetch_check_error_keeps_recording_and_stop_saves_fast_note(tmp_path, error):
    h = _harness(tmp_path, fetcher=_RaisingFetcher(error))
    h.start()
    assert h.state.phase is LivePhase.RECORDING
    assert h.recorders[0].started is True
    assert h.spy.errors == []
    h.log.clear()
    h.wf.stop()
    assert h.log[:4] == ["finalize", "on_wav_saved", "stop_with_timeout", "session.stop"]
    assert "save" in h.log and h.log[-1] == "on_done"
    assert h.state.phase is LivePhase.IDLE
    assert h.spy.done == [(h.wav, _fast_note(h))]
    assert h.refine.calls == []
    assert not any("secret" in m for m in h.spy.logs)
    assert any(f"確認できません（{type(error).__name__}）" in m for m in h.spy.logs)


def test_start_refine_fetch_unexpected_error_is_contained(tmp_path):
    h = _harness(tmp_path)

    def broken(*_a, **_k):
        raise RuntimeError("unexpected")

    h.wf._refine_fetch.start = broken
    h.start()
    assert h.state.phase is LivePhase.RECORDING
    h.wf.stop()
    assert h.spy.done == [(h.wav, _fast_note(h))]


def test_plan_refine_error_still_saves_fast_note(tmp_path):
    h = _harness(tmp_path)
    h.start()

    def broken():
        raise RuntimeError("plan failed")

    h.wf._plan_refine = broken
    h.wf.stop()
    assert h.state.phase is LivePhase.IDLE
    assert h.spy.done == [(h.wav, _fast_note(h))]
    assert h.refine.calls == []


# ──────────────── レビュー指摘(3回目): 要約待ちの打ち切り ────────────────


def _wait_until(predicate, timeout=3.0) -> bool:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_abort_drain_cancels_summary_and_allows_restart(tmp_path):
    h = _harness(tmp_path, config=_config(tmp_path, live_refine_enabled=False))
    h.start()
    session = h.sessions[0]
    session.block_stop = True  # 要約待ちで止まるドレイン
    h.wf._thread_factory = threading.Thread
    h.wf.stop()
    assert _wait_until(lambda: h.state.phase is LivePhase.DRAINING and session.stop_timeouts)
    assert h.wf.abort_drain() is True
    assert "session.cancel_summary" in h.log
    # 放棄したドレインスレッドは要約の打ち切りですぐ終わり、再開始できる
    assert _wait_until(lambda: h.state.phase is LivePhase.IDLE and not h.wf.is_active)
    h.wf._thread_factory = ImmediateThread
    assert h.start() is None


def test_shutdown_during_drain_cancels_summary(tmp_path):
    h = _harness(tmp_path, config=_config(tmp_path, live_refine_enabled=False))
    h.start()
    h.wf._thread_factory = DeferredThread
    h.wf.stop()
    run_named("live-stop")
    assert h.state.phase is LivePhase.DRAINING
    h.wf.shutdown()
    assert "session.cancel_summary" in h.log


def test_shutdown_while_recording_cancels_summary_and_skips_llm(tmp_path):
    h = _harness(tmp_path)
    h.start()
    h.wf.shutdown()
    session = h.sessions[0]
    assert "session.cancel_summary" in h.log
    assert session.summary_timeouts == [0.0]


@pytest.mark.parametrize("case", ["no_audio", "prepare_fail", "emergency"])
def test_error_paths_skip_final_summary(tmp_path, case):
    if case == "no_audio":
        h = _harness(tmp_path, finalized=None)
        h.start()
        h.wf.stop()
    elif case == "prepare_fail":
        h = _harness(tmp_path, fail_start=True)
        h.start()
    else:
        h = _harness(tmp_path)
        h.start()

        def no_threads(*_a, **_k):
            raise RuntimeError("can't start new thread")

        h.wf._thread_factory = no_threads
        h.wf.stop()
    assert h.sessions[0].summary_timeouts == [0.0]


def test_refine_thread_start_failure_finishes_with_fast_note(tmp_path):
    h = _harness(tmp_path)
    h.start()

    class _NoStart(ImmediateThread):
        def start(self):
            raise RuntimeError("can't start new thread")

    def factory(*args, **kwargs):
        if kwargs.get("name") == "live-refine":
            return _NoStart(*args, **kwargs)
        return ImmediateThread(*args, **kwargs)

    h.wf._thread_factory = factory
    h.wf.stop()
    assert h.state.phase is LivePhase.IDLE
    assert h.spy.done == [(h.wav, _fast_note(h))]
    assert h.wf.is_active is False
