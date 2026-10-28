"""gui/live_controller.py の実行ボタンのテスト: 停止・中断の直前に参加者・メモを反映する

macOS ではボタンがフォーカスを取らず <FocusOut> が発火しないため、デバウンス待ちの
編集はフォーカス頼みにせず、停止・中断の前に必ず送る。Tk は使わない(未束縛メソッドを
SimpleNamespace に対して呼ぶ)。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gui.live_context_logic import ContextSync, build_context
from gui.live_state import LivePhase, LiveViewState
from tests.test_gui_live_context_logic import FakeScheduler


def _controller_cls():
    try:
        from gui.live_controller import LiveController
    except Exception as e:  # pragma: no cover - GUI 依存が無い環境
        pytest.skip(f"GUI 依存を読み込めません: {e}")
    return LiveController


class _Recorder:
    """update_context / stop / abort_drain の呼び出し順を1本の記録に残すワークフロー"""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def update_context(self, context):
        self.calls.append(("update_context", context))

    def stop(self):
        self.calls.append(("stop", None))

    def abort_drain(self):
        self.calls.append(("abort_drain", None))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _stub(phase: LivePhase, inputs=("田中", "議題")):
    workflow = _Recorder()
    sched = FakeScheduler()
    sync = ContextSync(
        read_inputs=lambda: inputs,
        push=workflow.update_context,
        on_notice=lambda _text: None,
        schedule=sched.schedule,
        cancel=sched.cancel,
    )
    stub = SimpleNamespace(
        _state=LiveViewState(phase=phase), _workflow=workflow, _context_sync=sync
    )
    return stub, workflow, sched


_STOP_CASES = [
    (LivePhase.RECORDING, "stop"),
    (LivePhase.PREPARING, "stop"),
    (LivePhase.DRAINING, "abort_drain"),
]


class TestExecFlushesPendingContext:
    @pytest.mark.parametrize("phase, action", _STOP_CASES)
    def test_pending_edit_is_pushed_before_action(self, phase, action):
        cls = _controller_cls()
        stub, workflow, sched = _stub(phase)
        stub._context_sync.on_input_changed()  # デバウンス待ち(フォーカスは外れていない)
        cls.on_exec(stub)
        assert workflow.names() == ["update_context", action]
        assert workflow.calls[0][1] == build_context("田中", "議題")
        assert sched.pending == {}  # 後から二重送信しない

    @pytest.mark.parametrize("phase, action", _STOP_CASES)
    def test_no_pending_edit_sends_nothing_extra(self, phase, action):
        cls = _controller_cls()
        stub, workflow, _sched = _stub(phase)
        cls.on_exec(stub)
        assert workflow.names() == [action]

    @pytest.mark.parametrize("phase", [LivePhase.FINALIZING, LivePhase.SAVING])
    def test_disabled_phases_do_nothing(self, phase):
        cls = _controller_cls()
        stub, workflow, _sched = _stub(phase)
        stub._context_sync.on_input_changed()
        cls.on_exec(stub)
        assert workflow.calls == []
