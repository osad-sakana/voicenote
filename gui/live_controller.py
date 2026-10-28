"""App と LiveWorkflow の配線(ライブモードの操作・コールバック・メーター更新)

LiveWorkflow のコールバックはワーカースレッドから(on_state はワークフローのロックを
持ったまま)呼ばれるため、全て ThreadSafeUIQueue.submit で包んでメインスレッドに渡す。
描画はコールバックで受け取った状態から行い、描画経路でワークフローのロックを取らない。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox
from typing import Any

from config import VoiceNoteConfig

from . import theme
from .constants import MODE_LIVE, MODE_TRANSCRIBE_ONLY
from .devices import parse_device_id
from .live_context_logic import ContextSync
from .live_layout import LivePanelCommands, build_live_panel
from .live_panel import LivePanel
from .live_state import LivePhase, LiveViewState
from .live_view_logic import (
    ButtonAction,
    ButtonStyle,
    button_spec,
    controls_locked,
    meter_view,
    new_warning,
    result_text,
    status_text,
)
from .live_workflow import LiveWorkflow, LiveWorkflowCallbacks, validate_live_start
from .segmented import refresh_segment_text
from .workflow import resolve_recording_dest

METER_INTERVAL_MS = 250
COMPLETION_RESET_MS = 3000

# (config, callbacks, can_start) -> LiveWorkflow 互換。スモーク確認などで差し替える
LiveWorkflowFactory = Callable[[VoiceNoteConfig, LiveWorkflowCallbacks, Callable[[], bool]], Any]

_STYLES = {
    ButtonStyle.PRIMARY: theme.PRIMARY_STYLE,
    ButtonStyle.DANGER: theme.DANGER_STYLE,
    ButtonStyle.SECONDARY: theme.SECONDARY_STYLE,
}


def default_live_factory(config, callbacks, can_start):
    return LiveWorkflow(config, callbacks, can_start=can_start)


class LiveController:
    """ライブモードの UI 制御。App のウィジェット(layout.py が取り付けた属性)を操作する"""

    def __init__(self, app, factory: LiveWorkflowFactory | None = None):
        self._app = app
        submit = app._ui_queue.submit
        callbacks = LiveWorkflowCallbacks(
            on_state=lambda state: submit(self._on_state, state),
            on_log=lambda msg: submit(app._log, msg),
            on_recording_started=lambda: submit(self._on_recording_started),
            on_wav_saved=lambda wav: submit(self._on_wav_saved, wav),
            on_done=lambda wav, note: submit(self._on_done, wav, note),
            on_error=lambda msg: submit(self._on_error, msg),
        )
        self._workflow = (factory or default_live_factory)(app._config, callbacks, self._can_start)
        commands = LivePanelCommands(
            on_retranscribe=self._on_retranscribe,
            on_context_change=self._on_context_change,
            on_context_clear=self._on_context_clear,
            on_toggle_transcript=self._on_toggle_transcript,
        )
        self._panel = LivePanel(build_live_panel(app, commands))
        form = self._panel.context_form
        # 参加者・メモはどの段階でも編集でき、デバウンスしてワークフローへ渡す(本文はログに出さない)
        self._context_sync = ContextSync(
            read_inputs=form.read_inputs,
            push=self._workflow.update_context,
            on_notice=form.show_notice,
            schedule=app.after,
            cancel=app.after_cancel,
        )
        self._state = LiveViewState()
        self._last_wav: Path | None = None
        self._last_warning: str | None = None
        self._meter_after_id: str | None = None
        self._reset_after_id: str | None = None

    # ──────────────── App から呼ぶ ────────────────

    @property
    def is_active(self) -> bool:
        return self._workflow.is_active

    def has_pending_close(self) -> bool:
        """ライブで放棄したストリームが猶予内でまだ解放中か(通常録音の開始判定で使う)"""
        return self._workflow.has_pending_close()

    def update_config(self, config: VoiceNoteConfig) -> None:
        self._workflow.update_config(config)

    def _can_start(self) -> bool:
        """マイク排他: App が処理中でなく、通常録音もしておらず、
        通常録音で放棄したストリームも解放済みなら開始できる"""
        workflow = self._app._workflow
        return (
            not self._app._busy and not workflow.is_recording and not workflow.has_pending_close()
        )

    def show(self) -> None:
        self._panel.show(after=self._app._log_container)
        self.refresh_controls()

    def hide(self) -> None:
        self._panel.hide()

    def refresh_controls(self) -> None:
        """実行ボタン・状態カード・操作ロックを現在のライブ段階に合わせる"""
        phase = self._state.phase
        self._apply_lock(controls_locked(phase))
        if self._app._mode_var.get() != MODE_LIVE:
            return  # 他モード表示中(ライブ処理中はモードを固定するので IDLE のみ)
        spec = button_spec(phase)
        self._app._exec_btn.configure(
            text=spec.text,
            state="normal" if spec.enabled else "disabled",
            **_STYLES[spec.style],
        )
        if phase is not LivePhase.IDLE:
            self._app._status_label.configure(
                text=status_text(phase, self._state.status), text_color=theme.TEXT
            )

    def _apply_lock(self, locked: bool) -> None:
        state = "disabled" if locked or self._app._busy else "normal"
        for widget in (self._app._mode_selector, self._app._device_menu, self._app._settings_btn):
            widget.configure(state=state)
        self._panel.set_retranscribe_enabled(not locked)

    def on_exec(self) -> None:
        action = button_spec(self._state.phase).action
        if action is ButtonAction.START:
            self._start()
        elif action is ButtonAction.STOP:
            # 停止・中断の直前の編集もノートと最終要約に入れる(フォーカス移動に頼らない)
            self._context_sync.flush_pending(force=True)
            self._workflow.stop()
        elif action is ButtonAction.ABORT:
            self._context_sync.flush_pending(force=True)
            self._workflow.abort_drain()

    def _start(self) -> None:
        app = self._app
        device_label = app._device_var.get()
        busy = app._busy or app._workflow.is_recording
        error = validate_live_start(
            app._config,
            device_label,
            mic_busy=busy,
            stream_closing=app._workflow.has_pending_close(),
        )
        # デバウンス待ちの編集を取りこぼさないよう、開始前に最新の入力を必ず渡す
        self._context_sync.flush(force=True)
        if error:
            messagebox.showwarning(*error)
            return
        self._cancel_reset()
        self._panel.clear()
        self._last_wav = None
        self._last_warning = None
        message = self._workflow.start(
            parse_device_id(device_label), device_label, resolve_recording_dest(app._config)
        )
        if message:
            messagebox.showerror("ライブ開始エラー", message)

    def shutdown(self) -> None:
        """ウィンドウを閉じる時に呼ぶ(部分ノートを同期保存する)"""
        self._cancel_meter()
        self._cancel_reset()
        # 終了時に保存する部分ノートへ、デバウンス待ちの編集も入れる
        self._context_sync.flush()
        self._workflow.shutdown()

    # ──────────────── workflow コールバック (メインスレッドで実行) ────────────────

    def _on_state(self, state: LiveViewState) -> None:
        previous = self._state.phase
        self._state = state
        self._panel.render_state(state)
        warning = new_warning(state, self._last_warning)
        if warning is not None:
            self._last_warning = warning
            self._panel.show_warning(warning)
            self._app._log(f"警告: {warning}")
        if state.phase is not previous:
            self._on_phase_change(state.phase)
        self.refresh_controls()

    def _on_phase_change(self, phase: LivePhase) -> None:
        app = self._app
        if phase is LivePhase.RECORDING:
            app._status_dot.configure(text_color=theme.RECORDING)
            self._schedule_meter()
            return
        self._cancel_meter()
        self._panel.render_meter(meter_view(None))
        if phase is LivePhase.FINALIZING:
            app._stop_timer()
        if phase is LivePhase.IDLE:
            # 完了・エラーの通知は IDLE への遷移の後に届くので、ここでは待機中に戻すだけ
            app._stop_timer()
            app._status_dot.configure(text_color=theme.TEXT_SECONDARY)
            app._status_label.configure(text="待機中", text_color=theme.TEXT)
        else:
            app._status_dot.configure(text_color=theme.ACCENT)

    def _on_recording_started(self) -> None:
        self._app._start_timer()

    def _on_wav_saved(self, wav: Path) -> None:
        self._last_wav = wav

    def _on_done(self, wav: Path, note: Path) -> None:
        app = self._app
        self._last_wav = wav
        app._log(f"ライブ完了 → WAV: {wav} / ノート: {note}")
        app._status_label.configure(text=f"完了: {note.name}", text_color=theme.SUCCESS)
        app._status_dot.configure(text_color=theme.SUCCESS)
        self._panel.show_result(result_text(wav, note), error=False, can_retranscribe=True)
        app.bell()
        self._cancel_reset()
        self._reset_after_id = app.after(COMPLETION_RESET_MS, self._reset_status)

    def _on_error(self, msg: str) -> None:
        self._app._log(msg)
        self._app._stop_timer()
        wav = self._last_wav
        text = msg if wav is None else f"{msg}\n{result_text(wav, None)}"
        self._panel.show_result(text, error=True, can_retranscribe=wav is not None)

    def _reset_status(self) -> None:
        self._reset_after_id = None
        if self._state.phase is LivePhase.IDLE:
            self._app._status_label.configure(text="待機中", text_color=theme.TEXT)
            self._app._status_dot.configure(text_color=theme.TEXT_SECONDARY)

    def _on_retranscribe(self) -> None:
        """「高精度で文字起こし」: 文字起こしだけモードに切り替え、WAV を入力欄に入れる"""
        wav = self._last_wav
        if wav is None or self._state.phase is not LivePhase.IDLE:
            return
        app = self._app
        app._mode_selector.set(MODE_TRANSCRIBE_ONLY)
        refresh_segment_text(app._mode_selector)
        app._on_mode_change()
        app._file_entry.delete(0, "end")
        app._file_entry.insert(0, str(wav))

    # ──────────────── 参加者・メモ / 文字起こし欄 ────────────────

    def _on_context_change(self) -> None:
        if self._app._alive:
            self._context_sync.on_input_changed()

    def _on_context_clear(self) -> None:
        """「クリア」: 入力欄を空にし、参考情報も空にする(ライブ中でも可)"""
        self._context_sync.clear()
        self._panel.context_form.clear_inputs()

    def _on_toggle_transcript(self) -> None:
        self._panel.toggle_transcript()

    # ──────────────── 入力メーター ────────────────

    def _schedule_meter(self) -> None:
        self._cancel_meter()
        self._meter_after_id = self._app.after(METER_INTERVAL_MS, self._tick_meter)

    def _tick_meter(self) -> None:
        self._meter_after_id = None
        if not self._app._alive or self._state.phase is not LivePhase.RECORDING:
            return
        self._panel.render_meter(meter_view(self._workflow.poll_meter()))
        self._schedule_meter()

    def _cancel_meter(self) -> None:
        if self._meter_after_id is not None:
            self._app.after_cancel(self._meter_after_id)
            self._meter_after_id = None

    def _cancel_reset(self) -> None:
        if self._reset_after_id is not None:
            self._app.after_cancel(self._reset_after_id)
            self._reset_after_id = None
