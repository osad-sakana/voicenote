"""ライブ画面への表示反映(ウィジェット操作だけの薄い層。何を出すかは live_view_logic が決める)"""

from __future__ import annotations

from . import theme
from .live_context_form import ContextFormWidgets
from .live_layout import LivePanelWidgets
from .live_state import LiveViewState
from .live_view_logic import (
    BannerView,
    MeterView,
    banner_view,
    plan_segment_render,
    transcript_toggle_text,
)


class LivePanel:
    """ライブ画面のウィジェットを保持し、状態を差分で描画する(メインスレッド専用)"""

    def __init__(self, widgets: LivePanelWidgets):
        self._w = widgets
        self._rendered_count = 0
        self._summary = ""
        self._banner: BannerView | None = None
        self._progress_indeterminate = False
        self._transcript_collapsed = False

    @property
    def context_form(self) -> ContextFormWidgets:
        """参加者・メモ入力フォーム(clear() では消さない。実行をまたいで保持する)"""
        return self._w.context_form

    # ──────────────── 表示/非表示 ────────────────

    def show(self, after) -> None:
        """after(下端に置いた詳細ログ欄)の後に配置し、ウィンドウの残りの縦幅を使う"""
        self._w.frame.pack(fill="both", expand=True, padx=theme.SCREEN_PAD, after=after)

    def hide(self) -> None:
        self._w.frame.pack_forget()

    @property
    def transcript_collapsed(self) -> bool:
        return self._transcript_collapsed

    def toggle_transcript(self) -> None:
        """文字起こし欄を折りたたむ/戻す(折りたたむと要約欄がその分広がる)"""
        self.set_transcript_collapsed(not self._transcript_collapsed)

    def set_transcript_collapsed(self, collapsed: bool) -> None:
        w = self._w
        self._transcript_collapsed = collapsed
        w.transcript_toggle.configure(text=transcript_toggle_text(collapsed))
        if collapsed:
            w.segments_box.pack_forget()
        elif not w.segments_box.winfo_manager():
            w.segments_box.pack(fill="x", pady=(4, 0))
            w.segments_box.see("end")

    # ──────────────── 状態 ────────────────

    def clear(self) -> None:
        """次の実行に備えて区間・要約・バナー・警告・結果を消す(参加者・メモは残す)"""
        self._replace_text(self._w.segments_box, "")
        self._replace_text(self._w.summary_box, "")
        self._rendered_count = 0
        self._summary = ""
        self.render_banners(banner_view(()))
        self.show_warning(None)
        self.hide_result()

    def render_state(self, state: LiveViewState) -> None:
        self._render_segments(state)
        if state.summary != self._summary:
            self._summary = state.summary
            self._replace_text(self._w.summary_box, state.summary)
        self.render_banners(banner_view(state.banners))

    def _render_segments(self, state: LiveViewState) -> None:
        plan = plan_segment_render(state, self._rendered_count)
        delta = plan.delta
        box = self._w.segments_box
        if not plan.reset and not delta.lines:
            return
        box.configure(state="normal")
        if plan.reset:
            box.delete("1.0", "end")
        if delta.trim_count:
            box.delete("1.0", f"{delta.trim_count + 1}.0")
        if delta.lines:
            box.insert("end", "".join(f"{line}\n" for line in delta.lines))
            box.see("end")
        box.configure(state="disabled")
        self._rendered_count = delta.new_count

    @staticmethod
    def _replace_text(box, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        if text:
            box.insert("end", text)
        box.configure(state="disabled")

    def render_banners(self, view: BannerView) -> None:
        if view == self._banner:
            return
        self._banner = view
        card, progress = self._w.banner_card, self._w.banner_progress
        if not view.visible:
            self._stop_indeterminate()
            card.pack_forget()
            return
        self._w.banner_label.configure(text=view.text)
        if view.show_progress:
            progress.pack(fill="x", padx=theme.CARD_PAD, pady=(0, 10))
            self._set_progress(view.fraction)
        else:
            self._stop_indeterminate()
            progress.pack_forget()
        if not card.winfo_manager():
            card.pack(fill="x", pady=(8, 0), before=self._w.context_form.frame)

    def _set_progress(self, fraction: float | None) -> None:
        progress = self._w.banner_progress
        if fraction is None:
            if not self._progress_indeterminate:
                progress.configure(mode="indeterminate")
                progress.start()
                self._progress_indeterminate = True
            return
        self._stop_indeterminate()
        progress.set(fraction)

    def _stop_indeterminate(self) -> None:
        if self._progress_indeterminate:
            self._w.banner_progress.stop()
            self._w.banner_progress.configure(mode="determinate")
            self._progress_indeterminate = False

    def render_meter(self, view: MeterView) -> None:
        w = self._w
        w.meter_bar.set(view.fraction)
        w.meter_bar.configure(progress_color=theme.RECORDING if view.low_input else theme.SUCCESS)
        w.level_label.configure(text=view.level_text)
        w.backlog_label.configure(
            text=view.backlog_text,
            text_color=theme.RECORDING if view.backlog_high else theme.TEXT_SECONDARY,
        )
        w.dropped_label.configure(text=view.dropped_text)

    def show_warning(self, text: str | None) -> None:
        label = self._w.warning_label
        if not text:
            label.configure(text="")
            label.pack_forget()
            return
        label.configure(text=f"警告: {text}")
        if not label.winfo_manager():
            # 下端詰めなので、メーター行より前に積むとメーターの下に来る
            label.pack(side="bottom", fill="x", pady=(4, 0), before=self._w.meter_row)

    def show_result(self, text: str, *, error: bool, can_retranscribe: bool) -> None:
        w = self._w
        w.result_label.configure(text=text, text_color=theme.RECORDING if error else theme.TEXT)
        if can_retranscribe:
            w.retranscribe_btn.pack(side="right", padx=(0, theme.CARD_PAD), pady=8)
        else:
            w.retranscribe_btn.pack_forget()
        if not w.result_card.winfo_manager():
            below = w.warning_label if w.warning_label.winfo_manager() else w.meter_row
            w.result_card.pack(side="bottom", fill="x", pady=(6, 0), before=below)

    def hide_result(self) -> None:
        self._w.result_card.pack_forget()

    def set_retranscribe_enabled(self, enabled: bool) -> None:
        self._w.retranscribe_btn.configure(state="normal" if enabled else "disabled")
