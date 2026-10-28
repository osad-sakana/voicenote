"""ライブ画面の参加者・メモ入力フォームのウィジェット(配置と入力変更の通知だけ)

何をいつワークフローへ渡すかは gui/live_context_logic.py の ContextSync が決める。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import customtkinter as ctk

from . import theme
from .live_view_logic import textbox_height

PARTICIPANTS_PLACEHOLDER = "例: 田中、佐藤、鈴木（読点・カンマ区切り）"
MEMO_CAPTION = "メモ（議題・用語・固有名詞など。要約の参考にします）"
_MEMO_LINES = 3  # メモ欄はおよそ3行分(スクロールで続きを書ける)
_WRAP_PX = 560


@dataclass(frozen=True)
class ContextFormWidgets:
    """参加者・メモ入力フォームのウィジェット一式"""

    frame: ctk.CTkFrame
    participants_entry: ctk.CTkEntry
    memo_box: ctk.CTkTextbox
    notice_label: ctk.CTkLabel
    clear_btn: ctk.CTkButton

    def read_inputs(self) -> tuple[str, str]:
        """(参加者の文字列, メモの文字列)。プレースホルダ表示中の参加者は空"""
        return self.participants_entry.get(), self.memo_box.get("1.0", "end-1c")

    def clear_inputs(self) -> None:
        self.participants_entry.delete(0, "end")
        self.memo_box.delete("1.0", "end")

    def show_notice(self, text: str | None) -> None:
        label = self.notice_label
        if not text:
            label.configure(text="")
            label.grid_remove()
            return
        label.configure(text=text)
        label.grid()


def _caption(parent, text: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent, text=text, anchor="w", font=theme.font(12), text_color=theme.TEXT_SECONDARY
    )


def build_context_form(
    parent, on_change: Callable[[], None], on_clear: Callable[[], None]
) -> ContextFormWidgets:
    """フォームを parent の中に構築して返す(frame 自体はまだ配置しない)

    on_change: 入力が変わるたびに呼ぶ(デバウンスは呼び出し側)。常に編集可能のまま使う
    """
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    # 左に参加者(1行)とクリア、右にメモ(複数行)。横に並べて縦幅を要約に回す
    frame.grid_columnconfigure(0, weight=2, uniform="context")
    frame.grid_columnconfigure(1, weight=3, uniform="context")
    _caption(frame, "参加者").grid(row=0, column=0, sticky="w")
    entry = _build_participants_entry(frame, on_change)
    entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))
    clear_btn = ctk.CTkButton(
        frame,
        text="クリア",
        width=64,
        height=26,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(12),
        command=on_clear,
        **theme.SECONDARY_STYLE,
    )
    clear_btn.grid(row=2, column=0, sticky="sw", pady=(6, 0))
    _caption(frame, MEMO_CAPTION).grid(row=0, column=1, sticky="w", padx=(10, 0))
    memo = _build_memo_box(frame, on_change)
    memo.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(10, 0), pady=(2, 0))
    notice = ctk.CTkLabel(
        frame,
        text="",
        height=18,
        anchor="w",
        justify="left",
        wraplength=_WRAP_PX,
        font=theme.font(12),
        text_color=theme.RECORDING,
    )
    notice.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))
    notice.grid_remove()
    return ContextFormWidgets(frame, entry, memo, notice, clear_btn)


def _build_participants_entry(frame, on_change: Callable[[], None]) -> ctk.CTkEntry:
    entry = ctk.CTkEntry(
        frame,
        height=28,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        placeholder_text=PARTICIPANTS_PLACEHOLDER,
    )

    def on_validate() -> bool:
        # 挿入・削除のたびに呼ばれる(IME 確定・貼り付けを含む)。必ず True を返す
        on_change()
        return True

    # textvariable を使うとプレースホルダが出なくなるので、key 検証で変更を拾う
    command = entry.register(on_validate)
    entry.configure(validate="key", validatecommand=(command,))
    entry.bind("<FocusOut>", lambda _event: on_change())
    return entry


def _build_memo_box(frame, on_change: Callable[[], None]) -> ctk.CTkTextbox:
    box = ctk.CTkTextbox(
        frame,
        height=textbox_height(_MEMO_LINES, 13),
        wrap="char",
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        fg_color=theme.CARD_BG,
        text_color=theme.TEXT,
        border_width=1,
        border_color=theme.BORDER,
    )

    def on_modified(_event=None) -> None:
        if not box.edit_modified():
            return  # フラグを戻した時の <<Modified>> は無視する
        box.edit_modified(False)
        on_change()

    box.bind("<<Modified>>", on_modified)
    return box
