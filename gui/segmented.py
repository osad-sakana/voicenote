"""CTkSegmentedButton の文字色を選択状態に応じて切り替えるヘルパー。

CTkSegmentedButton の `text_color` は全セグメント共通のため、
未選択セグメント (薄いグレー背景) で白文字になり読みにくい。
"""

import customtkinter as ctk

from . import theme

_SELECTED_TEXT = ("white", "white")


def refresh_segment_text(selector: ctk.CTkSegmentedButton) -> None:
    """選択中は白、未選択は通常の文字色にする。"""
    current = selector.get()
    for value, button in selector._buttons_dict.items():
        button.configure(text_color=_SELECTED_TEXT if value == current else theme.TEXT)
