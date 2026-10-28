"""ライブモード画面のウィジェット構築(配置だけを行い、表示更新は gui/live_panel.py)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import customtkinter as ctk

from . import theme
from .live_context_form import ContextFormWidgets, build_context_form
from .live_view_logic import TRANSCRIPT_LINES, textbox_height, transcript_toggle_text

_METER_WIDTH = 160
_WRAP_PX = 560  # バナー・結果の折り返し幅(ウィンドウ最小幅に合わせる)
# 要約は要求高さを小さくし、余った縦幅は expand で要約欄に回す(主役)
_SUMMARY_MIN_LINES = 3
_TOGGLE_WIDTH = 120


@dataclass(frozen=True)
class LivePanelWidgets:
    """ライブ画面のウィジェット一式"""

    frame: ctk.CTkFrame
    banner_card: ctk.CTkFrame
    banner_label: ctk.CTkLabel
    banner_progress: ctk.CTkProgressBar
    context_form: ContextFormWidgets
    body: ctk.CTkFrame
    summary_box: ctk.CTkTextbox
    transcript_section: ctk.CTkFrame
    transcript_toggle: ctk.CTkButton
    segments_box: ctk.CTkTextbox
    meter_row: ctk.CTkFrame
    meter_bar: ctk.CTkProgressBar
    level_label: ctk.CTkLabel
    backlog_label: ctk.CTkLabel
    dropped_label: ctk.CTkLabel
    warning_label: ctk.CTkLabel
    result_card: ctk.CTkFrame
    result_label: ctk.CTkLabel
    retranscribe_btn: ctk.CTkButton


@dataclass(frozen=True)
class LivePanelCommands:
    """ライブ画面のボタン・入力から呼ぶ処理"""

    on_retranscribe: Callable[[], None]
    on_context_change: Callable[[], None]
    on_context_clear: Callable[[], None]
    on_toggle_transcript: Callable[[], None]


def _card(parent) -> ctk.CTkFrame:
    return ctk.CTkFrame(
        parent,
        fg_color=theme.CARD_BG,
        corner_radius=theme.CARD_RADIUS,
        border_width=1,
        border_color=theme.BORDER,
    )


def _caption(parent, text: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent, text=text, anchor="w", font=theme.font(12), text_color=theme.TEXT_SECONDARY
    )


def _readonly_box(parent, font_size: int, lines: int) -> ctk.CTkTextbox:
    return ctk.CTkTextbox(
        parent,
        height=textbox_height(lines, font_size),
        state="disabled",
        wrap="char",
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(font_size),
        fg_color=theme.CARD_BG,
        text_color=theme.TEXT,
        border_width=1,
        border_color=theme.BORDER,
    )


def build_live_panel(parent, commands: LivePanelCommands) -> LivePanelWidgets:
    """ライブ画面を parent の中に構築して返す(frame 自体はまだ配置しない)

    上から 案内バナー → 参加者・メモ → 要約(主役) → 文字起こし(副・折りたたみ可)。
    メーター・警告・結果は下端から詰め(side="bottom")、要約・文字起こし欄を最後に配置する。
    ウィンドウが足りない時は文字起こし → 要約の順に縮み、下の行やボタンが隠れない。
    """
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    banner_card, banner_label, banner_progress = _build_banner(frame)
    form = build_context_form(frame, commands.on_context_change, commands.on_context_clear)
    form.frame.pack(fill="x", pady=(8, 0))
    meter = _build_meter_row(frame)
    meter[0].pack(side="bottom", fill="x", pady=(6, 0))
    body, summary_box, transcript = _build_body(frame, commands.on_toggle_transcript)
    body.pack(fill="both", expand=True, pady=(8, 0))
    warning_label = ctk.CTkLabel(
        frame,
        text="",
        height=20,
        anchor="w",
        justify="left",
        wraplength=_WRAP_PX,
        font=theme.font(12),
        text_color=theme.RECORDING,
    )
    result_card, result_label, retranscribe_btn = _build_result(frame, commands.on_retranscribe)
    return LivePanelWidgets(
        frame,
        banner_card,
        banner_label,
        banner_progress,
        form,
        body,
        summary_box,
        *transcript,
        *meter,
        warning_label,
        result_card,
        result_label,
        retranscribe_btn,
    )


def _build_banner(frame):
    card = _card(frame)
    label = ctk.CTkLabel(
        card,
        text="",
        anchor="w",
        justify="left",
        wraplength=_WRAP_PX,
        font=theme.font(12),
        text_color=theme.TEXT,
    )
    label.pack(fill="x", padx=theme.CARD_PAD, pady=(8, 4))
    progress = ctk.CTkProgressBar(card, height=8, progress_color=theme.ACCENT)
    progress.set(0)
    return card, label, progress


def _build_body(frame, on_toggle: Callable[[], None]):
    """要約(上・伸縮)と文字起こし(下・数行)。縦幅が足りない時は後に詰めた文字起こしから縮む

    折りたたみボタンは要約の見出し行に置き、文字起こし欄が縮んでも隠れないようにする。
    """
    body = ctk.CTkFrame(frame, fg_color="transparent")
    header = ctk.CTkFrame(body, fg_color="transparent")
    header.pack(fill="x")
    _caption(header, "要約").pack(side="left")
    toggle = ctk.CTkButton(
        header,
        text=transcript_toggle_text(False),
        width=_TOGGLE_WIDTH,
        height=22,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(12),
        command=on_toggle,
        **theme.SECONDARY_STYLE,
    )
    toggle.pack(side="right")
    summary_box = _readonly_box(body, theme.LIVE_SUMMARY_FONT_SIZE, _SUMMARY_MIN_LINES)
    summary_box.pack(fill="both", expand=True, pady=(4, 0))
    section = ctk.CTkFrame(body, fg_color="transparent")
    section.pack(fill="x", pady=(8, 0))
    _caption(section, "文字起こし（速報）").pack(fill="x")
    segments_box = _readonly_box(section, theme.LIVE_TRANSCRIPT_FONT_SIZE, TRANSCRIPT_LINES)
    segments_box.pack(fill="x", pady=(4, 0))
    return body, summary_box, (section, toggle, segments_box)


def _build_meter_row(frame):
    row = ctk.CTkFrame(frame, fg_color="transparent")
    _caption(row, "入力").pack(side="left")
    bar = ctk.CTkProgressBar(row, width=_METER_WIDTH, height=10, progress_color=theme.SUCCESS)
    bar.set(0)
    bar.pack(side="left", padx=(6, 6))
    level = _caption(row, "- dB")
    level.pack(side="left")
    dropped = _caption(row, "破棄 -")
    dropped.pack(side="right")
    backlog = _caption(row, "遅延 -")
    backlog.pack(side="right", padx=(0, 12))
    return row, bar, level, backlog, dropped


def _build_result(frame, on_retranscribe: Callable[[], None]):
    card = _card(frame)
    label = ctk.CTkLabel(
        card,
        text="",
        anchor="w",
        justify="left",
        wraplength=_WRAP_PX - 140,
        font=theme.font(12),
        text_color=theme.TEXT,
    )
    label.pack(side="left", fill="x", expand=True, padx=theme.CARD_PAD, pady=8)
    button = ctk.CTkButton(
        card,
        text="高精度で文字起こし",
        width=140,
        height=30,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        command=on_retranscribe,
        **theme.PRIMARY_STYLE,
    )
    return card, label, button
