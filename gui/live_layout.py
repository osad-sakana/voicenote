"""ライブモード画面のウィジェット構築(配置だけを行い、表示更新は gui/live_panel.py)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import customtkinter as ctk

from . import theme

_METER_WIDTH = 160
_TEXTBOX_HEIGHT = 64  # 要求高さは小さくし、余った縦幅は expand で区間・要約欄に回す
_WRAP_PX = 560  # バナー・結果の折り返し幅(ウィンドウ最小幅に合わせる)


@dataclass(frozen=True)
class LivePanelWidgets:
    """ライブ画面のウィジェット一式"""

    frame: ctk.CTkFrame
    banner_card: ctk.CTkFrame
    banner_label: ctk.CTkLabel
    banner_progress: ctk.CTkProgressBar
    body: ctk.CTkFrame
    segments_box: ctk.CTkTextbox
    summary_box: ctk.CTkTextbox
    meter_row: ctk.CTkFrame
    meter_bar: ctk.CTkProgressBar
    level_label: ctk.CTkLabel
    backlog_label: ctk.CTkLabel
    dropped_label: ctk.CTkLabel
    warning_label: ctk.CTkLabel
    result_card: ctk.CTkFrame
    result_label: ctk.CTkLabel
    retranscribe_btn: ctk.CTkButton


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


def _readonly_box(parent) -> ctk.CTkTextbox:
    return ctk.CTkTextbox(
        parent,
        height=_TEXTBOX_HEIGHT,
        state="disabled",
        wrap="char",
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        fg_color=theme.CARD_BG,
        text_color=theme.TEXT,
        border_width=1,
        border_color=theme.BORDER,
    )


def build_live_panel(parent, on_retranscribe: Callable[[], None]) -> LivePanelWidgets:
    """ライブ画面を parent の中に構築して返す(frame 自体はまだ配置しない)

    メーター・警告・結果は下端から詰め(side="bottom")、区間・要約欄を最後に配置する。
    ウィンドウが足りない時に縮むのが区間・要約欄になり、下の行やボタンが隠れない。
    """
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    banner_card, banner_label, banner_progress = _build_banner(frame)
    meter = _build_meter_row(frame)
    meter[0].pack(side="bottom", fill="x", pady=(6, 0))
    body, segments_box, summary_box = _build_body(frame)
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
    result_card, result_label, retranscribe_btn = _build_result(frame, on_retranscribe)
    return LivePanelWidgets(
        frame,
        banner_card,
        banner_label,
        banner_progress,
        body,
        segments_box,
        summary_box,
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


def _build_body(frame):
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.grid_columnconfigure(0, weight=3, uniform="live")
    body.grid_columnconfigure(1, weight=2, uniform="live")
    body.grid_rowconfigure(1, weight=1)
    _caption(body, "文字起こし（速報）").grid(row=0, column=0, sticky="w")
    _caption(body, "要約").grid(row=0, column=1, sticky="w", padx=(8, 0))
    segments_box = _readonly_box(body)
    segments_box.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
    summary_box = _readonly_box(body)
    summary_box.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(4, 0))
    return body, segments_box, summary_box


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
