"""App のウィジェット構築処理 (`_build_ui`) を切り出したモジュール。"""

import customtkinter as ctk
from tkinterdnd2 import DND_FILES

from . import theme
from .constants import MODE_RECORD_TRANSCRIBE, MODES
from .segmented import refresh_segment_text

_LOG_SHOW_TEXT = "詳細ログを表示"
_LOG_HIDE_TEXT = "詳細ログを隠す"


def build_ui(app) -> None:
    """`app` (App インスタンス) にウィジェットを構築し、属性として取り付ける。"""
    app.configure(fg_color=theme.WINDOW_BG)
    _build_header(app)
    _build_mode_selector(app)
    _build_input_panel(app)
    _build_exec_button(app)
    _build_status_card(app)
    _build_log_section(app)


def toggle_log(app) -> None:
    """詳細ログの表示/非表示を切り替え、ウィンドウ高さを合わせる。"""
    app._log_visible = not app._log_visible
    if app._log_visible:
        app._log_box.pack(fill="both", expand=True, pady=(6, 0))
        app._log_toggle.configure(text=_LOG_HIDE_TEXT)
        app.geometry(f"{app.WINDOW_WIDTH}x{app.WINDOW_HEIGHT_EXPANDED}")
    else:
        app._log_box.pack_forget()
        app._log_toggle.configure(text=_LOG_SHOW_TEXT)
        app.geometry(f"{app.WINDOW_WIDTH}x{app.WINDOW_HEIGHT_COMPACT}")


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


def _build_header(app) -> None:
    header = ctk.CTkFrame(app, fg_color="transparent")
    header.pack(fill="x", padx=theme.SCREEN_PAD, pady=(theme.SCREEN_PAD, 8))
    ctk.CTkLabel(
        header, text="VoiceNote", font=theme.font(20, "bold"), text_color=theme.TEXT
    ).pack(side="left")
    app._settings_btn = ctk.CTkButton(
        header,
        text="設定",
        width=64,
        height=30,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        command=app._open_settings,
        **theme.SECONDARY_STYLE,
    )
    app._settings_btn.pack(side="right")


def _build_mode_selector(app) -> None:
    app._mode_var = ctk.StringVar(value=MODE_RECORD_TRANSCRIBE)
    app._mode_selector = ctk.CTkSegmentedButton(
        app,
        values=MODES,
        variable=app._mode_var,
        command=lambda _value: _on_mode_selected(app),
        height=34,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(12),
        selected_color=theme.ACCENT,
        selected_hover_color=theme.ACCENT_HOVER,
        unselected_color=theme.SECONDARY_BTN,
        unselected_hover_color=theme.SECONDARY_BTN_HOVER,
    )
    refresh_segment_text(app._mode_selector)
    app._mode_selector.pack(fill="x", padx=theme.SCREEN_PAD, pady=(0, 12))


def _on_mode_selected(app) -> None:
    refresh_segment_text(app._mode_selector)
    app._on_mode_change()


def _build_input_panel(app) -> None:
    app._panel = ctk.CTkFrame(app, fg_color="transparent")
    app._panel.pack(fill="x", padx=theme.SCREEN_PAD)
    _build_device_section(app)
    _build_file_section(app)


def _build_device_section(app) -> None:
    app._device_section = _card(app._panel)
    inner = ctk.CTkFrame(app._device_section, fg_color="transparent")
    inner.pack(fill="x", padx=theme.CARD_PAD, pady=theme.CARD_PAD)
    _caption(inner, "入力デバイス").pack(fill="x")
    app._device_var = ctk.StringVar()
    app._device_menu = ctk.CTkOptionMenu(
        inner,
        variable=app._device_var,
        values=["（読み込み中）"],
        height=32,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        dropdown_font=theme.font(13),
        **_option_menu_colors(),
    )
    app._device_menu.pack(fill="x", pady=(4, 0))


def _option_menu_colors() -> dict:
    return {
        "fg_color": theme.SECONDARY_BTN,
        "button_color": theme.SECONDARY_BTN_HOVER,
        "button_hover_color": theme.SECONDARY_BTN_HOVER,
        "text_color": theme.TEXT,
    }


def _build_file_section(app) -> None:
    app._file_section = _card(app._panel)
    inner = ctk.CTkFrame(app._file_section, fg_color="transparent")
    inner.pack(fill="x", padx=theme.CARD_PAD, pady=theme.CARD_PAD)
    _caption(inner, "音声ファイル（ここにドラッグ&ドロップ）").pack(fill="x")
    file_row = ctk.CTkFrame(inner, fg_color="transparent")
    file_row.pack(fill="x", pady=(4, 0))
    app._file_entry = ctk.CTkEntry(
        file_row,
        height=32,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        placeholder_text="ファイルを選択してください",
    )
    app._file_entry.pack(side="left", fill="x", expand=True)
    ctk.CTkButton(
        file_row,
        text="選択",
        width=56,
        height=32,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        command=app._browse_audio_file,
        **theme.SECONDARY_STYLE,
    ).pack(side="left", padx=(6, 0))
    for target in (app._file_entry, app._file_section, inner, file_row):
        _register_drop_target(app, target)


def _build_exec_button(app) -> None:
    app._exec_btn = ctk.CTkButton(
        app,
        text="実行",
        height=48,
        corner_radius=theme.CARD_RADIUS,
        font=theme.font(15, "bold"),
        command=app._on_exec,
        **theme.PRIMARY_STYLE,
    )
    app._exec_btn.pack(fill="x", padx=theme.SCREEN_PAD, pady=(14, 10))


def _build_status_card(app) -> None:
    card = _card(app)
    card.pack(fill="x", padx=theme.SCREEN_PAD)
    row = ctk.CTkFrame(card, fg_color="transparent")
    row.pack(fill="x", padx=theme.CARD_PAD, pady=10)
    app._status_dot = ctk.CTkLabel(
        row, text="●", width=16, font=theme.font(12), text_color=theme.TEXT_SECONDARY
    )
    app._status_dot.pack(side="left")
    app._status_label = ctk.CTkLabel(
        row, text="待機中", anchor="w", font=theme.font(14), text_color=theme.TEXT
    )
    app._status_label.pack(side="left", fill="x", expand=True)
    app._timer_label = ctk.CTkLabel(
        row, text="", font=theme.font(14, "bold"), text_color=theme.TEXT_SECONDARY
    )
    app._timer_label.pack(side="right")


def _build_log_section(app) -> None:
    container = ctk.CTkFrame(app, fg_color="transparent")
    container.pack(fill="both", expand=True, padx=theme.SCREEN_PAD, pady=(8, theme.SCREEN_PAD))
    app._log_toggle = ctk.CTkButton(
        container,
        text=_LOG_SHOW_TEXT,
        anchor="w",
        height=24,
        width=0,
        fg_color="transparent",
        hover=False,
        text_color=theme.TEXT_SECONDARY,
        font=theme.font(12),
        command=lambda: toggle_log(app),
    )
    app._log_toggle.pack(anchor="w")
    app._log_box = ctk.CTkTextbox(
        container,
        height=160,
        state="disabled",
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(12),
        fg_color=theme.CARD_BG,
        text_color=theme.TEXT,
        border_width=1,
        border_color=theme.BORDER,
    )
    app._log_visible = False


def _register_drop_target(app, widget) -> None:
    """指定ウィジェットを音声ファイルのドロップ先として登録する。"""
    widget.drop_target_register(DND_FILES)
    widget.dnd_bind("<<Drop>>", app._on_file_drop)
