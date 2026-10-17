"""App のウィジェット構築処理 (`_build_ui`) を切り出したモジュール。"""

import customtkinter as ctk
from tkinterdnd2 import DND_FILES

from .constants import MODE_RECORD_TRANSCRIBE, MODES


def build_ui(app) -> None:
    """`app` (App インスタンス) にウィジェットを構築し、属性として取り付ける。"""
    header = ctk.CTkFrame(app, fg_color="transparent")
    header.pack(fill="x", padx=20, pady=(12, 4))
    ctk.CTkLabel(header, text="VoiceNote", font=ctk.CTkFont(size=16, weight="bold")).pack(
        side="left"
    )
    app._settings_btn = ctk.CTkButton(
        header, text="設定", width=80, fg_color="gray", command=app._open_settings
    )
    app._settings_btn.pack(side="right")

    ctk.CTkLabel(app, text="モード", anchor="w").pack(fill="x", padx=20, pady=(8, 2))
    app._mode_var = ctk.StringVar(value=MODE_RECORD_TRANSCRIBE)
    mode_frame = ctk.CTkFrame(app, fg_color="transparent")
    mode_frame.pack(fill="x", padx=20, pady=2)
    app._mode_radios = []
    for mode in MODES:
        radio = ctk.CTkRadioButton(
            mode_frame,
            text=mode,
            variable=app._mode_var,
            value=mode,
            command=app._on_mode_change,
        )
        radio.pack(anchor="w", pady=2)
        app._mode_radios.append(radio)

    app._panel = ctk.CTkFrame(app, fg_color="transparent")
    app._panel.pack(fill="x", padx=20, pady=8)

    app._device_section = ctk.CTkFrame(app._panel, fg_color="transparent")
    ctk.CTkLabel(app._device_section, text="入力デバイス", anchor="w").pack(fill="x")
    app._device_var = ctk.StringVar()
    app._device_menu = ctk.CTkOptionMenu(
        app._device_section, variable=app._device_var, values=["（読み込み中）"]
    )
    app._device_menu.pack(fill="x", pady=2)

    app._file_section = ctk.CTkFrame(app._panel, fg_color="transparent")
    ctk.CTkLabel(
        app._file_section, text="音声ファイル（ドラッグ&ドロップ可）", anchor="w"
    ).pack(fill="x")
    file_row = ctk.CTkFrame(app._file_section, fg_color="transparent")
    file_row.pack(fill="x", pady=2)
    app._file_entry = ctk.CTkEntry(file_row, placeholder_text="ファイルを選択してください")
    app._file_entry.pack(side="left", fill="x", expand=True)
    ctk.CTkButton(file_row, text="...", width=36, command=app._browse_audio_file).pack(
        side="left", padx=(4, 0)
    )
    _register_drop_target(app, app._file_entry)
    _register_drop_target(app, app._file_section)

    app._exec_btn = ctk.CTkButton(
        app,
        text="実行",
        height=44,
        font=ctk.CTkFont(size=15),
        command=app._on_exec,
    )
    app._exec_btn.pack(fill="x", padx=20, pady=8)

    app._status_label = ctk.CTkLabel(app, text="待機中", font=ctk.CTkFont(size=14), anchor="w")
    app._status_label.pack(fill="x", padx=20, pady=(4, 2))

    ctk.CTkLabel(app, text="ログ", anchor="w").pack(fill="x", padx=20, pady=(8, 2))
    app._log_box = ctk.CTkTextbox(app, height=160, state="disabled")
    app._log_box.pack(fill="both", expand=True, padx=20, pady=(0, 12))


def _register_drop_target(app, widget) -> None:
    """指定ウィジェットを音声ファイルのドロップ先として登録する。"""
    widget.drop_target_register(DND_FILES)
    widget.dnd_bind("<<Drop>>", app._on_file_drop)
