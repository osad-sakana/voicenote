"""設定ダイアログ。保存フォルダ・文字起こしモード・モデル・APIキーを編集する。"""

from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from config import VoiceNoteConfig

from . import theme
from .segmented import refresh_segment_text

_MODE_LABELS = {"local": "ローカル (faster-whisper)", "openai": "OpenAI API"}


class SettingsDialog(ctk.CTkToplevel):
    """設定ダイアログ"""

    def __init__(self, parent, config: VoiceNoteConfig):
        super().__init__(parent)
        self.title("設定")
        self.geometry("480x480")
        self.resizable(False, False)
        self.grab_set()

        self._config = config
        self._result: VoiceNoteConfig | None = None

        self._build_ui()
        self._load_values()

    def _build_ui(self):
        self.configure(fg_color=theme.WINDOW_BG)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=theme.SCREEN_PAD, pady=(theme.SCREEN_PAD, 0))

        self._folder_entry = self._folder_field(
            body, "文字起こし保存フォルダ（絶対パス）", self._browse_folder
        )
        self._rec_folder_entry = self._folder_field(
            body, "録音の保存場所（空欄でデスクトップ）", self._browse_rec_folder
        )

        self._caption(body, "文字起こしモード")
        self._mode_var = ctk.StringVar(value="local")
        self._mode_selector = ctk.CTkSegmentedButton(
            body,
            values=list(_MODE_LABELS.values()),
            command=self._on_mode_label_change,
            height=32,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            selected_color=theme.ACCENT,
            selected_hover_color=theme.ACCENT_HOVER,
            unselected_color=theme.SECONDARY_BTN,
            unselected_hover_color=theme.SECONDARY_BTN_HOVER,
        )
        refresh_segment_text(self._mode_selector)
        self._mode_selector.pack(fill="x", pady=(0, 4))

        self._model_label = self._caption(body, "Whisperモデル")
        self._model_var = ctk.StringVar(value="small")
        self._model_menu = ctk.CTkOptionMenu(
            body,
            variable=self._model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
            height=32,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            dropdown_font=theme.font(13),
            fg_color=theme.SECONDARY_BTN,
            button_color=theme.SECONDARY_BTN_HOVER,
            button_hover_color=theme.SECONDARY_BTN_HOVER,
            text_color=theme.TEXT,
        )
        self._model_menu.pack(fill="x")

        self._apikey_label = self._caption(body, "OpenAI APIキー")
        self._apikey_entry = ctk.CTkEntry(
            body, show="*", height=32, corner_radius=theme.CONTROL_RADIUS, font=theme.font(13)
        )
        self._apikey_entry.pack(fill="x")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=theme.SCREEN_PAD, pady=theme.SCREEN_PAD)
        ctk.CTkButton(
            btn_frame,
            text="キャンセル",
            height=36,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            command=self.destroy,
            **theme.SECONDARY_STYLE,
        ).pack(side="left", expand=True, fill="x", padx=(0, 8))
        ctk.CTkButton(
            btn_frame,
            text="保存",
            height=36,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13, "bold"),
            command=self._save,
            **theme.PRIMARY_STYLE,
        ).pack(side="left", expand=True, fill="x")

    def _caption(self, parent, text: str) -> ctk.CTkLabel:
        label = ctk.CTkLabel(
            parent, text=text, anchor="w", font=theme.font(12), text_color=theme.TEXT_SECONDARY
        )
        label.pack(fill="x", pady=(10, 4))
        return label

    def _folder_field(self, parent, caption: str, on_browse) -> ctk.CTkEntry:
        self._caption(parent, caption)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        entry = ctk.CTkEntry(row, height=32, corner_radius=theme.CONTROL_RADIUS, font=theme.font(13))
        entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            row,
            text="選択",
            width=56,
            height=32,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            command=on_browse,
            **theme.SECONDARY_STYLE,
        ).pack(side="left", padx=(6, 0))
        return entry

    def _on_mode_label_change(self, label: str):
        refresh_segment_text(self._mode_selector)
        self._mode_var.set(next(key for key, value in _MODE_LABELS.items() if value == label))
        self._on_mode_change()

    def _load_values(self):
        self._folder_entry.insert(0, self._config.save_folder)
        self._rec_folder_entry.insert(0, self._config.recording_folder)
        self._mode_var.set(self._config.transcription_mode)
        self._mode_selector.set(_MODE_LABELS[self._config.transcription_mode])
        refresh_segment_text(self._mode_selector)
        self._model_var.set(self._config.whisper_model)
        self._apikey_entry.insert(0, self._config.openai_api_key or "")
        self._on_mode_change()

    def _on_mode_change(self):
        is_local = self._mode_var.get() == "local"
        active, inactive = theme.TEXT, theme.TEXT_SECONDARY
        self._model_label.configure(text_color=active if is_local else inactive)
        self._model_menu.configure(state="normal" if is_local else "disabled")
        self._apikey_label.configure(text_color=inactive if is_local else active)
        self._apikey_entry.configure(state="normal" if not is_local else "disabled")

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="保存フォルダを選択")
        if folder:
            self._folder_entry.delete(0, "end")
            self._folder_entry.insert(0, folder)

    def _browse_rec_folder(self):
        folder = filedialog.askdirectory(title="録音の保存場所を選択")
        if folder:
            self._rec_folder_entry.delete(0, "end")
            self._rec_folder_entry.insert(0, folder)

    def _save(self):
        save_folder = self._folder_entry.get().strip()
        if not save_folder:
            messagebox.showerror("エラー", "保存フォルダを指定してください", parent=self)
            return
        folder_path = Path(save_folder).expanduser()
        if not folder_path.parent.exists():
            messagebox.showerror(
                "エラー", f"親ディレクトリが存在しません:\n{folder_path.parent}", parent=self
            )
            return

        rec_folder = self._rec_folder_entry.get().strip()
        if rec_folder:
            rec_folder_path = Path(rec_folder).expanduser()
            if not rec_folder_path.parent.exists():
                messagebox.showerror(
                    "エラー",
                    f"録音の保存場所: 親ディレクトリが存在しません:\n{rec_folder_path.parent}",
                    parent=self,
                )
                return
            rec_folder = str(rec_folder_path)

        api_key = self._apikey_entry.get().strip()
        self._result = replace(
            self._config,
            save_folder=str(folder_path),
            recording_folder=rec_folder,
            whisper_model=self._model_var.get(),
            transcription_mode=self._mode_var.get(),
            openai_api_key=api_key or None,
        )
        self.destroy()

    def get_result(self) -> VoiceNoteConfig | None:
        return self._result
