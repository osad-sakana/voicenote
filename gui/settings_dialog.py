"""設定ダイアログ。保存フォルダ・文字起こしモード・モデル・APIキーを編集する。"""

from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk


class SettingsDialog(ctk.CTkToplevel):
    """設定ダイアログ"""

    def __init__(self, parent, config: dict):
        super().__init__(parent)
        self.title("設定")
        self.geometry("480x380")
        self.resizable(False, False)
        self.grab_set()

        self._config = config
        self._result: dict | None = None

        self._build_ui()
        self._load_values()

    def _build_ui(self):
        pad = {"padx": 20, "pady": 6}

        ctk.CTkLabel(self, text="文字起こし保存フォルダ（絶対パス）", anchor="w").pack(
            fill="x", **pad
        )
        folder_frame = ctk.CTkFrame(self, fg_color="transparent")
        folder_frame.pack(fill="x", padx=20, pady=2)
        self._folder_entry = ctk.CTkEntry(folder_frame)
        self._folder_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(folder_frame, text="...", width=36, command=self._browse_folder).pack(
            side="left", padx=(4, 0)
        )

        ctk.CTkLabel(self, text="文字起こしモード", anchor="w").pack(fill="x", **pad)
        self._mode_var = ctk.StringVar(value="local")
        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.pack(fill="x", padx=20, pady=2)
        ctk.CTkRadioButton(
            mode_frame,
            text="ローカル (faster-whisper)",
            variable=self._mode_var,
            value="local",
            command=self._on_mode_change,
        ).pack(side="left", padx=(0, 16))
        ctk.CTkRadioButton(
            mode_frame,
            text="OpenAI API",
            variable=self._mode_var,
            value="openai",
            command=self._on_mode_change,
        ).pack(side="left")

        self._model_label = ctk.CTkLabel(self, text="Whisperモデル", anchor="w")
        self._model_label.pack(fill="x", **pad)
        self._model_var = ctk.StringVar(value="small")
        self._model_menu = ctk.CTkOptionMenu(
            self,
            variable=self._model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
        )
        self._model_menu.pack(fill="x", padx=20, pady=2)

        self._apikey_label = ctk.CTkLabel(self, text="OpenAI APIキー", anchor="w")
        self._apikey_label.pack(fill="x", **pad)
        self._apikey_entry = ctk.CTkEntry(self, show="*")
        self._apikey_entry.pack(fill="x", padx=20, pady=2)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=16)
        ctk.CTkButton(btn_frame, text="キャンセル", fg_color="gray", command=self.destroy).pack(
            side="left", expand=True, padx=(0, 8)
        )
        ctk.CTkButton(btn_frame, text="保存", command=self._save).pack(side="left", expand=True)

    def _load_values(self):
        self._folder_entry.insert(0, self._config.get("save_folder", ""))
        self._mode_var.set(self._config.get("transcription_mode", "local"))
        self._model_var.set(self._config.get("whisper_model", "small"))
        self._apikey_entry.insert(0, self._config.get("openai_api_key", ""))
        self._on_mode_change()

    def _on_mode_change(self):
        is_local = self._mode_var.get() == "local"
        self._model_label.configure(text_color=("black", "white") if is_local else "gray")
        self._model_menu.configure(state="normal" if is_local else "disabled")
        self._apikey_label.configure(text_color=("black", "white") if not is_local else "gray")
        self._apikey_entry.configure(state="normal" if not is_local else "disabled")

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="保存フォルダを選択")
        if folder:
            self._folder_entry.delete(0, "end")
            self._folder_entry.insert(0, folder)

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
        config: dict = {
            "save_folder": str(folder_path),
            "whisper_model": self._model_var.get(),
            "transcription_mode": self._mode_var.get(),
        }
        api_key = self._apikey_entry.get().strip()
        if api_key:
            config["openai_api_key"] = api_key
        self._result = config
        self.destroy()

    def get_result(self) -> dict | None:
        return self._result
