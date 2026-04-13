#!/usr/bin/env python3
"""
VoiceNote GUIエントリーポイント（CustomTkinter）
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from dotenv import load_dotenv
from scipy.io import wavfile

load_dotenv()

from config import load_config, save_config
from obsidian import save_to_obsidian
from recorder import SAMPLE_RATE, ThreadedRecorder, list_devices
from transcriber import transcribe_audio, transcribe_audio_openai

CONFIG_PATH = Path.home() / ".config" / "voicenote" / "config.json"

MODE_RECORD_TRANSCRIBE = "録音して文字起こしする"
MODE_RECORD_ONLY = "録音だけする"
MODE_TRANSCRIBE_ONLY = "文字起こしだけする"
MODES = [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY, MODE_TRANSCRIBE_ONLY]

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


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

        # 保存フォルダ
        ctk.CTkLabel(self, text="文字起こし保存フォルダ（絶対パス）", anchor="w").pack(fill="x", **pad)
        folder_frame = ctk.CTkFrame(self, fg_color="transparent")
        folder_frame.pack(fill="x", padx=20, pady=2)
        self._folder_entry = ctk.CTkEntry(folder_frame)
        self._folder_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(folder_frame, text="📁", width=36, command=self._browse_folder).pack(side="left", padx=(4, 0))

        # 文字起こしモード
        ctk.CTkLabel(self, text="文字起こしモード", anchor="w").pack(fill="x", **pad)
        self._mode_var = ctk.StringVar(value="local")
        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.pack(fill="x", padx=20, pady=2)
        ctk.CTkRadioButton(mode_frame, text="ローカル (faster-whisper)", variable=self._mode_var, value="local", command=self._on_mode_change).pack(side="left", padx=(0, 16))
        ctk.CTkRadioButton(mode_frame, text="OpenAI API", variable=self._mode_var, value="openai", command=self._on_mode_change).pack(side="left")

        # モデル選択（ローカル用）
        self._model_label = ctk.CTkLabel(self, text="Whisperモデル", anchor="w")
        self._model_label.pack(fill="x", **pad)
        self._model_var = ctk.StringVar(value="small")
        self._model_menu = ctk.CTkOptionMenu(
            self,
            variable=self._model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
        )
        self._model_menu.pack(fill="x", padx=20, pady=2)

        # APIキー（OpenAI用）
        self._apikey_label = ctk.CTkLabel(self, text="OpenAI APIキー", anchor="w")
        self._apikey_label.pack(fill="x", **pad)
        self._apikey_entry = ctk.CTkEntry(self, show="●")
        self._apikey_entry.pack(fill="x", padx=20, pady=2)

        # ボタン
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=16)
        ctk.CTkButton(btn_frame, text="キャンセル", fg_color="gray", command=self.destroy).pack(side="left", expand=True, padx=(0, 8))
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
            messagebox.showerror("エラー", f"親ディレクトリが存在しません:\n{folder_path.parent}", parent=self)
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


class App(ctk.CTk):
    """メインウィンドウ"""

    def __init__(self):
        super().__init__()
        self.title("VoiceNote")
        self.geometry("480x560")
        self.resizable(False, False)

        self._config: dict = {}
        self._recorder: ThreadedRecorder | None = None
        self._recording = False
        self._elapsed = 0
        self._alive = True

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._load_config()
        self._refresh_devices()
        self._on_mode_change()

    # ──────────────── UI構築 ────────────────

    def _build_ui(self):
        # ヘッダー（設定ボタン）
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(12, 4))
        ctk.CTkLabel(header, text="VoiceNote", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="⚙  設定", width=80, fg_color="gray", command=self._open_settings).pack(side="right")

        # モード選択
        ctk.CTkLabel(self, text="モード", anchor="w").pack(fill="x", padx=20, pady=(8, 2))
        self._mode_var = ctk.StringVar(value=MODE_RECORD_TRANSCRIBE)
        ctk.CTkOptionMenu(
            self,
            variable=self._mode_var,
            values=MODES,
            command=lambda _: self._on_mode_change(),
        ).pack(fill="x", padx=20, pady=2)

        # 動的パネル（モードに応じて表示切替）
        self._panel = ctk.CTkFrame(self, fg_color="transparent")
        self._panel.pack(fill="x", padx=20, pady=8)

        # --- 録音関連ウィジェット ---
        self._device_section = ctk.CTkFrame(self._panel, fg_color="transparent")
        ctk.CTkLabel(self._device_section, text="入力デバイス", anchor="w").pack(fill="x")
        self._device_var = ctk.StringVar()
        self._device_menu = ctk.CTkOptionMenu(self._device_section, variable=self._device_var, values=["（読み込み中）"])
        self._device_menu.pack(fill="x", pady=2)

        self._rec_dest_section = ctk.CTkFrame(self._panel, fg_color="transparent")
        ctk.CTkLabel(self._rec_dest_section, text="録音の保存場所", anchor="w").pack(fill="x")
        rec_dest_row = ctk.CTkFrame(self._rec_dest_section, fg_color="transparent")
        rec_dest_row.pack(fill="x", pady=2)
        self._rec_dest_entry = ctk.CTkEntry(rec_dest_row)
        self._rec_dest_entry.insert(0, str(Path.home() / "Desktop"))
        self._rec_dest_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(rec_dest_row, text="📁", width=36, command=self._browse_rec_dest).pack(side="left", padx=(4, 0))

        # --- 文字起こしのみ用ウィジェット ---
        self._file_section = ctk.CTkFrame(self._panel, fg_color="transparent")
        ctk.CTkLabel(self._file_section, text="音声ファイル", anchor="w").pack(fill="x")
        file_row = ctk.CTkFrame(self._file_section, fg_color="transparent")
        file_row.pack(fill="x", pady=2)
        self._file_entry = ctk.CTkEntry(file_row, placeholder_text="ファイルを選択してください")
        self._file_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(file_row, text="📂", width=36, command=self._browse_audio_file).pack(side="left", padx=(4, 0))

        # 実行ボタン
        self._exec_btn = ctk.CTkButton(
            self, text="実行", height=44, font=ctk.CTkFont(size=15),
            command=self._on_exec,
        )
        self._exec_btn.pack(fill="x", padx=20, pady=8)

        # 状態表示
        self._status_label = ctk.CTkLabel(
            self, text="待機中", font=ctk.CTkFont(size=14), anchor="w"
        )
        self._status_label.pack(fill="x", padx=20, pady=(4, 2))

        # ログ
        ctk.CTkLabel(self, text="ログ", anchor="w").pack(fill="x", padx=20, pady=(8, 2))
        self._log_box = ctk.CTkTextbox(self, height=160, state="disabled")
        self._log_box.pack(fill="both", expand=True, padx=20, pady=(0, 12))

    def _on_mode_change(self):
        mode = self._mode_var.get()
        # 全セクションをいったん非表示
        for section in (self._device_section, self._rec_dest_section, self._file_section):
            section.pack_forget()

        if mode == MODE_RECORD_TRANSCRIBE:
            self._device_section.pack(fill="x", pady=(0, 8))
            self._rec_dest_section.pack(fill="x")
            self._exec_btn.configure(text="実行")
        elif mode == MODE_RECORD_ONLY:
            self._device_section.pack(fill="x", pady=(0, 8))
            self._rec_dest_section.pack(fill="x")
            self._exec_btn.configure(text="実行")
        elif mode == MODE_TRANSCRIBE_ONLY:
            self._file_section.pack(fill="x")
            self._exec_btn.configure(text="文字起こし開始")

    # ──────────────── 設定 ────────────────

    def _load_config(self):
        config = load_config(CONFIG_PATH)
        if config:
            self._config = config
            if not os.environ.get("OPENAI_API_KEY") and config.get("openai_api_key"):
                os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
            self._log("設定を読み込みました")
        else:
            self._log("設定が見つかりません。⚙ 設定から保存フォルダを設定してください")

    def _open_settings(self):
        dialog = SettingsDialog(self, self._config)
        self.wait_window(dialog)
        result = dialog.get_result()
        if result is None:
            return
        try:
            save_config(CONFIG_PATH, result)
            self._config = result
            if not os.environ.get("OPENAI_API_KEY") and result.get("openai_api_key"):
                os.environ["OPENAI_API_KEY"] = result["openai_api_key"]
            self._log("設定を保存しました")
        except RuntimeError as e:
            messagebox.showerror("エラー", str(e))

    # ──────────────── デバイス ────────────────

    def _refresh_devices(self):
        try:
            devices = list_devices()
            names = [f"[{d['id']}] {d['name']}" for d in devices]
            if not names:
                names = ["デバイスなし"]
            self._device_menu.configure(values=names)
            self._device_var.set(names[0])
        except Exception as e:
            self._log(f"デバイス取得エラー: {e}")

    def _selected_device_id(self) -> int | None:
        val = self._device_var.get()
        if val.startswith("["):
            try:
                return int(val.split("]")[0][1:])
            except ValueError:
                pass
        return None

    # ──────────────── ファイル選択 ────────────────

    def _browse_rec_dest(self):
        folder = filedialog.askdirectory(title="録音の保存場所を選択")
        if folder:
            self._rec_dest_entry.delete(0, "end")
            self._rec_dest_entry.insert(0, folder)

    def _browse_audio_file(self):
        path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[("音声ファイル", "*.wav *.mp3 *.m4a *.ogg *.flac"), ("すべて", "*.*")],
        )
        if path:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, path)

    # ──────────────── 実行 ────────────────

    def _on_exec(self):
        mode = self._mode_var.get()
        if mode in (MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY):
            if self._recording:
                self._stop_recording()
            else:
                self._start_recording()
        elif mode == MODE_TRANSCRIBE_ONLY:
            self._run_transcribe_only()

    def _start_recording(self):
        mode = self._mode_var.get()
        if mode == MODE_RECORD_TRANSCRIBE and not self._config.get("save_folder"):
            messagebox.showwarning("設定が必要", "先に ⚙ 設定から文字起こし保存フォルダを設定してください")
            return

        device_id = self._selected_device_id()
        self._recorder = ThreadedRecorder(device_id)
        try:
            self._recorder.start()
        except Exception as e:
            messagebox.showerror("録音エラー", str(e))
            return

        self._recording = True
        self._elapsed = 0
        self._exec_btn.configure(text="⏹  録音停止", fg_color="red")
        self._log("録音開始")
        threading.Thread(target=self._timer_loop, daemon=True).start()

    def _stop_recording(self):
        self._recording = False
        self._exec_btn.configure(text="実行", fg_color=["#3B8ED0", "#1F6AA5"])
        self._set_status("保存中...")
        self._log("録音停止")

        if self._recorder is None:
            return
        self._recorder.stop()
        try:
            audio_data = self._recorder.get_data()
        except RuntimeError as e:
            self._set_status("エラー")
            self._log(f"エラー: {e}")
            return

        # ウィジェットの値はメインスレッドで取得してからスレッドに渡す
        rec_dest = Path(self._rec_dest_entry.get().strip() or str(Path.home() / "Desktop"))
        mode = self._mode_var.get()
        threading.Thread(target=self._process_audio, args=(audio_data, rec_dest, mode), daemon=True).start()

    def _timer_loop(self):
        while self._recording:
            mins, secs = divmod(self._elapsed, 60)
            self._safe_after(self._set_status, f"⏺  {mins:02d}:{secs:02d}  録音中...")
            time.sleep(1)
            self._elapsed += 1

    # ──────────────── 音声処理 ────────────────

    def _process_audio(self, audio_data: np.ndarray, rec_dest: Path, mode: str):
        try:
            audio_file = self._write_wav(audio_data, rec_dest)
            self._safe_after(self._log, f"音声ファイルを保存: {audio_file.name}")
        except Exception as e:
            self._safe_after(self._set_status, f"エラー: {e}")
            self._safe_after(self._log, f"エラー: {e}")
            return

        if mode == MODE_RECORD_ONLY:
            self._safe_after(self._set_status, "完了!")
            self._safe_after(self._log, f"保存完了 → {audio_file}")
            return

        self._run_transcription(audio_file)

    def _write_wav(self, audio_data: np.ndarray, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        audio_file = dest_dir / f"{timestamp}_recording.wav"
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(audio_file, SAMPLE_RATE, audio_int16)
        return audio_file

    def _run_transcribe_only(self):
        audio_path = self._file_entry.get().strip()
        if not audio_path:
            messagebox.showwarning("ファイル未選択", "音声ファイルを選択してください")
            return
        if not self._config.get("save_folder"):
            messagebox.showwarning("設定が必要", "先に ⚙ 設定から文字起こし保存フォルダを設定してください")
            return
        self._exec_btn.configure(state="disabled")
        threading.Thread(target=self._run_transcription, args=(Path(audio_path),), daemon=True).start()

    def _run_transcription(self, audio_file: Path):
        mode = self._config.get("transcription_mode", "local")
        start_time = time.time()

        def on_progress(msg: str):
            elapsed = time.time() - start_time
            self._safe_after(self._set_status, f"{msg} ({elapsed:.1f}s)")
            self._safe_after(self._log, msg)

        try:
            if mode == "openai":
                transcription = transcribe_audio_openai(audio_file, progress_callback=on_progress)
            else:
                transcription = transcribe_audio(
                    audio_file,
                    self._config.get("whisper_model", "small"),
                    progress_callback=on_progress,
                )
        except Exception as e:
            self._safe_after(self._set_status, f"エラー: {e}")
            self._safe_after(self._log, f"文字起こしエラー: {e}")
            self._safe_after(self._exec_btn.configure, {"state": "normal"})
            return

        try:
            saved_path = save_to_obsidian(Path(self._config["save_folder"]), transcription)
        except RuntimeError as e:
            self._safe_after(self._set_status, f"エラー: {e}")
            self._safe_after(self._log, f"保存エラー: {e}")
            self._safe_after(self._exec_btn.configure, {"state": "normal"})
            return

        self._safe_after(self._set_status, "完了!")
        self._safe_after(self._log, f"保存完了 → {saved_path}")
        self._safe_after(self._exec_btn.configure, {"state": "normal"})

    # ──────────────── ウィンドウ終了 ────────────────

    def _on_close(self):
        self._alive = False
        self._recording = False
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                pass
        self.destroy()

    # ──────────────── ユーティリティ ────────────────

    def _safe_after(self, fn, *args):
        """バックグラウンドスレッドからスレッドセーフにUI更新をスケジュールする"""
        if self._alive:
            self.after(0, fn, *args)

    def _set_status(self, text: str):
        self._status_label.configure(text=text)

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"[{timestamp}] {msg}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
