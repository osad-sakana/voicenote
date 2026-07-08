"""VoiceNote GUI のメインウィンドウ。"""

import contextlib
import logging
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from config import CONFIG_PATH, InvalidConfigError, VoiceNoteConfig, load_config, save_config
from pipeline import save_wav, transcribe_and_save
from recorder import ThreadedRecorder, list_devices

from .constants import (
    MODE_RECORD_ONLY,
    MODE_RECORD_TRANSCRIBE,
    MODE_TRANSCRIBE_ONLY,
    MODES,
    PRIMARY_BUTTON_COLOR,
    strip_emoji,
)
from .settings_dialog import SettingsDialog
from .ui_queue import ThreadSafeUIQueue

_logger = logging.getLogger("voicenote")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    """メインウィンドウ"""

    def __init__(self, log_file: Path):
        super().__init__()
        self.title("VoiceNote")
        self.geometry("480x560")
        self.resizable(False, False)

        self._log_file = log_file
        self._config: VoiceNoteConfig = VoiceNoteConfig()
        self._recorder: ThreadedRecorder | None = None
        self._recording = False
        self._elapsed = 0
        self._alive = True
        self._ui_queue = ThreadSafeUIQueue(self, alive_fn=lambda: self._alive)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._load_config()
        self._refresh_devices()
        self._on_mode_change()
        self._ui_queue.start()

    # ──────────────── UI構築 ────────────────

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(12, 4))
        ctk.CTkLabel(header, text="VoiceNote", font=ctk.CTkFont(size=16, weight="bold")).pack(
            side="left"
        )
        self._settings_btn = ctk.CTkButton(
            header, text="設定", width=80, fg_color="gray", command=self._open_settings
        )
        self._settings_btn.pack(side="right")

        ctk.CTkLabel(self, text="モード", anchor="w").pack(fill="x", padx=20, pady=(8, 2))
        self._mode_var = ctk.StringVar(value=MODE_RECORD_TRANSCRIBE)
        self._mode_menu = ctk.CTkOptionMenu(
            self,
            variable=self._mode_var,
            values=MODES,
            command=lambda _: self._on_mode_change(),
        )
        self._mode_menu.pack(fill="x", padx=20, pady=2)

        self._panel = ctk.CTkFrame(self, fg_color="transparent")
        self._panel.pack(fill="x", padx=20, pady=8)

        self._device_section = ctk.CTkFrame(self._panel, fg_color="transparent")
        ctk.CTkLabel(self._device_section, text="入力デバイス", anchor="w").pack(fill="x")
        self._device_var = ctk.StringVar()
        self._device_menu = ctk.CTkOptionMenu(
            self._device_section, variable=self._device_var, values=["（読み込み中）"]
        )
        self._device_menu.pack(fill="x", pady=2)

        self._rec_dest_section = ctk.CTkFrame(self._panel, fg_color="transparent")
        ctk.CTkLabel(self._rec_dest_section, text="録音の保存場所", anchor="w").pack(fill="x")
        rec_dest_row = ctk.CTkFrame(self._rec_dest_section, fg_color="transparent")
        rec_dest_row.pack(fill="x", pady=2)
        self._rec_dest_entry = ctk.CTkEntry(rec_dest_row)
        self._rec_dest_entry.insert(0, str(Path.home() / "Desktop"))
        self._rec_dest_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(rec_dest_row, text="...", width=36, command=self._browse_rec_dest).pack(
            side="left", padx=(4, 0)
        )

        self._file_section = ctk.CTkFrame(self._panel, fg_color="transparent")
        ctk.CTkLabel(self._file_section, text="音声ファイル", anchor="w").pack(fill="x")
        file_row = ctk.CTkFrame(self._file_section, fg_color="transparent")
        file_row.pack(fill="x", pady=2)
        self._file_entry = ctk.CTkEntry(file_row, placeholder_text="ファイルを選択してください")
        self._file_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(file_row, text="...", width=36, command=self._browse_audio_file).pack(
            side="left", padx=(4, 0)
        )

        self._exec_btn = ctk.CTkButton(
            self,
            text="実行",
            height=44,
            font=ctk.CTkFont(size=15),
            command=self._on_exec,
        )
        self._exec_btn.pack(fill="x", padx=20, pady=8)

        self._status_label = ctk.CTkLabel(
            self, text="待機中", font=ctk.CTkFont(size=14), anchor="w"
        )
        self._status_label.pack(fill="x", padx=20, pady=(4, 2))

        ctk.CTkLabel(self, text="ログ", anchor="w").pack(fill="x", padx=20, pady=(8, 2))
        self._log_box = ctk.CTkTextbox(self, height=160, state="disabled")
        self._log_box.pack(fill="both", expand=True, padx=20, pady=(0, 12))

    def _on_mode_change(self):
        mode = self._mode_var.get()
        for section in (self._device_section, self._rec_dest_section, self._file_section):
            section.pack_forget()

        if mode in (MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY):
            self._device_section.pack(fill="x", pady=(0, 8))
            self._rec_dest_section.pack(fill="x")
            self._exec_btn.configure(text="実行")
        elif mode == MODE_TRANSCRIBE_ONLY:
            self._file_section.pack(fill="x")
            self._exec_btn.configure(text="文字起こし開始")

    # ──────────────── 設定 ────────────────

    def _load_config(self):
        try:
            config = load_config(CONFIG_PATH)
        except InvalidConfigError as e:
            self._log(f"設定の読み込みに失敗しました: {e}")
            config = None
        if config:
            self._config = config
            if not os.environ.get("OPENAI_API_KEY") and config.openai_api_key:
                os.environ["OPENAI_API_KEY"] = config.openai_api_key
            self._log("設定を読み込みました")
        else:
            self._log("設定が見つかりません。設定から保存フォルダを設定してください")
        self._log(f"ログファイル: {self._log_file}")

    def _open_settings(self):
        dialog = SettingsDialog(self, self._config)
        self.wait_window(dialog)
        result = dialog.get_result()
        if result is None:
            return
        try:
            save_config(CONFIG_PATH, result)
            self._config = result
            if not os.environ.get("OPENAI_API_KEY") and result.openai_api_key:
                os.environ["OPENAI_API_KEY"] = result.openai_api_key
            self._log("設定を保存しました")
        except RuntimeError as e:
            messagebox.showerror("エラー", str(e))

    # ──────────────── デバイス ────────────────

    def _refresh_devices(self):
        try:
            devices = list_devices()
            names = [f"[{d['id']}] {strip_emoji(d['name'])}" for d in devices]
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
        if mode == MODE_RECORD_TRANSCRIBE and not self._config.save_folder:
            messagebox.showwarning(
                "設定が必要", "先に 設定から文字起こし保存フォルダを設定してください"
            )
            return

        device_id = self._selected_device_id()
        device_label = self._device_var.get()
        self._recorder = ThreadedRecorder(device_id)
        try:
            self._recorder.start()
        except Exception as e:
            messagebox.showerror("録音エラー", str(e))
            return

        self._recording = True
        self._elapsed = 0
        self._exec_btn.configure(text="録音停止", fg_color="red")
        self._log(f"録音開始 (デバイス: {device_label})")
        threading.Thread(target=self._timer_loop, daemon=True).start()

    def _stop_recording(self):
        self._recording = False
        self._exec_btn.configure(text="実行", fg_color=PRIMARY_BUTTON_COLOR)
        self._set_status("保存中...")
        self._log("録音停止 → ストリームを閉じています...")

        if self._recorder is None:
            return
        self._recorder.stop()
        self._log("ストリームを閉じました。データを結合中...")
        try:
            audio_data = self._recorder.get_data()
        except RuntimeError as e:
            self._log(f"エラー: {e}")
            self._reset_ui()
            return
        self._log(f"録音データ取得完了 ({len(audio_data) / 16000:.1f}秒)")

        rec_dest = Path(self._rec_dest_entry.get().strip() or str(Path.home() / "Desktop"))
        mode = self._mode_var.get()
        self._set_processing(True)
        self._exec_btn.configure(text="文字起こし中...")
        threading.Thread(
            target=self._process_audio, args=(audio_data, rec_dest, mode), daemon=True
        ).start()

    def _timer_loop(self):
        while self._recording:
            mins, secs = divmod(self._elapsed, 60)
            self._ui_queue.submit(self._set_status, f"[REC]  {mins:02d}:{secs:02d}  録音中...")
            time.sleep(1)
            self._elapsed += 1

    # ──────────────── 音声処理 ────────────────

    def _process_audio(self, audio_data, rec_dest: Path, mode: str):
        try:
            self._ui_queue.submit(self._log, f"WAVファイルを書き込み中... → {rec_dest}")
            try:
                audio_file = save_wav(audio_data, rec_dest)
                self._ui_queue.submit(self._log, f"音声ファイルを保存: {audio_file.name}")
            except Exception as e:
                self._ui_queue.submit(self._log, f"エラー: {e}")
                self._ui_queue.submit(self._reset_ui)
                return

            if mode == MODE_RECORD_ONLY:
                self._ui_queue.submit(self._log, f"録音完了 → {audio_file}")
                self._ui_queue.submit(self._reset_ui)
                return

            self._run_transcription(audio_file)
        except Exception:
            _logger.error("_process_audio で未捕捉の例外:\n%s", traceback.format_exc())
            self._ui_queue.submit(
                self._log, f"予期せぬエラーが発生しました（ログを確認: {self._log_file.name}）"
            )
            self._ui_queue.submit(self._reset_ui)

    def _run_transcribe_only(self):
        audio_path = self._file_entry.get().strip()
        if not audio_path:
            messagebox.showwarning("ファイル未選択", "音声ファイルを選択してください")
            return
        if not self._config.save_folder:
            messagebox.showwarning(
                "設定が必要", "先に 設定から文字起こし保存フォルダを設定してください"
            )
            return
        self._set_processing(True)
        self._exec_btn.configure(text="文字起こし中...")
        threading.Thread(
            target=self._run_transcription, args=(Path(audio_path),), daemon=True
        ).start()

    def _run_transcription(self, audio_file: Path):
        _logger.debug("_run_transcription 開始: %s", audio_file)
        start_time = time.time()

        self._ui_queue.submit(self._log, "文字起こし開始...")

        def on_progress(msg: str):
            elapsed = time.time() - start_time
            self._ui_queue.submit(self._set_status, f"{msg} ({elapsed:.1f}s)")
            self._ui_queue.submit(self._log, msg)

        try:
            saved_path = transcribe_and_save(
                audio_file, self._config, progress_callback=on_progress
            )
        except Exception as e:
            _logger.error("transcribe_and_save エラー:\n%s", traceback.format_exc())
            self._ui_queue.submit(self._log, f"文字起こしエラー: {e}")
            self._ui_queue.submit(self._reset_ui)
            return

        _logger.debug("保存完了: %s", saved_path)
        self._ui_queue.submit(self._log, f"文字起こし完了 → {saved_path}")
        self._ui_queue.submit(self._show_completion, saved_path)

    # ──────────────── ウィンドウ終了 ────────────────

    def _on_close(self):
        self._alive = False
        self._recording = False
        if self._recorder is not None:
            with contextlib.suppress(Exception):
                self._recorder.stop()
        self.destroy()

    # ──────────────── ユーティリティ ────────────────

    def _set_processing(self, busy: bool):
        state = "disabled" if busy else "normal"
        self._exec_btn.configure(state=state)
        self._settings_btn.configure(state=state)
        self._mode_menu.configure(state=state)
        self._device_menu.configure(state=state)
        self._rec_dest_entry.configure(state=state)

    def _reset_ui(self):
        self._status_label.configure(text="待機中", text_color=("gray14", "gray84"))
        self._exec_btn.configure(text="実行", fg_color=PRIMARY_BUTTON_COLOR)
        self._set_processing(False)

    def _show_completion(self, saved_path: Path):
        self._set_processing(False)
        self._exec_btn.configure(text="実行", fg_color=PRIMARY_BUTTON_COLOR)
        self._status_label.configure(text=f"完了: {saved_path.name}", text_color="green")
        self.bell()
        self.after(3000, self._reset_ui)

    def _set_status(self, text: str):
        self._status_label.configure(text=text)

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)
        _logger.info(msg)
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"[{timestamp}] {msg}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")
