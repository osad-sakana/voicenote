"""VoiceNote GUI のメインウィンドウ。"""

import logging
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from config import CONFIG_PATH, InvalidConfigError, VoiceNoteConfig, load_config, save_config
from recorder import list_devices

from .constants import (
    MODE_RECORD_ONLY,
    MODE_RECORD_TRANSCRIBE,
    MODE_TRANSCRIBE_ONLY,
    PRIMARY_BUTTON_COLOR,
)
from .devices import build_device_labels, parse_device_id
from .layout import build_ui
from .settings_dialog import SettingsDialog
from .ui_queue import ThreadSafeUIQueue
from .workflow import RecordingWorkflow, WorkflowCallbacks, validate_start, validate_transcribe_only

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
        self._alive = True
        self._ui_queue = ThreadSafeUIQueue(self, alive_fn=lambda: self._alive)
        self._workflow = RecordingWorkflow(
            self._config,
            WorkflowCallbacks(
                on_status=lambda msg: self._ui_queue.submit(self._set_status, msg),
                on_log=lambda msg: self._ui_queue.submit(self._log, msg),
                on_recording_started=lambda: self._ui_queue.submit(self._on_recording_started),
                on_processing_started=lambda: self._ui_queue.submit(self._on_processing_started),
                on_done=lambda path: self._ui_queue.submit(self._show_completion, path),
                on_record_only_done=lambda _path: self._ui_queue.submit(self._reset_ui),
                on_error=lambda msg: self._ui_queue.submit(self._handle_error, msg),
            ),
            log_file=log_file,
        )

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._load_config()
        self._refresh_devices()
        self._on_mode_change()
        self._ui_queue.start()

    # ──────────────── UI構築 ────────────────

    def _build_ui(self):
        build_ui(self)

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
            self._workflow.update_config(config)
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
            self._workflow.update_config(result)
            self._log("設定を保存しました")
        except RuntimeError as e:
            messagebox.showerror("エラー", str(e))

    # ──────────────── デバイス ────────────────

    def _refresh_devices(self):
        try:
            names = build_device_labels(list_devices())
            self._device_menu.configure(values=names)
            self._device_var.set(names[0])
        except Exception as e:
            self._log(f"デバイス取得エラー: {e}")

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
            if self._workflow.is_recording:
                self._stop_recording()
            else:
                self._start_recording()
        elif mode == MODE_TRANSCRIBE_ONLY:
            self._run_transcribe_only()

    def _start_recording(self):
        mode = self._mode_var.get()
        error = validate_start(mode, self._config.save_folder)
        if error:
            title, message = error
            messagebox.showwarning(title, message)
            return

        device_label = self._device_var.get()
        device_id = parse_device_id(device_label)
        error_message = self._workflow.start(device_id, device_label)
        if error_message:
            messagebox.showerror("録音エラー", error_message)

    def _stop_recording(self):
        self._exec_btn.configure(text="実行", fg_color=PRIMARY_BUTTON_COLOR)
        rec_dest = Path(self._rec_dest_entry.get().strip() or str(Path.home() / "Desktop"))
        mode = self._mode_var.get()
        self._workflow.stop_and_process(rec_dest, mode)

    def _run_transcribe_only(self):
        audio_path = self._file_entry.get().strip()
        error = validate_transcribe_only(audio_path, self._config.save_folder)
        if error:
            title, message = error
            messagebox.showwarning(title, message)
            return
        self._workflow.run_transcribe_only(Path(audio_path))

    # ──────────────── workflow コールバック (メインスレッドで実行) ────────────────

    def _on_recording_started(self):
        self._exec_btn.configure(text="録音停止", fg_color="red")

    def _on_processing_started(self):
        self._set_processing(True)
        self._exec_btn.configure(text="文字起こし中...")

    def _handle_error(self, msg: str):
        self._log(msg)
        self._reset_ui()

    # ──────────────── ウィンドウ終了 ────────────────

    def _on_close(self):
        self._alive = False
        self._workflow.shutdown()
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
