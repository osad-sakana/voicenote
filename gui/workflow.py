"""録音・文字起こしのライフサイクルとスレッド管理 (Tkinter 非依存)。

`App` はここで定義するコールバックを `ThreadSafeUIQueue.submit` でラップして
渡すこと。ワーカースレッドから直接 Tk ウィジェットを触ると macOS では
SIGBUS で落ちるため、その安全性はこのモジュールではなく `App` 側が担保する。
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import VoiceNoteConfig
from pipeline import save_wav, transcribe_and_save
from recorder import ThreadedRecorder

from .constants import MODE_RECORD_ONLY, MODE_RECORD_TRANSCRIBE

_logger = logging.getLogger("voicenote")


def validate_start(mode: str, save_folder: str) -> tuple[str, str] | None:
    """録音開始前のバリデーション。問題なければ None、あれば (title, message)。"""
    if mode == MODE_RECORD_TRANSCRIBE and not save_folder:
        return ("設定が必要", "先に 設定から文字起こし保存フォルダを設定してください")
    return None


def validate_transcribe_only(audio_path: str, save_folder: str) -> tuple[str, str] | None:
    """文字起こしのみモードのバリデーション。問題なければ None、あれば (title, message)。"""
    if not audio_path:
        return ("ファイル未選択", "音声ファイルを選択してください")
    if not save_folder:
        return ("設定が必要", "先に 設定から文字起こし保存フォルダを設定してください")
    return None


@dataclass(frozen=True)
class WorkflowCallbacks:
    """workflow から呼ばれる UI 更新コールバック群。"""

    on_status: Callable[[str], None]
    on_log: Callable[[str], None]
    on_recording_started: Callable[[], None]
    on_processing_started: Callable[[], None]
    on_done: Callable[[Path], None]
    on_record_only_done: Callable[[Path], None]
    on_error: Callable[[str], None]


class RecordingWorkflow:
    """録音開始→停止→WAV保存→文字起こしの状態遷移とスレッド管理を担う。"""

    def __init__(
        self,
        config: VoiceNoteConfig,
        callbacks: WorkflowCallbacks,
        recorder_factory: Callable[[int | None], ThreadedRecorder] = ThreadedRecorder,
        log_file: Path | None = None,
    ):
        self._config = config
        self._callbacks = callbacks
        self._recorder_factory = recorder_factory
        self._log_file = log_file
        self._recorder: ThreadedRecorder | None = None
        self._recording = False
        self._elapsed = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def update_config(self, config: VoiceNoteConfig) -> None:
        self._config = config

    def start(self, device_id: int | None, device_label: str) -> str | None:
        """録音を開始する。失敗時はエラーメッセージを返す。"""
        recorder = self._recorder_factory(device_id)
        try:
            recorder.start()
        except Exception as e:
            return str(e)

        self._recorder = recorder
        self._recording = True
        self._elapsed = 0
        self._callbacks.on_recording_started()
        self._callbacks.on_log(f"録音開始 (デバイス: {device_label})")
        threading.Thread(target=self._timer_loop, daemon=True).start()
        return None

    def stop_and_process(self, rec_dest: Path, mode: str) -> None:
        """録音を停止し、WAV保存 → (必要なら) 文字起こしをバックグラウンドスレッドで実行する。"""
        self._recording = False
        self._callbacks.on_status("保存中...")
        self._callbacks.on_log("録音停止 → ストリームを閉じています...")

        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return
        recorder.stop()
        self._callbacks.on_log("ストリームを閉じました。データを結合中...")
        try:
            audio_data = recorder.get_data()
        except RuntimeError as e:
            self._callbacks.on_error(f"エラー: {e}")
            return
        self._callbacks.on_log(f"録音データ取得完了 ({len(audio_data) / 16000:.1f}秒)")

        self._callbacks.on_processing_started()
        threading.Thread(
            target=self._process_audio, args=(audio_data, rec_dest, mode), daemon=True
        ).start()

    def run_transcribe_only(self, audio_file: Path) -> None:
        """文字起こしのみモードをバックグラウンドスレッドで実行する。"""
        self._callbacks.on_processing_started()
        threading.Thread(target=self._run_transcription, args=(audio_file,), daemon=True).start()

    def shutdown(self) -> None:
        """アプリ終了時に録音中であれば安全に停止する。"""
        self._recording = False
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.stop()

    # ──────────────── 内部: バックグラウンドスレッドで実行される処理 ────────────────

    def _timer_loop(self):
        while self._recording:
            mins, secs = divmod(self._elapsed, 60)
            self._callbacks.on_status(f"[REC]  {mins:02d}:{secs:02d}  録音中...")
            time.sleep(1)
            self._elapsed += 1

    def _process_audio(self, audio_data: np.ndarray, rec_dest: Path, mode: str):
        try:
            self._callbacks.on_log(f"WAVファイルを書き込み中... → {rec_dest}")
            try:
                audio_file = save_wav(audio_data, rec_dest)
                self._callbacks.on_log(f"音声ファイルを保存: {audio_file.name}")
            except Exception as e:
                self._callbacks.on_error(f"エラー: {e}")
                return

            if mode == MODE_RECORD_ONLY:
                self._callbacks.on_log(f"録音完了 → {audio_file}")
                self._callbacks.on_record_only_done(audio_file)
                return

            self._run_transcription(audio_file)
        except Exception:
            _logger.error("_process_audio で未捕捉の例外:\n%s", traceback.format_exc())
            log_name = self._log_file.name if self._log_file else "ログファイル"
            self._callbacks.on_error(f"予期せぬエラーが発生しました（ログを確認: {log_name}）")

    def _run_transcription(self, audio_file: Path):
        _logger.debug("_run_transcription 開始: %s", audio_file)
        start_time = time.time()
        self._callbacks.on_log("文字起こし開始...")

        def on_progress(msg: str):
            elapsed = time.time() - start_time
            self._callbacks.on_status(f"{msg} ({elapsed:.1f}s)")
            self._callbacks.on_log(msg)

        try:
            saved_path = transcribe_and_save(
                audio_file, self._config, progress_callback=on_progress
            )
        except Exception as e:
            _logger.error("transcribe_and_save エラー:\n%s", traceback.format_exc())
            self._callbacks.on_error(f"文字起こしエラー: {e}")
            return

        _logger.debug("保存完了: %s", saved_path)
        self._callbacks.on_log(f"文字起こし完了 → {saved_path}")
        self._callbacks.on_done(saved_path)
