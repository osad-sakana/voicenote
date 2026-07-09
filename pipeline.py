"""
GUI/CLI 共通の業務ロジックモジュール。

`save_wav`、`load_or_configure`、`transcribe_and_save` を提供し、
エントリーポイント (`main.py` / `main_cli.py`) からは UI に集中できるようにする。
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from config import CONFIG_PATH, VoiceNoteConfig, configure_interactive, load_config, save_config
from formatter import format_transcription
from note_writer import save_transcript
from recorder import SAMPLE_RATE
from transcriber import transcribe


def load_or_configure(
    force_config: bool = False, interactive_fallback: bool = True
) -> VoiceNoteConfig:
    """設定ファイルを読み込み、必要なら対話的設定を実行する。

    Args:
        force_config: True なら既存設定を無視し対話的設定を実行する (CLI --config)。
        interactive_fallback: 設定が無いときに対話的設定にフォールバックするか。
            GUI 側は False を指定し、空設定を受け取って設定ダイアログで補完する。

    Returns:
        設定。GUI で interactive_fallback=False かつ設定無しなら `VoiceNoteConfig()`。

    Raises:
        InvalidConfigError: 設定ファイルが破損している場合。
        RuntimeError: 対話的設定後の保存に失敗した場合。
    """
    config = None if force_config else load_config(CONFIG_PATH)

    if config is None and interactive_fallback:
        config = configure_interactive()
        save_config(CONFIG_PATH, config)
    elif config is None:
        config = VoiceNoteConfig()

    return config


def save_wav(audio_data: np.ndarray, dest_dir: Path) -> Path:
    """録音データを `dest_dir/YYYY-MM-DD_HHMMSS_recording.wav` として保存する。

    float32 (-1.0〜1.0) を int16 に変換して書き出す。
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    audio_file = dest_dir / f"{timestamp}_recording.wav"
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(audio_file, SAMPLE_RATE, audio_int16)
    return audio_file


def transcribe_and_save(
    audio_file: Path,
    config: VoiceNoteConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """音声ファイルを文字起こし → 整形 → ノート保存し、保存先パスを返す。

    Args:
        audio_file: 文字起こし対象の音声ファイル。
        config: 設定。`save_folder` `transcription_mode` `whisper_model`
            `vad_filter` `format_mode` を参照する。
        progress_callback: 進捗メッセージを受け取るコールバック。
            GUI なら UI キュー経由、CLI なら Rich Progress 経由で消費する。

    Returns:
        保存された Markdown ファイルパス。

    Raises:
        RuntimeError: 文字起こし・整形・保存のいずれかが失敗した場合。
    """

    transcription = transcribe(audio_file, config, progress_callback=progress_callback)

    if config.format_mode != "none":
        transcription = format_transcription(
            transcription, config, progress_callback=progress_callback
        )

    save_folder = Path(config.save_folder)
    return save_transcript(save_folder, transcription, config.format_mode)
