"""
GUI/CLI 共通の業務ロジックモジュール。

`load_or_configure`、`transcribe_and_save` を提供し、
エントリーポイント (`main.py` / `main_cli.py`) からは UI に集中できるようにする。
"""

from collections.abc import Callable
from pathlib import Path

from config import CONFIG_PATH, VoiceNoteConfig, configure_interactive, load_config, save_config
from formatter import format_transcription
from note_writer import save_transcript
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
