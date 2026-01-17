#!/usr/bin/env python3
"""
録音・文字起こしツール
ローカルで音声を録音し、faster-whisperで文字起こしを行い、Obsidianに保存する。
"""

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from scipy.io import wavfile

from config import configure_interactive, load_config, save_config
from obsidian import save_to_obsidian
from recorder import SAMPLE_RATE, record_audio
from transcriber import transcribe_audio

console = Console()


def get_config_dir() -> Path:
    """設定ディレクトリを取得（なければ作成）"""
    config_dir = Path.home() / ".config" / "voicenote"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="録音・文字起こしツール"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="設定を再入力する"
    )
    args = parser.parse_args()

    # 設定ファイルパス
    config_dir = get_config_dir()
    config_path = config_dir / "config.json"

    # 設定の読み込みまたは作成
    config = None
    if not args.config:
        config = load_config(config_path)

    if config is None or args.config:
        config = configure_interactive()
        save_config(config_path, config)
    else:
        console.print("[cyan]設定を読み込みました。[/cyan]")

    # 設定の取得
    vault_path = Path(config["vault_path"])
    save_folder = config["save_folder"]
    whisper_model = config["whisper_model"]

    # 録音
    audio_data = record_audio()

    # 一時ファイルに保存（文字起こし用）
    temp_wav = config_dir / "temp_recording.wav"
    console.print(f"\n[cyan]音声データを一時保存中...[/cyan]")

    # float32からint16に変換
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(temp_wav, SAMPLE_RATE, audio_int16)

    # 文字起こし
    transcription = transcribe_audio(temp_wav, whisper_model)

    # 一時ファイル削除
    temp_wav.unlink()

    # Obsidianに保存
    console.print(f"\n[cyan]Obsidianに保存中...[/cyan]")
    saved_path = save_to_obsidian(vault_path, save_folder, transcription)

    # 完了メッセージ
    console.print(Panel.fit(
        f"[bold green]完了![/bold green]\n\n"
        f"[bold]保存先:[/bold]\n{saved_path.absolute()}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
