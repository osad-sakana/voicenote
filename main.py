#!/usr/bin/env python3
"""
録音・文字起こしツール
ローカルで音声を録音し、faster-whisperで文字起こしを行い、Obsidianに保存する。
"""

import argparse
import sys
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
    parser.add_argument(
        "--file",
        type=str,
        help="既存の音声ファイルを文字起こしする（録音をスキップ）"
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="録音のみ実行（文字起こしをスキップしてDesktopに保存）"
    )
    args = parser.parse_args()

    # オプションの排他チェック
    if args.file and args.record_only:
        console.print("[red]エラー: --fileと--record-onlyは同時に指定できません[/red]")
        sys.exit(1)

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

    # 録音のみモード: Desktopに保存して終了
    if args.record_only:
        from datetime import datetime

        # 録音
        audio_data = record_audio()

        # Desktopに保存
        desktop_path = Path.home() / "Desktop"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        audio_file = desktop_path / f"{timestamp}_recording.wav"

        console.print(f"\n[cyan]Desktopに保存中...[/cyan]")

        # float32からint16に変換
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(audio_file, SAMPLE_RATE, audio_int16)

        # 完了メッセージ
        console.print(Panel.fit(
            f"[bold green]録音完了![/bold green]\n\n"
            f"[bold]保存先:[/bold]\n{audio_file.absolute()}",
            border_style="green"
        ))
        return

    # ファイルモード: 既存ファイルを文字起こし
    if args.file:
        audio_file = Path(args.file)

        # ファイルの存在確認
        if not audio_file.exists():
            console.print(f"[red]エラー: ファイルが見つかりません: {audio_file}[/red]")
            sys.exit(1)

        if not audio_file.is_file():
            console.print(f"[red]エラー: 指定されたパスはファイルではありません: {audio_file}[/red]")
            sys.exit(1)

        console.print(f"[cyan]音声ファイル: {audio_file.name}[/cyan]")

        # 文字起こし（faster-whisperは様々な音声形式をサポート）
        transcription = transcribe_audio(audio_file, whisper_model)

    # 録音モード: 新規録音して文字起こし
    else:
        from datetime import datetime

        # 録音
        audio_data = record_audio()

        # Desktopに保存
        desktop_path = Path.home() / "Desktop"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        audio_file = desktop_path / f"{timestamp}_recording.wav"

        console.print(f"\n[cyan]Desktopに音声データを保存中...[/cyan]")

        # float32からint16に変換
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(audio_file, SAMPLE_RATE, audio_int16)

        console.print(f"[green]✓ 保存完了: {audio_file.name}[/green]")

        # 文字起こし
        transcription = transcribe_audio(audio_file, whisper_model)

    # Obsidianに保存
    console.print(f"\n[cyan]Obsidianに保存中...[/cyan]")
    saved_path = save_to_obsidian(vault_path, save_folder, transcription)

    # 完了メッセージ
    console.print(Panel.fit(
        f"[bold green]完了![/bold green]\n\n"
        f"[bold]音声ファイル:[/bold]\n{audio_file.absolute()}\n\n"
        f"[bold]文字起こし結果:[/bold]\n{saved_path.absolute()}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
