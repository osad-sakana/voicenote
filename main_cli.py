#!/usr/bin/env python3
"""
録音・文字起こしツール（CLIエントリーポイント）
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from rich.console import Console
from rich.panel import Panel
from scipy.io import wavfile

from config import configure_interactive, load_config, save_config
from formatter import format_transcription
from obsidian import save_to_obsidian
from recorder import SAMPLE_RATE, print_devices, record_audio, resolve_device_id
from transcriber import transcribe_with_cli_progress

console = Console()

CONFIG_PATH = Path.home() / ".config" / "voicenote" / "config.json"


def _save_wav(audio_data: np.ndarray, dest_dir: Path) -> Path:
    from datetime import datetime

    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    audio_file = dest_dir / f"{timestamp}_recording.wav"
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(audio_file, SAMPLE_RATE, audio_int16)
    return audio_file


def _load_or_configure(force_config: bool) -> dict:
    config = None if force_config else load_config(CONFIG_PATH)
    if config is None:
        config = configure_interactive()
        try:
            save_config(CONFIG_PATH, config)
            console.print(f"[green]設定を保存しました: {CONFIG_PATH}[/green]")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    else:
        console.print("[cyan]設定を読み込みました。[/cyan]")

    # 設定からAPIキーを環境変数にセット（環境変数が未設定の場合のみ）
    if not os.environ.get("OPENAI_API_KEY") and config.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

    return config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="録音・文字起こしツール")
    parser.add_argument("--config", action="store_true", help="設定を再入力する")
    parser.add_argument("--file", type=str, help="既存の音声ファイルを文字起こしする")
    parser.add_argument("--record-only", action="store_true", help="録音のみ（文字起こしをスキップ）")
    parser.add_argument("--list-devices", action="store_true", help="利用可能なオーディオデバイス一覧を表示")
    parser.add_argument("--device", type=str, help="録音に使用するデバイス（名前またはID）")
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    if args.file and args.record_only:
        console.print("[red]エラー: --fileと--record-onlyは同時に指定できません[/red]")
        sys.exit(1)

    config = _load_or_configure(args.config)
    save_folder = Path(config["save_folder"])
    desktop = Path.home() / "Desktop"

    # ファイルモード
    if args.file:
        audio_file = Path(args.file)
        if not audio_file.is_file():
            console.print(f"[red]エラー: ファイルが見つかりません: {audio_file}[/red]")
            sys.exit(1)
        console.print(f"[cyan]音声ファイル: {audio_file.name}[/cyan]")
        try:
            transcription = transcribe_with_cli_progress(audio_file, config)
            transcription = format_transcription(transcription, config)
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        try:
            saved_path = save_to_obsidian(save_folder, transcription, config.get("format_mode", "none"))
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        console.print(Panel.fit(
            f"[bold green]完了![/bold green]\n\n"
            f"[bold]文字起こし結果:[/bold]\n{saved_path.absolute()}",
            border_style="green",
        ))
        return

    # 録音
    try:
        audio_data = record_audio(device=args.device)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]エラー: {e}[/red]")
        sys.exit(1)

    console.print("\n[cyan]Desktopに音声データを保存中...[/cyan]")
    audio_file = _save_wav(audio_data, desktop)
    console.print(f"[green]✓ 保存完了: {audio_file.name}[/green]")

    # 録音のみモード
    if args.record_only:
        console.print(Panel.fit(
            f"[bold green]録音完了![/bold green]\n\n"
            f"[bold]保存先:[/bold]\n{audio_file.absolute()}",
            border_style="green",
        ))
        return

    # 文字起こし
    try:
        transcription = transcribe_with_cli_progress(audio_file, config)
        transcription = format_transcription(transcription, config)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    try:
        saved_path = save_to_obsidian(save_folder, transcription, config.get("format_mode", "none"))
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold green]完了![/bold green]\n\n"
        f"[bold]音声ファイル:[/bold]\n{audio_file.absolute()}\n\n"
        f"[bold]文字起こし結果:[/bold]\n{saved_path.absolute()}",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
