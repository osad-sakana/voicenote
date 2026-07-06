#!/usr/bin/env python3
"""
録音・文字起こしツール（CLIエントリーポイント）
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import VoiceNoteConfig
from logging_setup import setup_logging
from pipeline import load_or_configure, save_wav, transcribe_and_save
from recorder import print_devices, record_audio

console = Console()


def _run_transcription(audio_file: Path, config: VoiceNoteConfig) -> Path:
    """Rich Progress を駆動しつつ pipeline.transcribe_and_save を実行する。"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("準備中...", total=None)

        def on_progress(msg: str):
            progress.update(task, description=msg)

        saved_path = transcribe_and_save(audio_file, config, progress_callback=on_progress)
        progress.update(task, completed=True)

    console.print("[green]✓ 文字起こし完了[/green]")
    return saved_path


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="録音・文字起こしツール")
    parser.add_argument("--config", action="store_true", help="設定を再入力する")
    parser.add_argument("--file", type=str, help="既存の音声ファイルを文字起こしする")
    parser.add_argument(
        "--record-only", action="store_true", help="録音のみ（文字起こしをスキップ）"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="利用可能なオーディオデバイス一覧を表示"
    )
    parser.add_argument("--device", type=str, help="録音に使用するデバイス（名前またはID）")
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    if args.file and args.record_only:
        console.print("[red]エラー: --fileと--record-onlyは同時に指定できません[/red]")
        sys.exit(1)

    setup_logging()
    config = load_or_configure(force_config=args.config)
    desktop = Path.home() / "Desktop"

    if args.file:
        audio_file = Path(args.file)
        if not audio_file.is_file():
            console.print(f"[red]エラー: ファイルが見つかりません: {audio_file}[/red]")
            sys.exit(1)
        console.print(f"[cyan]音声ファイル: {audio_file.name}[/cyan]")
        try:
            saved_path = _run_transcription(audio_file, config)
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        console.print(
            Panel.fit(
                f"[bold green]完了![/bold green]\n\n"
                f"[bold]文字起こし結果:[/bold]\n{saved_path.absolute()}",
                border_style="green",
            )
        )
        return

    try:
        audio_data = record_audio(device=args.device)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]エラー: {e}[/red]")
        sys.exit(1)

    console.print("\n[cyan]Desktopに音声データを保存中...[/cyan]")
    audio_file = save_wav(audio_data, desktop)
    console.print(f"[green]✓ 保存完了: {audio_file.name}[/green]")

    if args.record_only:
        console.print(
            Panel.fit(
                f"[bold green]録音完了![/bold green]\n\n"
                f"[bold]保存先:[/bold]\n{audio_file.absolute()}",
                border_style="green",
            )
        )
        return

    try:
        saved_path = _run_transcription(audio_file, config)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold green]完了![/bold green]\n\n"
            f"[bold]音声ファイル:[/bold]\n{audio_file.absolute()}\n\n"
            f"[bold]文字起こし結果:[/bold]\n{saved_path.absolute()}",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
