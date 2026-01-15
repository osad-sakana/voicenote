"""
文字起こし機能モジュール
faster-whisperを使用したローカル文字起こし
"""

import sys
from pathlib import Path

from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    """
    faster-whisperで音声を文字起こしする

    Args:
        audio_path: 音声ファイルのパス
        model_name: 使用するWhisperモデル名

    Returns:
        文字起こしされたテキスト
    """
    console.print(f"\n[cyan]Whisperモデル '{model_name}' をロード中...[/cyan]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # モデルのロード
            task = progress.add_task("モデルをロード中...", total=None)
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            progress.update(task, completed=True)

            # 文字起こし
            task = progress.add_task("文字起こし中...", total=None)
            segments, info = model.transcribe(
                str(audio_path),
                language="ja",
                beam_size=5
            )

            # セグメントを結合
            transcription = ""
            for segment in segments:
                transcription += segment.text

            progress.update(task, completed=True)

        console.print("[green]✓ 文字起こし完了[/green]")
        return transcription.strip()

    except Exception as e:
        console.print(f"[red]文字起こしエラー: {e}[/red]")
        sys.exit(1)
