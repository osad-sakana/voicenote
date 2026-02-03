"""
文字起こし機能モジュール
faster-whisperを使用したローカル文字起こし、またはOpenAI APIを使用したクラウド文字起こし
"""

import os
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

            # 文字起こし（言語は自動検出）
            task = progress.add_task("文字起こし中...", total=None)
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5
            )

            # セグメントを結合（2秒以上の間隔で改行）
            PAUSE_THRESHOLD = 2.0
            result_parts = []
            prev_end = 0.0

            for segment in segments:
                gap = segment.start - prev_end
                if result_parts and gap >= PAUSE_THRESHOLD:
                    result_parts.append("\n\n")
                elif result_parts:
                    result_parts.append(" ")
                result_parts.append(segment.text.strip())
                prev_end = segment.end

            transcription = "".join(result_parts)

            progress.update(task, completed=True)

        console.print("[green]✓ 文字起こし完了[/green]")
        return transcription.strip()

    except Exception as e:
        console.print(f"[red]文字起こしエラー: {e}[/red]")
        sys.exit(1)


def transcribe_audio_openai(audio_path: Path) -> str:
    """
    OpenAI Whisper APIで音声を文字起こしする

    Args:
        audio_path: 音声ファイルのパス

    Returns:
        文字起こしされたテキスト
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]エラー: OPENAI_API_KEY環境変数が設定されていません[/red]")
        sys.exit(1)

    # ファイルサイズチェック（25MB制限）
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 25:
        console.print(f"[red]エラー: ファイルサイズが25MBを超えています ({file_size_mb:.1f}MB)[/red]")
        console.print("[yellow]ヒント: 長い録音はローカルモード(faster-whisper)を使用してください[/yellow]")
        sys.exit(1)

    console.print(f"\n[cyan]OpenAI Whisper API で文字起こし中...[/cyan]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("APIリクエスト送信中...", total=None)

            client = OpenAI(api_key=api_key)

            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )

            # セグメントを結合（2秒以上の間隔で改行）
            PAUSE_THRESHOLD = 2.0
            result_parts = []
            prev_end = 0.0

            for segment in response.segments:
                gap = segment.start - prev_end
                if result_parts and gap >= PAUSE_THRESHOLD:
                    result_parts.append("\n\n")
                elif result_parts:
                    result_parts.append(" ")
                result_parts.append(segment.text.strip())
                prev_end = segment.end

            transcription = "".join(result_parts)

            progress.update(task, completed=True)

        console.print("[green]✓ 文字起こし完了[/green]")
        return transcription.strip()

    except Exception as e:
        console.print(f"[red]OpenAI APIエラー: {e}[/red]")
        sys.exit(1)
