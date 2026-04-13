"""
文字起こし機能モジュール
faster-whisperを使用したローカル文字起こし、またはOpenAI APIを使用したクラウド文字起こし
"""

import os
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

PAUSE_THRESHOLD = 2.0


def _merge_segments(segments) -> str:
    """セグメントを結合する（2秒以上の間隔で段落分け）"""
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
    return "".join(result_parts).strip()


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    faster-whisperで音声を文字起こしする

    Args:
        audio_path: 音声ファイルのパス
        model_name: 使用するWhisperモデル名
        progress_callback: 進捗メッセージを受け取るコールバック（GUIから渡す）

    Returns:
        文字起こしされたテキスト

    Raises:
        RuntimeError: 文字起こし失敗時
    """
    def notify(msg: str):
        if progress_callback:
            progress_callback(msg)

    notify(f"モデル '{model_name}' をロード中...")

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        notify("文字起こし中...")

        segments, _ = model.transcribe(str(audio_path), beam_size=5)
        transcription = _merge_segments(segments)

        notify("文字起こし完了")
        return transcription

    except Exception as e:
        raise RuntimeError(f"文字起こしエラー: {e}") from e


def transcribe_audio_openai(
    audio_path: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    OpenAI Whisper APIで音声を文字起こしする

    Args:
        audio_path: 音声ファイルのパス
        progress_callback: 進捗メッセージを受け取るコールバック（GUIから渡す）

    Returns:
        文字起こしされたテキスト

    Raises:
        ValueError: APIキー未設定、ファイルサイズ超過
        RuntimeError: API呼び出し失敗時
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY環境変数が設定されていません")

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 25:
        raise ValueError(
            f"ファイルサイズが25MBを超えています ({file_size_mb:.1f}MB)。"
            "長い録音はローカルモードを使用してください。"
        )

    def notify(msg: str):
        if progress_callback:
            progress_callback(msg)

    notify("OpenAI APIリクエスト送信中...")

    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
            )

        transcription = _merge_segments(response.segments)
        notify("文字起こし完了")
        return transcription

    except Exception as e:
        raise RuntimeError(f"OpenAI APIエラー: {e}") from e


def transcribe_with_cli_progress(audio_path: Path, config: dict) -> str:
    """CLI用: richプログレス表示つきで文字起こしを実行する"""
    mode = config.get("transcription_mode", "local")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("準備中...", total=None)

        def on_progress(msg: str):
            progress.update(task, description=msg)

        if mode == "openai":
            result = transcribe_audio_openai(audio_path, progress_callback=on_progress)
        else:
            result = transcribe_audio(
                audio_path, config["whisper_model"], progress_callback=on_progress
            )

        progress.update(task, completed=True)

    console.print("[green]✓ 文字起こし完了[/green]")
    return result
