"""
文字起こし機能モジュール
faster-whisperを使用したローカル文字起こし、またはOpenAI APIを使用したクラウド文字起こし
"""

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

PAUSE_THRESHOLD = 2.0
TARGET_SAMPLE_RATE = 16000


def _preprocess_audio(audio_path: Path) -> Path:
    """
    音声ファイルをWhisper最適形式（16kHz・モノラル）に変換する。
    変換が不要な場合は元のパスをそのまま返す。

    Args:
        audio_path: 入力音声ファイルパス

    Returns:
        変換後（または元）のファイルパス
    """
    from scipy.io import wavfile

    if audio_path.suffix.lower() != ".wav":
        return audio_path

    sample_rate, data = wavfile.read(str(audio_path))

    needs_resample = sample_rate != TARGET_SAMPLE_RATE
    needs_mono = data.ndim > 1

    if not needs_resample and not needs_mono:
        return audio_path

    # モノラル変換
    if needs_mono:
        data = data.mean(axis=1)

    # リサンプリング
    if needs_resample:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(TARGET_SAMPLE_RATE, sample_rate)
        data = resample_poly(data, TARGET_SAMPLE_RATE // g, sample_rate // g)

    # float → int16に変換（wavfile書き込みのため）
    if data.dtype != np.int16:
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val * 32767).astype(np.int16)
        else:
            data = data.astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    wavfile.write(str(tmp_path), TARGET_SAMPLE_RATE, data)
    return tmp_path


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
    vad_filter: bool = True,
) -> str:
    """
    faster-whisperで音声を文字起こしする

    Args:
        audio_path: 音声ファイルのパス
        model_name: 使用するWhisperモデル名
        progress_callback: 進捗メッセージを受け取るコールバック（GUIから渡す）
        vad_filter: 音声区間検出フィルタの有効/無効（無音・ノイズを除去してループを抑制）

    Returns:
        文字起こしされたテキスト

    Raises:
        RuntimeError: 文字起こし失敗時
    """
    def notify(msg: str):
        if progress_callback:
            progress_callback(msg)

    notify(f"モデル '{model_name}' をロード中...")

    preprocessed_path = None
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        notify("音声ファイルを最適化中...")

        preprocessed_path = _preprocess_audio(audio_path)
        notify("文字起こし中...")

        segments, _ = model.transcribe(
            str(preprocessed_path),
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
        )
        transcription = _merge_segments(segments)

        notify("文字起こし完了")
        return transcription

    except Exception as e:
        raise RuntimeError(f"文字起こしエラー: {e}") from e
    finally:
        if preprocessed_path and preprocessed_path != audio_path:
            preprocessed_path.unlink(missing_ok=True)


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
                audio_path,
                config["whisper_model"],
                progress_callback=on_progress,
                vad_filter=config.get("vad_filter", True),
            )

        progress.update(task, completed=True)

    console.print("[green]✓ 文字起こし完了[/green]")
    return result
