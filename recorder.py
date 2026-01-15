"""
録音機能モジュール
sounddeviceを使用したリアルタイム録音
"""

import signal
import sys

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel

console = Console()

# グローバル変数（録音データ格納用）
_recording_data = []
_is_recording = False
SAMPLE_RATE = 16000


def _signal_handler(sig, frame):
    """Ctrl+Cでの録音停止ハンドラー"""
    global _is_recording
    _is_recording = False
    console.print("\n[yellow]録音を停止しています...[/yellow]")


def _audio_callback(indata, frames, time, status):
    """録音コールバック（sounddeviceで使用）"""
    if status:
        console.print(f"[yellow]Warning: {status}[/yellow]")
    if _is_recording:
        _recording_data.append(indata.copy())


def record_audio() -> np.ndarray:
    """
    音声を録音する

    Returns:
        録音された音声データ（float32のnumpy配列）
    """
    global _is_recording, _recording_data

    _recording_data = []
    _is_recording = True

    # Ctrl+Cのシグナルハンドラーを設定
    signal.signal(signal.SIGINT, _signal_handler)

    console.print(Panel.fit(
        "[bold green]録音を開始します[/bold green]\n[yellow]Ctrl+C[/yellow] で録音を終了します",
        border_style="green"
    ))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=_audio_callback
        ):
            while _is_recording:
                sd.sleep(100)
    except Exception as e:
        console.print(f"[red]録音エラー: {e}[/red]")
        sys.exit(1)

    console.print("[green]✓ 録音完了[/green]")

    # 録音データを結合
    if len(_recording_data) == 0:
        console.print("[red]録音データがありません。[/red]")
        sys.exit(1)

    audio_data = np.concatenate(_recording_data, axis=0)
    return audio_data.flatten()
