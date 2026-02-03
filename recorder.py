"""
録音機能モジュール
sounddeviceを使用したリアルタイム録音
"""

import signal
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# グローバル変数（録音データ格納用）
_recording_data = []
_is_recording = False
SAMPLE_RATE = 16000


def list_devices():
    """利用可能なオーディオデバイス一覧を表示"""
    devices = sd.query_devices()

    table = Table(title="利用可能なオーディオデバイス")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("デバイス名", style="green")
    table.add_column("入力Ch", justify="right")
    table.add_column("出力Ch", justify="right")

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # 入力デバイスのみ表示
            table.add_row(
                str(i),
                device['name'],
                str(device['max_input_channels']),
                str(device['max_output_channels'])
            )

    console.print(table)

    # デフォルトデバイス情報
    default_input = sd.query_devices(kind='input')
    console.print(f"\n[dim]デフォルト入力: {default_input['name']}[/dim]")


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


def record_audio(device: Optional[str] = None) -> np.ndarray:
    """
    音声を録音する

    Args:
        device: 入力デバイス名またはID（Noneの場合はデフォルト）

    Returns:
        録音された音声データ（float32のnumpy配列）
    """
    global _is_recording, _recording_data

    _recording_data = []
    _is_recording = True

    # デバイスの解決
    device_id = None
    if device is not None:
        if device.isdigit():
            device_id = int(device)
        else:
            # デバイス名で検索
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if device.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                    device_id = i
                    break
            if device_id is None:
                console.print(f"[red]エラー: デバイス '{device}' が見つかりません[/red]")
                console.print("[yellow]ヒント: --list-devices で利用可能なデバイスを確認してください[/yellow]")
                sys.exit(1)

    # Ctrl+Cのシグナルハンドラーを設定
    signal.signal(signal.SIGINT, _signal_handler)

    device_name = sd.query_devices(device_id)['name'] if device_id is not None else "デフォルト"
    console.print(Panel.fit(
        f"[bold green]録音を開始します[/bold green]\n"
        f"[dim]デバイス: {device_name}[/dim]\n"
        f"[yellow]Ctrl+C[/yellow] で録音を終了します",
        border_style="green"
    ))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            device=device_id,
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
