"""
録音機能モジュール
sounddeviceを使用したリアルタイム録音
"""

import signal
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SAMPLE_RATE = 16000


def list_devices() -> list[dict]:
    """利用可能な入力デバイス一覧を返す"""
    devices = sd.query_devices()
    return [
        {"id": i, "name": d["name"], "input_channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def print_devices():
    """利用可能なオーディオデバイス一覧を表示（CLI用）"""
    table = Table(title="利用可能なオーディオデバイス")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("デバイス名", style="green")
    table.add_column("入力Ch", justify="right")

    for d in list_devices():
        table.add_row(str(d["id"]), d["name"], str(d["input_channels"]))

    console.print(table)
    default_input = sd.query_devices(kind="input")
    console.print(f"\n[dim]デフォルト入力: {default_input['name']}[/dim]")


def resolve_device_id(device: Optional[str]) -> Optional[int]:
    """デバイス名またはIDを数値IDに解決する。見つからない場合はValueErrorを送出。"""
    if device is None:
        return None
    if device.isdigit():
        return int(device)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if device.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    raise ValueError(f"デバイス '{device}' が見つかりません")


class ThreadedRecorder:
    """
    GUI用スレッドセーフ録音クラス。
    start() で録音開始、stop() で停止、get_data() でnumpy配列を取得。
    """

    def __init__(self, device_id: Optional[int] = None):
        self._device_id = device_id
        self._data: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _callback(self, indata, frames, time, status):
        if self._running:
            with self._lock:
                self._data.append(indata.copy())

    def start(self):
        self._data = []
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device_id,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_data(self) -> np.ndarray:
        with self._lock:
            if not self._data:
                raise RuntimeError("録音データがありません")
            return np.concatenate(self._data, axis=0).flatten()


def record_audio(device: Optional[str] = None) -> np.ndarray:
    """
    音声を録音する（CLI用・Ctrl+Cで停止）

    Args:
        device: 入力デバイス名またはID（Noneの場合はデフォルト）

    Returns:
        録音された音声データ（float32のnumpy配列）

    Raises:
        ValueError: デバイスが見つからない場合
        RuntimeError: 録音データが空の場合
    """
    device_id = resolve_device_id(device)

    recorder = ThreadedRecorder(device_id)
    stop_event = threading.Event()

    def _signal_handler(sig, frame):
        console.print("\n[yellow]録音を停止しています...[/yellow]")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    device_name = sd.query_devices(device_id)["name"] if device_id is not None else "デフォルト"
    console.print(Panel.fit(
        f"[bold green]録音を開始します[/bold green]\n"
        f"[dim]デバイス: {device_name}[/dim]\n"
        f"[yellow]Ctrl+C[/yellow] で録音を終了します",
        border_style="green",
    ))

    recorder.start()
    stop_event.wait()
    recorder.stop()

    console.print("[green]✓ 録音完了[/green]")
    return recorder.get_data()
