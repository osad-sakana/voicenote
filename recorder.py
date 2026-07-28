"""
録音機能モジュール
sounddeviceを使用したリアルタイム録音
"""

import logging
import signal
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

_logger = logging.getLogger("voicenote")

SAMPLE_RATE = 16000
STREAM_STOP_TIMEOUT_SEC = 5.0


def list_devices() -> list[dict]:
    """利用可能な入力デバイス一覧を返す"""
    devices = sd.query_devices()
    return [
        {"id": i, "name": d["name"], "input_channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def default_input_name() -> str:
    """デフォルト入力デバイス名を返す"""
    return sd.query_devices(kind="input")["name"]


def resolve_device_id(device: str | None) -> int | None:
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

    def __init__(self, device_id: int | None = None):
        self._device_id = device_id
        self._data: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._running = False
        self._closer_thread: threading.Thread | None = None

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

    def stop_with_timeout(self, timeout: float = STREAM_STOP_TIMEOUT_SEC) -> bool:
        """
        stop() をタイムアウト付きで実行する。
        PortAudio の stop()/close() が長時間ブロックする既知の問題（macOS）に対応するため、
        実際の停止処理は内部スレッドで行い、呼び出し元は最大 timeout 秒だけ待つ。
        タイムアウトした場合、ストリームは放棄され（例外送出中の可能性があるため触れない）、
        以降 get_data() で録音済みデータを取得することはできる。

        Returns:
            timeout 内に stop()/close() が完了すれば True、タイムアウトすれば False
        """
        self._running = False
        stream = self._stream
        self._stream = None
        if stream is None:
            return True

        def _close():
            try:
                try:
                    stream.stop()
                finally:
                    stream.close()
            except Exception:
                _logger.exception("ストリームの停止/クローズ中にエラーが発生しました")

        closer = threading.Thread(target=_close, daemon=True)
        self._closer_thread = closer
        closer.start()
        closer.join(timeout)
        return not closer.is_alive()

    def is_closing(self) -> bool:
        """stop_with_timeout() がタイムアウトして放棄したストリームの停止処理が、
        バックグラウンドでまだ完了していないかを返す。"""
        return self._closer_thread is not None and self._closer_thread.is_alive()

    def get_data(self) -> np.ndarray:
        with self._lock:
            if not self._data:
                raise RuntimeError("録音データがありません")
            return np.concatenate(self._data, axis=0).flatten()


def record_audio(
    device: str | None = None,
    on_start: Callable[[str], None] | None = None,
    on_stop: Callable[[], None] | None = None,
) -> np.ndarray:
    """
    音声を録音する（CLI用・Ctrl+Cで停止）

    Args:
        device: 入力デバイス名またはID（Noneの場合はデフォルト）
        on_start: 録音開始時に解決済みデバイス名を受け取るコールバック
        on_stop: Ctrl+C受信（録音停止処理開始）時に呼ばれるコールバック

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
        if on_stop:
            on_stop()
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    device_name = sd.query_devices(device_id)["name"] if device_id is not None else "デフォルト"
    if on_start:
        on_start(device_name)

    recorder.start()
    stop_event.wait()
    recorder.stop()

    return recorder.get_data()
