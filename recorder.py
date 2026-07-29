"""
録音機能モジュール
sounddeviceを使用したリアルタイム録音
"""

import contextlib
import logging
import signal
import threading
from collections.abc import Callable
from pathlib import Path

import sounddevice as sd

from wav_writer import StreamingWavWriter

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
    start() で録音開始、stop() で停止、finalize_recording() でWAVファイルを確定する。

    録音データはメモリに溜め込まず、コールバックで受け取った音声チャンクを
    `StreamingWavWriter` 経由で逐次ディスクへ書き込む。これにより stop() が
    ハングしても、それまでの録音データは既にディスク上の有効な WAV として残る。
    """

    def __init__(
        self,
        device_id: int | None = None,
        dest_dir: Path | None = None,
        writer_factory: Callable[[Path, int], StreamingWavWriter] = StreamingWavWriter,
    ):
        self._device_id = device_id
        self._dest_dir = dest_dir
        self._writer_factory = writer_factory
        self._writer: StreamingWavWriter | None = None
        self._stream: sd.InputStream | None = None
        self._running = False
        self._closer_thread: threading.Thread | None = None

    def _callback(self, indata, frames, time, status):
        writer = self._writer
        if self._running and writer is not None:
            try:
                writer.write(indata)
            except Exception:
                _logger.exception("録音コールバックでの書き込みに失敗しました")

    def start(self):
        dest_dir = self._dest_dir if self._dest_dir is not None else Path.home() / "Desktop"
        writer = self._writer_factory(dest_dir, SAMPLE_RATE)
        # コールバックが _writer/_running を見た瞬間から書き込めるよう、
        # stream.start() より前に代入しておく（取りこぼし窓をなくす）。
        self._writer = writer
        self._running = True
        stream = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self._device_id,
                callback=self._callback,
            )
            stream.start()
        except Exception:
            self._writer = None
            self._running = False
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
            writer.abort()
            raise
        self._stream = stream

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
        録音済みデータは finalize_recording() で既に確定済みのため影響を受けない。

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

    def finalize_recording(self, timeout: float = 2.0) -> Path | None:
        """録音を確定し、WAVファイルのパスを返す。PortAudio の停止は待たない。

        Returns:
            1フレーム以上録音されていれば保存先パス、そうでなければ None。
        """
        self._running = False
        writer = self._writer
        self._writer = None
        if writer is None:
            return None
        return writer.finalize(timeout)


def record_audio(
    device: str | None = None,
    dest_dir: Path | None = None,
    on_start: Callable[[str], None] | None = None,
    on_stop: Callable[[], None] | None = None,
    on_saved: Callable[[Path], None] | None = None,
) -> Path:
    """
    音声を録音する（CLI用・Ctrl+Cで停止）

    Args:
        device: 入力デバイス名またはID（Noneの場合はデフォルト）
        dest_dir: 録音WAVファイルの保存先ディレクトリ
        on_start: 録音開始時に解決済みデバイス名を受け取るコールバック
        on_stop: Ctrl+C受信（録音停止処理開始）時に呼ばれるコールバック
        on_saved: 録音データがWAVファイルとして確定した直後に呼ばれるコールバック。
            PortAudio の停止処理がハングしても、この時点で既にファイルは
            ディスク上に存在する。

    Returns:
        保存されたWAVファイルのパス

    Raises:
        ValueError: デバイスが見つからない場合
        RuntimeError: 録音データが空の場合
    """
    device_id = resolve_device_id(device)

    recorder = ThreadedRecorder(device_id, dest_dir)
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
    # finalize_recording() を stop_with_timeout() より先に呼ぶ: PortAudio の
    # 停止処理がハングしても (#19)、その手前でファイルを確定させておく。
    audio_file = recorder.finalize_recording()
    if audio_file is None:
        recorder.stop_with_timeout()
        raise RuntimeError("録音データがありません")
    if on_saved:
        on_saved(audio_file)
    recorder.stop_with_timeout()

    return audio_file
