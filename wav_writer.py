"""録音チャンクをストリーミングでWAVファイルへ逐次書き込むモジュール。

`wave.Wave_write.writeframes()` は呼び出しのたびに RIFF/data チャンクの
サイズフィールドを書き戻す（`_patchheader()`）。そのため、書き込み中の
任意の時点でファイルは常に有効な WAV として読み取れ、停止処理がハングしたり
プロセスが異常終了したりしても、それまでに書き込まれた分は失われない。

ディスクI/O（特に `rec_dest` がネットワーク/クラウド同期フォルダの場合）を
PortAudio のコールバックスレッド内で直接行うとストリームを詰まらせうるため、
実際の書き込みは専用スレッドに逃がし、コールバック側はキューに積むだけにする。
"""

import contextlib
import logging
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

_logger = logging.getLogger("voicenote")

SAMPLE_WIDTH = 2  # int16
# 書き込みスレッドがディスクI/Oで詰まった場合でもキューが録音長に比例して
# 増え続けない上限。16kHz mono では 1 チャンクは通常数十ms分なので、
# 数百個の余裕があれば実用上のI/O遅延は十分吸収できる。
_MAX_QUEUE_SIZE = 500
_SENTINEL = object()


def float32_to_int16(data: np.ndarray) -> np.ndarray:
    """float32 (-1.0〜1.0) の音声データを int16 に変換する。"""
    return (data * 32767).astype(np.int16)


def _resolve_unique_path(dest_dir: Path, timestamp: str) -> Path:
    candidate = dest_dir / f"{timestamp}_recording.wav"
    suffix = 2
    while candidate.exists():
        candidate = dest_dir / f"{timestamp}_recording_{suffix}.wav"
        suffix += 1
    return candidate


class StreamingWavWriter:
    """録音開始時にWAVファイルをオープンし、チャンクを逐次書き込む。"""

    def __init__(self, dest_dir: Path, sample_rate: int):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.path = _resolve_unique_path(dest_dir, timestamp)

        self._wav = wave.open(str(self.path), "wb")  # noqa: SIM115 (kept open across writes)
        self._wav.setnchannels(1)
        self._wav.setsampwidth(SAMPLE_WIDTH)
        self._wav.setframerate(sample_rate)

        self._queue: queue.Queue = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._frames_written = 0
        self._finalized = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            try:
                self._wav.writeframes(item.tobytes())
                with self._lock:
                    self._frames_written += len(item)
            except Exception:
                _logger.exception("WAV書き込み中にエラーが発生しました")

    def write(self, chunk: np.ndarray) -> None:
        """録音コールバックから呼ばれる。キューに積むだけで即座に返る。

        書き込みスレッドがディスクI/Oで詰まりキューが満杯の場合、メモリが
        録音長に比例して増え続けないよう、そのチャンクは諦めてログに残す
        （録音の一部が欠けるが、メモリ使用量の上限は保たれる）。
        """
        try:
            self._queue.put_nowait(float32_to_int16(chunk))
        except queue.Full:
            _logger.warning("WAV書き込みが追いついていないため、録音チャンクを破棄しました")

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames_written

    def finalize(self, timeout: float = 2.0) -> Path | None:
        """書き込みスレッドを止めてファイルを確定する。冪等。

        Returns:
            1フレーム以上書き込まれていれば保存先パス、0フレームなら
            ファイルを削除して None。
        """
        with self._lock:
            already_finalized = self._finalized
            self._finalized = True

        if not already_finalized:
            self._queue.put(_SENTINEL)
            self._thread.join(timeout)
            if self._thread.is_alive():
                # 書き込みスレッドがまだ writeframes() の途中である可能性がある。
                # ここで close() すると seek が交錯しヘッダーを壊しかねないため、
                # ハンドルは解放せず放棄する。最後に成功した writeframes() の時点で
                # 既にディスク上のヘッダーは有効なので、ファイル自体は失われない。
                _logger.warning(
                    "WAV書き込みスレッドがタイムアウトしました（%s秒）。"
                    "ファイルハンドルは解放せず放棄します",
                    timeout,
                )
            else:
                with contextlib.suppress(Exception):
                    self._wav.close()

        if self.frames_written == 0:
            self.path.unlink(missing_ok=True)
            return None
        return self.path

    def abort(self) -> None:
        """録音開始に失敗した際、オープン済みファイルを破棄する。"""
        with self._lock:
            already_finalized = self._finalized
            self._finalized = True
        if already_finalized:
            return
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._wav.close()
        self.path.unlink(missing_ok=True)
