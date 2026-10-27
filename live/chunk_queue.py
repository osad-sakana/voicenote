"""
有界チャンクキューモジュール
音声コールバックスレッドをブロックさせないための、満杯時破棄型キュー
"""

import queue
import threading
from typing import Any


class BoundedChunkQueue:
    """
    スレッドセーフな有界キュー。
    put_nowait() は満杯時にブロックも例外送出もせず、チャンクを破棄して dropped を加算する。
    get() はタイムアウトすると None を返す(queue.Empty は送出しない)。
    """

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError(f"maxsize は正の整数が必要です: {maxsize}")
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def dropped(self) -> int:
        """満杯のため破棄されたチャンク数"""
        with self._lock:
            return self._dropped

    def put_nowait(self, chunk: Any) -> bool:
        """チャンクを投入する。受け入れたら True、満杯で破棄したら False"""
        try:
            self._queue.put_nowait(chunk)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def get(self, timeout: float | None = None) -> Any | None:
        """チャンクを取り出す。timeout 秒以内に取れなければ None"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """現在のキュー長(概算)"""
        return self._queue.qsize()
