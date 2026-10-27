"""
発話区間キューモジュール
文字起こし待ちの区間を保持する、満杯時に最古を破棄する有界キュー
"""

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """
    スレッドセーフな有界キュー。満杯時の put() は最古の要素を破棄して新しい要素を入れる。
    リアルタイム表示では「古い区間より最新の区間」を優先するため、BoundedChunkQueue
    (最新を破棄)とは逆の方針を取る。
    """

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError(f"maxsize は正の整数が必要です: {maxsize}")
        self._items: deque[T] = deque()
        self._maxsize = maxsize
        self._cond = threading.Condition()

    def put(self, item: T) -> T | None:
        """要素を追加する。満杯で最古を破棄した場合はその要素を返す(無ければ None)"""
        with self._cond:
            evicted = self._items.popleft() if len(self._items) >= self._maxsize else None
            self._items.append(item)
            self._cond.notify()
            return evicted

    def get(self, timeout: float | None = None) -> T | None:
        """最古の要素を取り出す。timeout 秒以内に取れなければ None"""
        with self._cond:
            if not self._cond.wait_for(lambda: bool(self._items), timeout=timeout):
                return None
            return self._items.popleft()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
