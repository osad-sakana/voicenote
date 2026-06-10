"""バックグラウンドスレッドからメインスレッドへ UI 更新を流すためのキュー。

CustomTkinter (Tk) のウィジェット操作はメインスレッドからしか行えないため、
バックグラウンドスレッドからの呼び出しはここを経由する。

macOS では `Tk.after` を非メインスレッドから呼ぶと SIGBUS で落ちるため、
`put` ではキューに積むだけにし、メインスレッドが `poll` で消費する。
"""

import queue
from collections.abc import Callable


class ThreadSafeUIQueue:
    """非メインスレッドから安全に UI 更新をスケジュールするキュー。"""

    POLL_INTERVAL_MS = 50

    def __init__(self, widget, alive_fn: Callable[[], bool]):
        """
        Args:
            widget: ポーリングに `after` を使う Tk ウィジェット (通常はトップレベル)。
            alive_fn: アプリがまだ動いているかを返す関数 (終了後はポーリングを止める)。
        """
        self._queue: queue.Queue = queue.Queue()
        self._widget = widget
        self._alive_fn = alive_fn

    def start(self):
        """ポーリングを開始する。メインスレッドから呼ぶこと。"""
        self._poll()

    def submit(self, fn: Callable, *args):
        """バックグラウンドスレッドからメインスレッドで実行する関数をキューに積む。"""
        if self._alive_fn():
            self._queue.put((fn, args))

    def _poll(self):
        try:
            while True:
                fn, args = self._queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        if self._alive_fn():
            self._widget.after(self.POLL_INTERVAL_MS, self._poll)
