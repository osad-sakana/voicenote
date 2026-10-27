"""
ライブ文字起こしの画面状態(Rich 描画用)
ワーカースレッドのコールバックは状態を更新するだけで、描画は Rich Live が get_renderable 経由で行う
"""

import threading

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from live.note import format_timestamp
from live.session_models import LiveSegment

DEFAULT_MAX_SEGMENTS = 8


class LiveView:
    """直近 N 区間・要約・状態表示を保持するスレッドセーフな状態オブジェクト"""

    def __init__(self, max_segments: int = DEFAULT_MAX_SEGMENTS):
        if max_segments <= 0:
            raise ValueError(f"max_segments は正の整数が必要です: {max_segments}")
        self._max_segments = max_segments
        self._lock = threading.Lock()
        self._segments: tuple[LiveSegment, ...] = ()
        self._summary = ""
        self._status = "準備中..."

    @property
    def recent_segments(self) -> tuple[LiveSegment, ...]:
        with self._lock:
            return self._segments

    def add_segment(self, index: int, start_sec: float, end_sec: float, text: str) -> None:
        segment = LiveSegment(index, start_sec, end_sec, text)
        with self._lock:
            self._segments = (*self._segments, segment)[-self._max_segments :]

    def set_summary(self, summary: str) -> None:
        with self._lock:
            self._summary = summary

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def render(self) -> Group:
        """現在の状態から描画用オブジェクトを作る"""
        with self._lock:
            segments, summary, status = self._segments, self._summary, self._status
        body = Text()
        for i, segment in enumerate(segments):
            if i:
                body.append("\n")
            body.append(f"[{format_timestamp(segment.start_sec)}] ", style="cyan")
            body.append(segment.text)
        if not segments:
            body.append("（発話を待っています）", style="dim")
        return Group(
            Panel(body, title="ライブ文字起こし", border_style="green"),
            Panel(Text(summary or "（要約はまだありません）"), title="要約", border_style="cyan"),
            Text(status, style="dim"),
        )
