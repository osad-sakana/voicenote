"""リアルタイム文字起こしのローリング要約。

前回の要約と「前回成功以降に増えた文字起こしテキスト」だけをローカルLLMへ渡し、
要約を少しずつ更新していく。判定ロジックは副作用のない純粋関数として切り出し、
`RollingSummarizer` は状態管理（送信中フラグ・未要約テキスト）だけを担う。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable

from live.language_guard import is_plausible_japanese
from live.local_llm import ChatBackend

_logger = logging.getLogger("voicenote")

DEFAULT_INTERVAL_SEC = 30.0
DEFAULT_MAX_BACKLOG_SEC = 10.0
DEFAULT_MAX_CHARS = 4000
DEFAULT_MIN_CHARS = 20
DEFAULT_MIN_RATIO = 0.3
_EMPTY_PREV_MARKER = "（なし）"
# float の丸め誤差（100 * 0.3 = 30.000000000000004）で境界値が弾かれないための許容差
_RATIO_EPSILON = 1e-9

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)

_SYSTEM_PROMPT = """\
あなたは会議や講義の文字起こしをリアルタイムに要約するアシスタントです。
「前回の要約」と「新しい文字起こし」を読み、要約を更新してください。

出力は次の形式のプレーンテキストのみとします（JSON や Markdown の見出しは使わない）。

現在の話題: <いま話されている話題を1行で>
進捗:
- <これまでに話された要点を箇条書きで>

規則:
- 入力に無い内容を追加しない（推測・補完・創作をしない）
- 前回の要約の要点は、新しい文字起こしと矛盾しない限り残す
- 必ず日本語で出力する。中国語・英語で書かない（固有名詞・専門用語は原文の表記のままでよい）
- 見出しは『現在の話題:』『進捗:』をこの表記のまま使う
- 新しい文字起こしが短すぎる・意味を成さない場合は、前回の要約をそのまま返す
- 簡潔に書く
"""


def build_messages(prev_summary: str, new_text: str, max_chars: int) -> list[dict[str, str]]:
    """LLM へ送るメッセージ列を組み立てる。

    新しいテキストが max_chars を超える場合は、直近の内容を優先して末尾側を残す。
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars は正の整数で指定してください: {max_chars}")
    trimmed = new_text[-max_chars:] if len(new_text) > max_chars else new_text
    prev = prev_summary.strip() or _EMPTY_PREV_MARKER
    user_content = f"# 前回の要約\n{prev}\n\n# 新しい文字起こし\n{trimmed}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def strip_think(text: str) -> str:
    """推論モデルが出力する `<think>...</think>` ブロックを除去する。

    閉じタグの無い `<think>`（出力途中で切れた場合）以降もすべて思考とみなして除去する。
    """
    without_blocks = _THINK_BLOCK_RE.sub("", text or "")
    return _UNCLOSED_THINK_RE.sub("", without_blocks).strip()


def accept_summary(
    new: str | None,
    prev: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_ratio: float = DEFAULT_MIN_RATIO,
) -> str:
    """LLM の出力を要約として採用するか判定し、採用する要約を返す。

    空・min_chars 未満・前回より極端に短い（len(new) < len(prev) * min_ratio）・
    日本語として不正（中国語の混入など。live.language_guard で判定）な場合は
    出力が壊れているとみなして前回の要約を返す。
    """
    cleaned = strip_think(new or "")
    if len(cleaned) < max(min_chars, 1):
        return prev
    if prev and len(cleaned) < len(prev) * min_ratio - _RATIO_EPSILON:
        return prev
    if not is_plausible_japanese(cleaned):
        return prev
    return cleaned


def should_summarize(
    now: float,
    last_run: float | None,
    interval: float,
    backlog_sec: float,
    max_backlog_sec: float,
    in_flight: bool,
    has_new_text: bool,
) -> bool:
    """要約を実行すべきタイミングかを返す。

    送信中でなく、前回実行から interval 秒以上経過し、文字起こしの遅れ（backlog_sec）が
    max_backlog_sec 以下で、新しいテキストがある場合のみ True。
    文字起こしが遅れている間は LLM に CPU/GPU を譲るため要約を見送る。
    """
    if in_flight or not has_new_text:
        return False
    if backlog_sec > max_backlog_sec:
        return False
    return last_run is None or now - last_run >= interval


class RollingSummarizer:
    """前回要約 + 新規テキストで要約を逐次更新する。スレッドセーフ。"""

    def __init__(
        self,
        backend: ChatBackend,
        model: str,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        max_backlog_sec: float = DEFAULT_MAX_BACKLOG_SEC,
        max_chars: int = DEFAULT_MAX_CHARS,
        min_chars: int = DEFAULT_MIN_CHARS,
        min_ratio: float = DEFAULT_MIN_RATIO,
        clock: Callable[[], float] = time.monotonic,
        on_warning: Callable[[str], None] | None = None,
    ):
        """
        Args:
            backend: チャットLLMバックエンド
            model: 使用モデル名（警告メッセージ等の表示用）
            interval_sec: 要約の最小実行間隔（秒）
            max_backlog_sec: この秒数を超えて文字起こしが遅れている間は要約しない
            max_chars: LLM に渡す新規テキストの最大文字数（末尾側を残す）
            min_chars: 要約として採用する最小文字数
            min_ratio: 前回要約に対する最小長さ比率
            clock: 現在時刻を返す関数（テスト用に注入可能）
            on_warning: 要約失敗時に警告メッセージを受け取るコールバック
        """
        if max_chars <= 0:
            raise ValueError(f"max_chars は正の整数で指定してください: {max_chars}")
        self._backend = backend
        self.model = model
        self._interval = interval_sec
        self._max_backlog = max_backlog_sec
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._min_ratio = min_ratio
        self._clock = clock
        self._on_warning = on_warning
        self._lock = threading.Lock()
        self._summary = ""
        # (連番, テキスト) のタプル。連番で「今回送信した分」を識別するため、
        # 送信中に古いチャンクが切り詰められても正しく消費済み分だけを除去できる。
        self._pending: tuple[tuple[int, str], ...] = ()
        self._next_seq = 0
        self._last_run: float | None = None
        self._in_flight = False

    @property
    def summary(self) -> str:
        with self._lock:
            return self._summary

    @property
    def in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    @property
    def has_pending_text(self) -> bool:
        with self._lock:
            return bool(self._pending)

    @property
    def pending_chars(self) -> int:
        with self._lock:
            return len(_join(self._pending))

    def add_text(self, text: str) -> None:
        """確定した文字起こしテキストを追加する。空白のみのテキストは無視する。"""
        stripped = text.strip()
        if not stripped:
            return
        with self._lock:
            appended = (*self._pending, (self._next_seq, stripped))
            self._next_seq += 1
            self._pending = _trim_to_tail(appended, self._max_chars)

    def maybe_run(self, now: float | None = None, backlog_sec: float = 0.0) -> bool:
        """条件を満たせば要約を同期実行する。要約が更新された場合のみ True。

        LLM 呼び出しは時間がかかるため、呼び出し側でワーカースレッドから呼ぶこと。
        送信中に別スレッドから呼ばれても二重送信はしない。例外は外に出さない。
        """
        current = self._clock() if now is None else now
        with self._lock:
            if not should_summarize(
                now=current,
                last_run=self._last_run,
                interval=self._interval,
                backlog_sec=backlog_sec,
                max_backlog_sec=self._max_backlog,
                in_flight=self._in_flight,
                has_new_text=bool(self._pending),
            ):
                return False
            self._in_flight = True
            self._last_run = current
            prev = self._summary
            snapshot = self._pending
        try:
            return self._run(prev, snapshot)
        finally:
            with self._lock:
                self._in_flight = False

    def _run(self, prev: str, snapshot: tuple[tuple[int, str], ...]) -> bool:
        messages = build_messages(prev, _join(snapshot), self._max_chars)
        try:
            raw = self._backend.complete(messages)
        except Exception as e:
            _logger.exception("ローカルLLMによる要約に失敗しました (model=%s)", self.model)
            self._warn(f"要約に失敗しました（前回の要約を維持します）: {e}")
            return False
        cleaned = strip_think(raw or "")
        accepted = accept_summary(raw, prev, self._min_chars, self._min_ratio)
        # prev が空の初回は「不採用で prev を返した」と「cleaned が空」が区別できないため、
        # 空出力は明示的に不採用として扱う
        if not cleaned or accepted != cleaned:
            # 要約本文（文字起こし由来）はログ・警告に出さない
            self._warn(_rejection_message(cleaned))
            return False
        last_seq = snapshot[-1][0]
        with self._lock:
            self._summary = accepted
            self._pending = tuple(item for item in self._pending if item[0] > last_seq)
        return True

    def _warn(self, message: str) -> None:
        if self._on_warning is None:
            return
        try:
            self._on_warning(message)
        except Exception:
            _logger.exception("要約の警告コールバックでエラーが発生しました")


def _rejection_message(cleaned: str) -> str:
    """不採用理由に応じた警告メッセージを返す（本文は含めない）。"""
    if cleaned and not is_plausible_japanese(cleaned):
        return "要約結果が日本語として不正なため（中国語の混入など）、前回の要約を維持します"
    return "要約結果が不正なため、前回の要約を維持します"


def _join(pending: tuple[tuple[int, str], ...]) -> str:
    return "\n".join(text for _, text in pending)


def _trim_to_tail(
    pending: tuple[tuple[int, str], ...], max_chars: int
) -> tuple[tuple[int, str], ...]:
    """結合後の長さが max_chars 以下になるまで古いチャンクを捨てる（最新の1件は必ず残す）。"""
    trimmed = pending
    while len(trimmed) > 1 and len(_join(trimmed)) > max_chars:
        trimmed = trimmed[1:]
    return trimmed
