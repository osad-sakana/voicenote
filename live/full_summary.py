"""録音全体の文字起こしからの全文要約。

ローリング要約（live.summarizer）は直近の差分だけを見て少しずつ更新するため、
録音終了後に全セグメントの文字起こしから要約し直すのがこのモジュール。
長い文字起こしは max_chunk_chars ごとに分割して塊ごとに要約し、複数あれば統合要約する。
LLM 呼び出しは注入された backend に限り、プロンプト組み立て・分割は副作用のない純粋関数。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from live.context import LiveContext
from live.local_llm import ChatBackend
from live.summarizer import (
    CONTEXT_RULES,
    GROUNDING_RULES,
    OUTPUT_FORMAT,
    accept_summary,
    render_context_blocks,
)

_logger = logging.getLogger("voicenote")

FULL_CHUNK_CHARS = 3000
# 全文要約の上限時間 = 呼び出し回数(塊数 + 統合1回) × これ。下限・上限で丸める
FULL_SUMMARY_SEC_PER_CALL = 60.0
FULL_SUMMARY_MIN_TIMEOUT_SEC = 120.0
FULL_SUMMARY_MAX_TIMEOUT_SEC = 1200.0
# 見積もりちょうどの期限でも、計算から呼び出しまでの経過で「残り不足」と判定されないための余裕
FULL_SUMMARY_MARGIN_SEC = 5.0

_FULL_SYSTEM_PROMPT = f"""\
あなたは会議や講義の文字起こしを要約するアシスタントです。
「文字起こし」を読み、要約してください。

{OUTPUT_FORMAT}
規則:
{GROUNDING_RULES}\
- 『現在の話題:』は直近の文字起こし（末尾側）から判断する
- 簡潔に書く
"""

_MERGE_SYSTEM_PROMPT = f"""\
あなたは会議や講義の要約を統合するアシスタントです。
1つの文字起こしを前から順に分けて要約した「部分要約」を読み、全体の要約に統合してください。

{OUTPUT_FORMAT}
規則:
{GROUNDING_RULES}\
- 部分要約に無い内容を足さない。重複する要点はまとめる
- 『現在の話題:』は直近の内容（最後の部分要約）から判断する
- 簡潔に書く
"""


class CancelFlag(Protocol):
    """threading.Event 風のキャンセル指示（is_set だけを使う）。"""

    def is_set(self) -> bool: ...


def split_chunks(texts: Sequence[str], max_chars: int) -> tuple[str, ...]:
    """セグメント文を改行で連結しつつ、1塊が max_chars 以下になるよう分割する。

    セグメントは途中で切らずに詰め、1件で max_chars を超えるものだけ文字数で切る。
    空白のみのセグメントは捨てる。
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars は正の整数で指定してください: {max_chars}")
    pieces = [
        part[i : i + max_chars]
        for part in (t.strip() for t in texts)
        if part
        for i in range(0, len(part), max_chars)
    ]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        current = piece
    return (*chunks, current) if current else tuple(chunks)


def _with_context(
    system: str, user_content: str, context: LiveContext | None
) -> list[dict[str, str]]:
    """参考情報が空でなければ規則と区切りブロックを足す（build_messages と同じ様式）。"""
    if context is not None and not context.is_empty:
        system = f"{system}\n{CONTEXT_RULES}"
        user_content = f"{render_context_blocks(context)}\n\n{user_content}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_full_messages(text: str, context: LiveContext | None) -> list[dict[str, str]]:
    """文字起こし1塊を要約するメッセージ列を組み立てる。"""
    return _with_context(_FULL_SYSTEM_PROMPT, f"# 文字起こし\n{text}", context)


def build_merge_messages(
    partials: Sequence[str], context: LiveContext | None
) -> list[dict[str, str]]:
    """部分要約（文字起こしの前から順）を統合するメッセージ列を組み立てる。"""
    body = "\n\n".join(f"# 部分要約{i}\n{p}" for i, p in enumerate(partials, start=1))
    return _with_context(_MERGE_SYSTEM_PROMPT, body, context)


def _should_stop(deadline: float, clock: Callable[[], float], cancel: CancelFlag | None) -> bool:
    return (cancel is not None and cancel.is_set()) or clock() >= deadline


def _ask(backend: ChatBackend, messages: list[dict[str, str]]) -> str | None:
    """1回呼び出して採否を判定する。不採用・例外は None（本文はログに出さない）。"""
    try:
        raw = backend.complete(messages)
    except Exception as e:
        # 例外のメッセージは送信内容や接続情報を含みうるので型名だけを出す
        _logger.warning("ローカルLLMによる全文要約に失敗しました: %s", type(e).__name__)
        return None
    # prev="" なので長さ比の比較はせず、最小文字数と日本語の妥当性だけで判定される
    accepted = accept_summary(raw, prev="")
    if not accepted:
        _logger.warning("全文要約の結果が不正なため採用しません")
        return None
    return accepted


def summarize_full(
    backend: ChatBackend,
    texts: Sequence[str],
    context: LiveContext | None,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
    cancel: CancelFlag | None = None,
    max_chunk_chars: int = FULL_CHUNK_CHARS,
    sec_per_call: float = FULL_SUMMARY_SEC_PER_CALL,
    on_skip: Callable[[int, float, float], None] | None = None,
) -> str | None:
    """全セグメントの文字起こしから要約を作る。作れなければ None。

    必要時間の見積もり((塊数 + 統合1回) × sec_per_call)が、deadline までの残り時間を
    超えるなら最初から呼び出さずに None を返す(途中で打ち切られる要約に時間を使わない)。
    そのとき on_skip(塊数, 見積もり秒, 残り秒) を呼ぶ(利用者への通知用。本文は渡さない)。
    各 LLM 呼び出しの前に deadline（clock 基準の絶対時刻）と cancel を確認し、
    超過・キャンセル・例外・不採用の応答が1つでもあれば None を返す。
    1塊に収まれば呼び出しは1回だけ。複数塊なら塊ごとに要約してから統合する。
    """
    chunks = split_chunks(texts, max_chunk_chars)
    estimate = _estimated_sec(len(chunks), sec_per_call)
    remaining = deadline - clock()
    if chunks and estimate > remaining:
        # 残り時間内に終わらない見込みなら最初から試さない(本文は出さず、塊数だけを残す)
        _logger.warning(
            "全文要約は時間内に終わらない見込みのため省きます（%d 塊・見込み %.0f 秒 > 残り %.0f 秒）",
            len(chunks),
            estimate,
            max(0.0, remaining),
        )
        if on_skip is not None:
            try:
                on_skip(len(chunks), estimate, max(0.0, remaining))
            except Exception:
                _logger.exception("全文要約の省略の通知に失敗しました")
        return None
    partials: list[str] = []
    for chunk in chunks:
        if _should_stop(deadline, clock, cancel):
            return None
        partial = _ask(backend, build_full_messages(chunk, context))
        if partial is None:
            return None
        partials.append(partial)
    if len(partials) <= 1:
        return partials[0] if partials else None
    if _should_stop(deadline, clock, cancel):
        return None
    return _ask(backend, build_merge_messages(partials, context))


def _estimated_sec(chunk_count: int, sec_per_call: float = FULL_SUMMARY_SEC_PER_CALL) -> float:
    """必要時間の見積もり(LLM の呼び出し回数 = 塊数 + 統合1回)"""
    return sec_per_call * (chunk_count + 1)


def full_summary_timeout(texts: Sequence[str], max_chunk_chars: int = FULL_CHUNK_CHARS) -> float:
    """summarize_full に与える上限時間(秒)。LLM の呼び出し回数(塊数 + 統合1回)に比例させる

    見積もりを下限・上限で丸め、FULL_SUMMARY_MARGIN_SEC を足す。見積もりが上限を超える量は
    残り時間が足りないため summarize_full が最初から試さない。
    """
    estimate = _estimated_sec(len(split_chunks(texts, max_chunk_chars)))
    clamped = min(FULL_SUMMARY_MAX_TIMEOUT_SEC, max(FULL_SUMMARY_MIN_TIMEOUT_SEC, estimate))
    return clamped + FULL_SUMMARY_MARGIN_SEC
