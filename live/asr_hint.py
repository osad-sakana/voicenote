"""Whisper 用語ヒント(initial_prompt)

ライブの参考情報(参加者・メモ)から、固有名詞・専門用語の誤認識を減らすための
initial_prompt を作る純関数群。Tk 非依存。

根拠: faster-whisper / mlx-whisper はともにプロンプトの末尾(約223トークン)だけを残すため、
「用語 → 参加者」の順に並べて参加者を末尾に置き、全体を MAX_HINT_CHARS 文字に収める。
上限を超える場合は先頭のメモ用語から削り、それでも収まらなければ先頭の参加者から削る。
ヒントは参加者名・メモ由来の個人情報を含みうるため、ログや例外メッセージに出さない。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from live.context import LiveContext
from live.hallucination import normalize_for_compare

MAX_HINT_CHARS = 120  # ヒント全体の最大文字数
MAX_TERM_CHARS = 20  # これを超える語は文とみなして捨てる
HINT_MIN_SEGMENT_SEC = 1.0  # これより短い区間にはヒントを付けない(反響しやすいため)

# 反響判定: ヒントの部分文字列とみなす最小文字数と、ヒント由来の語が占める比率・語数
ECHO_MIN_SUBSTRING_CHARS = 6
ECHO_MIN_TERM_RATIO = 0.8
ECHO_MIN_DISTINCT_TERMS = 2

_TERMS_LABEL = "用語: "
_PARTICIPANTS_LABEL = "参加者: "
_JOINER = "、"
_TERMINATOR = "。"

# 丸数字は NFKC で普通の数字になり「1月」などと区別できなくなるため、正規化前に除く
_CIRCLED_PREFIX = re.compile(r"^\s*[①-⓿❶-➓㉑-㉟㊱-㊿]\s*")
_LINE_MARKER = re.compile(r"^\s*(?:[-*•・#>]+|\(?\d+[.)])\s*")
_TOKEN_MARKER = re.compile(r"^[-*•・#>]+")
_TOKEN_SEPARATORS = re.compile(r"[\s、,，。]+")
_EDGE_PUNCTUATION = ".。:：!！?？;；"
_URL = re.compile(r"(?:[a-z][a-z0-9+.-]*://|www\.)", re.IGNORECASE)
_HINT_TERM_SEPARATORS = re.compile(f"[{_JOINER}{_TERMINATOR}]")


def _strip_line_marker(line: str) -> str:
    """行頭の箇条書き記号・番号(丸数字は正規化前に)を除き、NFKC 正規化した行を返す"""
    without_circled = _CIRCLED_PREFIX.sub("", line)
    return _LINE_MARKER.sub("", unicodedata.normalize("NFKC", without_circled))


def _clean_token(token: str) -> str:
    """語頭の記号と前後の句読点を除く"""
    return _TOKEN_MARKER.sub("", token).strip(_EDGE_PUNCTUATION)


def _is_term(token: str) -> bool:
    """用語として採用するか(文字を含み、URL でなく、長すぎない)"""
    if not token or len(token) > MAX_TERM_CHARS or _URL.search(token):
        return False
    return any(unicodedata.category(ch).startswith("L") for ch in token)


def extract_memo_terms(memo: str) -> tuple[str, ...]:
    """メモから用語を取り出す(NFKC 正規化・区切りで分割・記号や文・URL を除外・重複除去)"""
    seen: dict[str, None] = {}
    for line in memo.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        for token in _TOKEN_SEPARATORS.split(_strip_line_marker(line)):
            term = _clean_token(token)
            if _is_term(term):
                seen.setdefault(term, None)
    return tuple(seen)


def _section(label: str, items: Sequence[str]) -> str:
    return f"{label}{_JOINER.join(items)}{_TERMINATOR}" if items else ""


def _fit_tail(items: tuple[str, ...], label: str, budget: int) -> tuple[str, ...]:
    """末尾側を残し、先頭から削って section の長さを budget 以下にする"""
    start = 0
    while start < len(items) and len(_section(label, items[start:])) > budget:
        start += 1
    return items[start:]


def build_asr_hint(context: LiveContext | None) -> str | None:
    """参考情報から「用語: a、b。参加者: x、y。」形式のヒントを作る。空なら None"""
    if context is None:
        return None
    participants = tuple(
        dict.fromkeys(unicodedata.normalize("NFKC", name) for name in context.participants)
    )
    terms = tuple(t for t in extract_memo_terms(context.memo) if t not in participants)
    kept_participants = _fit_tail(participants, _PARTICIPANTS_LABEL, MAX_HINT_CHARS)
    rest = MAX_HINT_CHARS - len(_section(_PARTICIPANTS_LABEL, kept_participants))
    kept_terms = _fit_tail(terms, _TERMS_LABEL, rest)
    hint = _section(_TERMS_LABEL, kept_terms) + _section(_PARTICIPANTS_LABEL, kept_participants)
    return hint or None


def _hint_terms(hint: str) -> tuple[str, ...]:
    """build_asr_hint が作った文字列から語を取り出す(ラベルは除く)"""
    body = hint.replace(_TERMS_LABEL, _TERMINATOR).replace(_PARTICIPANTS_LABEL, _TERMINATOR)
    return tuple(t for t in _HINT_TERM_SEPARATORS.split(body) if t)


def _compare_form(text: str) -> str:
    """比較用に NFKC → 空白・句読点除去・小文字化し、コロンも除く"""
    return normalize_for_compare(unicodedata.normalize("NFKC", text)).replace(":", "")


def _term_coverage(normalized: str, terms: tuple[str, ...]) -> tuple[int, int]:
    """(ヒントの語が覆う文字数, 一致した異なる語の数)"""
    covered: set[int] = set()
    matched = 0
    for term in terms:
        hits = [m.start() for m in re.finditer(re.escape(term), normalized)] if term else []
        if hits:
            matched += 1
            covered.update(i for start in hits for i in range(start, start + len(term)))
    return len(covered), matched


def is_hint_echo(text: str, hint: str | None) -> bool:
    """認識結果がヒントの読み上げ(反響)なら True

    ヒントの部分文字列で ECHO_MIN_SUBSTRING_CHARS 文字以上、またはヒント由来の語が
    出力の ECHO_MIN_TERM_RATIO 以上を占め、異なる語が ECHO_MIN_DISTINCT_TERMS 以上。
    1語の名前だけの短い発話は捨てない。
    """
    if not hint:
        return False
    normalized = _compare_form(text)
    if not normalized:
        return False
    terms = tuple(dict.fromkeys(_compare_form(t) for t in _hint_terms(hint)))
    if len(normalized) >= ECHO_MIN_SUBSTRING_CHARS and (
        normalized in _compare_form(hint) or normalized in "".join(terms)
    ):
        return True
    covered, matched = _term_coverage(normalized, terms)
    return matched >= ECHO_MIN_DISTINCT_TERMS and covered >= len(normalized) * ECHO_MIN_TERM_RATIO
