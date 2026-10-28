"""ライブ文字起こしの参考情報（参加者・メモ）

要約の表記ゆれ補正・論点把握の参考にし、ライブノートにも残す。値はすべて正規化済みの
イミュータブルな LiveContext として扱う。本文は個人情報を含みうるため repr やログに出さない。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_PARTICIPANTS = 30  # 参加者の人数上限（超えた分は捨てる）
MAX_PARTICIPANT_CHARS = 40  # 1人あたりの最大文字数（超えた分は切り詰める）
MAX_MEMO_CHARS = 2000  # メモの最大文字数（先頭側を残す）

_PARTICIPANT_SEPARATORS = re.compile(r"[、，,\r\n]")
# 表示順を入れ替えて見た目を偽装できる双方向制御文字（Cf のうち除去するもの）
_BIDI_CONTROLS = frozenset(
    chr(code) for code in (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A))
)


def _remove_controls(text: str, keep: frozenset[str] = frozenset()) -> str:
    """制御文字（Cc）と双方向制御文字を除去する。keep に含む文字は残す"""
    return "".join(
        ch
        for ch in text
        if ch in keep or (unicodedata.category(ch) != "Cc" and ch not in _BIDI_CONTROLS)
    )


def _normalize_name(name: str) -> str:
    """参加者名1件を正規化する（制御文字除去・前後空白除去・最大文字数で切り詰め）"""
    cleaned = _remove_controls(name).strip()
    return cleaned[:MAX_PARTICIPANT_CHARS].strip()


def _normalize_names(names) -> tuple[str, ...]:
    """参加者名の列を正規化する（空要素除去・重複除去（順序保持）・人数上限）"""
    seen: dict[str, None] = {}
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"参加者名は文字列が必要です: {type(name).__name__}")
        normalized = _normalize_name(name)
        if normalized and normalized not in seen:
            seen[normalized] = None
        if len(seen) >= MAX_PARTICIPANTS:
            break
    return tuple(seen)


def parse_participants(text: str | None) -> tuple[str, ...]:
    """参加者入力欄の文字列を名前のタプルにする

    読点「、」「，」・カンマ「,」・改行で区切り、前後の空白を除き、空要素と重複を除く
    （順序は最初の出現順）。1人あたり MAX_PARTICIPANT_CHARS 文字、最大 MAX_PARTICIPANTS 人。
    """
    if text is None:
        return ()
    if not isinstance(text, str):
        raise TypeError(f"参加者は文字列が必要です: {type(text).__name__}")
    return _normalize_names(_PARTICIPANT_SEPARATORS.split(text))


def _normalize_memo(memo: str) -> tuple[str, bool]:
    """メモを正規化する。(正規化後のメモ, 切り詰めたか) を返す

    改行を \\n に統一し、改行・タブ以外の制御文字を除き、前後の空白を除いてから
    先頭 MAX_MEMO_CHARS 文字を残す。
    """
    unified = memo.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _remove_controls(unified, keep=frozenset("\n\t")).strip()
    if len(cleaned) <= MAX_MEMO_CHARS:
        return cleaned, False
    return cleaned[:MAX_MEMO_CHARS].rstrip(), True


@dataclass(frozen=True)
class LiveContext:
    """ライブ中に入力された参考情報。生成時に正規化される（直接生成しても上限は守られる）

    Attributes:
        participants: 参加者名（正規化済み・重複なし・最大 MAX_PARTICIPANTS 人）
        memo: メモ本文（正規化済み・最大 MAX_MEMO_CHARS 文字）
        memo_truncated: メモが上限を超えて切り詰められたか
    """

    participants: tuple[str, ...] = ()
    memo: str = ""
    memo_truncated: bool = False

    def __post_init__(self):
        if isinstance(self.participants, (str, bytes)):
            raise TypeError("participants は文字列のタプルが必要です（文字列は不可）")
        if not isinstance(self.memo, str):
            raise TypeError(f"memo は文字列が必要です: {type(self.memo).__name__}")
        if not isinstance(self.memo_truncated, bool):
            raise TypeError("memo_truncated は真偽値が必要です")
        memo, truncated = _normalize_memo(self.memo)
        # frozen dataclass の生成時正規化（生成後は変更されない）
        object.__setattr__(self, "participants", _normalize_names(self.participants))
        object.__setattr__(self, "memo", memo)
        object.__setattr__(self, "memo_truncated", self.memo_truncated or truncated)

    def __repr__(self) -> str:
        # 参加者名・メモ本文はログや例外表示に出さない（件数と長さだけ）
        return (
            f"LiveContext(participants={len(self.participants)}, memo_chars={len(self.memo)}, "
            f"memo_truncated={self.memo_truncated})"
        )

    @property
    def is_empty(self) -> bool:
        """参加者もメモも無ければ True"""
        return not self.participants and not self.memo

    @classmethod
    def from_inputs(cls, participants_text: str | None, memo_text: str | None) -> LiveContext:
        """画面の入力欄（参加者・メモ）の文字列から生成する。None は空として扱う"""
        if memo_text is not None and not isinstance(memo_text, str):
            raise TypeError(f"メモは文字列が必要です: {type(memo_text).__name__}")
        return cls(parse_participants(participants_text), memo_text or "")
