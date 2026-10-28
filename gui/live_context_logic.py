"""ライブ画面の参加者・メモ入力のロジック(Tk 非依存)

入力欄の文字列 → LiveContext の変換、上限超過の案内文、入力変更のデバウンス、
ワークフローへの反映(重複送信の抑止・開始時の即時反映・クリア)を扱う。
入力本文は個人情報を含みうるため、ログ・案内文には一切出さない。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from live.context import (
    MAX_MEMO_CHARS,
    MAX_PARTICIPANT_CHARS,
    MAX_PARTICIPANTS,
    LiveContext,
    clean_participant_name,
    parse_participants,
)

DEBOUNCE_MS = 600  # 入力が止まってからワークフローへ反映するまでの待ち時間

_logger = logging.getLogger("voicenote")

# live/context.py の parse_participants と同じ区切り(一致はテストで確認する)
_PARTICIPANT_SEPARATORS = re.compile(r"[、，,\r\n]")

Schedule = Callable[[int, Callable[[], None]], Any]  # (ms, fn) -> 取り消し用トークン
Cancel = Callable[[Any], None]


def build_context(participants_text: str | None, memo_text: str | None) -> LiveContext:
    """入力欄の文字列から LiveContext を作る(正規化・上限は LiveContext 側で行う)"""
    return LiveContext.from_inputs(participants_text, memo_text)


def should_push(last_pushed: LiveContext | None, current: LiveContext) -> bool:
    """前回送った値と正規化後に異なれば True(未送信なら常に True)"""
    return last_pushed is None or last_pushed != current


def _participant_names(text: str | None) -> tuple[str, ...]:
    """上限を掛ける前の参加者名(区切りごとに正規化・空と重複を除く)"""
    if not text:
        return ()
    names: dict[str, None] = {}
    for piece in _PARTICIPANT_SEPARATORS.split(text):
        normalized = parse_participants(piece)
        if normalized:
            names[normalized[0]] = None
    return tuple(names)


def count_participants(text: str | None) -> int:
    """入力された参加者の人数(上限で切る前。重複・空要素は数えない)"""
    return len(_participant_names(text))


def _has_long_name(text: str | None) -> bool:
    """切り詰められる参加者名があるか(保存値と同じく、不可視文字を除いた後の長さで測る)"""
    if not text:
        return False
    return any(
        len(clean_participant_name(piece)) > MAX_PARTICIPANT_CHARS
        for piece in _PARTICIPANT_SEPARATORS.split(text)
    )


def limit_notice(
    participants_text: str | None, memo_text: str | None, *, compact: bool = False
) -> str | None:
    """上限を超えた入力への案内文。超えていなければ None(本文は含めない)

    compact: 小さい画面向けに1行へ省略する(要約欄に縦幅を回すため)
    """
    over_count = count_participants(participants_text) > MAX_PARTICIPANTS
    long_name = _has_long_name(participants_text)
    memo_over = build_context(None, memo_text).memo_truncated
    if compact:
        limits = [
            label
            for label, over in (
                (f"参加者{MAX_PARTICIPANTS}人", over_count),
                (f"名前{MAX_PARTICIPANT_CHARS}文字", long_name),
                (f"メモ{MAX_MEMO_CHARS}文字", memo_over),
            )
            if over
        ]
        return f"上限を超えた分は要約に使いません（{'・'.join(limits)}）" if limits else None
    notices = []
    if over_count:
        notices.append(f"参加者は先頭{MAX_PARTICIPANTS}人だけを要約に使います")
    if long_name:
        notices.append(f"参加者名は1人{MAX_PARTICIPANT_CHARS}文字までに切り詰めます")
    if memo_over:
        notices.append(f"メモは先頭{MAX_MEMO_CHARS}文字だけを要約に使います")
    return " / ".join(notices) if notices else None


class Debouncer:
    """最後の trigger から delay_ms 後に1回だけ callback を呼ぶ(スケジューラは注入)"""

    def __init__(
        self,
        schedule: Schedule,
        cancel: Cancel,
        callback: Callable[[], None],
        delay_ms: int = DEBOUNCE_MS,
    ):
        if delay_ms <= 0:
            raise ValueError(f"delay_ms は正の値が必要です: {delay_ms}")
        self._schedule = schedule
        self._cancel = cancel
        self._callback = callback
        self._delay_ms = delay_ms
        self._token: Any = None

    @property
    def pending(self) -> bool:
        return self._token is not None

    def trigger(self) -> None:
        self.cancel()
        self._token = self._schedule(self._delay_ms, self._fire)

    def cancel(self) -> None:
        if self._token is not None:
            token, self._token = self._token, None
            self._cancel(token)

    def flush(self) -> bool:
        """待ちがあれば取り消して今すぐ callback を呼ぶ。呼んだら True"""
        if self._token is None:
            return False
        self.cancel()
        self._callback()
        return True

    def _fire(self) -> None:
        self._token = None
        self._callback()


class ContextSync:
    """入力欄とワークフローの参考情報を同期する(メインスレッド専用)

    read_inputs: (参加者の文字列, メモの文字列) を返す
    push: LiveWorkflow.update_context 相当(None は空)
    on_notice: 上限超過の案内文(無ければ None)を表示する
    compact_notice: 案内文を1行に省略する(小さい画面で要約欄に縦幅を回すため)
    """

    def __init__(
        self,
        *,
        read_inputs: Callable[[], tuple[str, str]],
        push: Callable[[LiveContext | None], None],
        on_notice: Callable[[str | None], None],
        schedule: Schedule,
        cancel: Cancel,
        delay_ms: int = DEBOUNCE_MS,
        compact_notice: bool = False,
    ):
        self._read_inputs = read_inputs
        self._compact_notice = compact_notice
        self._push = push
        self._on_notice = on_notice
        self._last: LiveContext | None = None
        self._debouncer = Debouncer(schedule, cancel, self._sync, delay_ms)

    def on_input_changed(self) -> None:
        """入力欄が変わった(デバウンスして反映する)"""
        self._debouncer.trigger()

    def flush(self, *, force: bool = False) -> None:
        """デバウンス待ちを取り消し、最新の入力を今すぐ反映する(ライブ開始・終了時)"""
        self._debouncer.cancel()
        self._sync(force=force)

    def flush_pending(self, *, force: bool = False) -> bool:
        """デバウンス待ちの編集がある時だけ、今すぐ反映する(停止・中断の直前)。反映したら True

        macOS ではボタンがフォーカスを取らず入力欄の <FocusOut> が発火しないので、
        フォーカスに頼らずここで送る。待ちが無ければ何も送らない。
        """
        if not self._debouncer.pending:
            return False
        self.flush(force=force)
        return True

    def clear(self) -> None:
        """参考情報を空にする(入力欄を空にするのは呼び出し側)"""
        self._debouncer.cancel()
        self._on_notice(None)
        if self._send(None):
            self._last = LiveContext()

    def cancel(self) -> None:
        self._debouncer.cancel()

    def _sync(self, *, force: bool = False) -> None:
        participants, memo = self._read_inputs()
        self._on_notice(limit_notice(participants, memo, compact=self._compact_notice))
        context = build_context(participants, memo)
        if (force or should_push(self._last, context)) and self._send(context):
            self._last = context

    def _send(self, context: LiveContext | None) -> bool:
        try:
            self._push(context)
        except Exception as e:
            # 本文はログに出さない(例外の詳細も本文を含みうるので型名だけ)
            _logger.error("参加者・メモを要約に反映できませんでした: %s", type(e).__name__)
            return False
        return True
