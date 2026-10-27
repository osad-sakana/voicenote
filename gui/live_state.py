"""
GUI ライブ表示の状態管理(Tk 非依存・イミュータブル)

ワーカースレッドのコールバックから届いた区間・要約・状態・警告を、
純関数で新しい LiveViewState に反映する。描画側は状態を読むだけにする。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import NamedTuple

from live.note import format_timestamp
from live.session_models import LiveResult, LiveSegment

MAX_DISPLAY_LINES = 500  # テキストボックスに残す最大行数
WARNING_THROTTLE_SEC = 10.0  # 同じ種類の警告を受け付ける最小間隔
MAX_WARNINGS = 100  # 保持する警告の上限(超えたら古いものから捨てる)

_DETAIL_SEPARATOR = re.compile(r"[:：]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")


class LivePhase(Enum):
    """ライブ録音の段階"""

    IDLE = "idle"
    PREPARING = "preparing"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    DRAINING = "draining"
    SAVING = "saving"


@dataclass(frozen=True)
class LiveBanner:
    """画面上部に出す案内バナー。kind ごとに1つだけ表示する"""

    kind: str  # "model_download" | "ollama" | "vad_fallback" | "engine_fallback" | ...
    text: str


@dataclass(frozen=True)
class LiveViewState:
    """ライブ表示の状態。各フィールドはタプル/不変値のみで構成する"""

    segments: tuple[LiveSegment, ...] = ()  # 全区間(部分ノートの材料)
    summary: str = ""
    status: str = ""
    warnings: tuple[str, ...] = ()  # 重複除去・間引き済み
    banners: tuple[LiveBanner, ...] = ()
    phase: LivePhase = LivePhase.IDLE
    warning_times: tuple[tuple[str, float], ...] = ()  # 種類キーごとの最後に受理した時刻


class DisplayDelta(NamedTuple):
    """テキストボックスへの差分更新指示

    lines を末尾に追記する前に、先頭から trim_count 行を削除すること。
    new_count は次回 new_display_lines に渡す描画済み区間数。
    """

    lines: tuple[str, ...]
    new_count: int
    trim_count: int


def apply_segment(
    state: LiveViewState, index: int, start_sec: float, end_sec: float, text: str
) -> LiveViewState:
    """文字起こし済みの区間を末尾に追加した新しい状態を返す"""
    segment = LiveSegment(index, start_sec, end_sec, text)
    return replace(state, segments=(*state.segments, segment))


def apply_summary(state: LiveViewState, text: str) -> LiveViewState:
    """現在の要約を差し替えた新しい状態を返す(変化がなければ同じオブジェクト)"""
    if text == state.summary:
        return state
    return replace(state, summary=text)


def apply_status(state: LiveViewState, msg: str) -> LiveViewState:
    """状態メッセージを差し替えた新しい状態を返す(変化がなければ同じオブジェクト)"""
    if msg == state.status:
        return state
    return replace(state, status=msg)


def warning_kind(msg: str) -> str:
    """警告の種類キーを求める

    コロン(全角含む)以降の例外詳細を落とし、数値を `#` に置き換える。
    例: 「区間 12.3s の文字起こしに失敗しました: boom」→「区間 #s の文字起こしに失敗しました」
    """
    head = _DETAIL_SEPARATOR.split(msg, maxsplit=1)[0]
    return _NUMBER.sub("#", head).strip()


def apply_warning(state: LiveViewState, msg: str, now: float) -> LiveViewState:
    """警告を追加した新しい状態を返す

    - 空の警告・既に表示済みの同一文言は無視する
    - 同じ種類(warning_kind)は WARNING_THROTTLE_SEC 秒に1回まで(境界ちょうどは受理)
    - 無視した場合は元の状態オブジェクトをそのまま返す
    """
    text = msg.strip()
    if not text or text in state.warnings:
        return state
    kind = warning_kind(text)
    times = dict(state.warning_times)
    last = times.get(kind)
    if last is not None and now - last < WARNING_THROTTLE_SEC:
        return state
    new_times = (*((k, t) for k, t in state.warning_times if k != kind), (kind, now))
    warnings = (*state.warnings, text)[-MAX_WARNINGS:]
    return replace(state, warnings=warnings, warning_times=new_times)


def add_banner(state: LiveViewState, kind: str, text: str) -> LiveViewState:
    """案内バナーを追加する。同じ kind があれば位置を保って差し替える"""
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"バナーの kind は空でない文字列が必要です: {kind!r}")
    banner = LiveBanner(kind, text)
    if banner in state.banners:
        return state
    if any(b.kind == kind for b in state.banners):
        banners = tuple(banner if b.kind == kind else b for b in state.banners)
    else:
        banners = (*state.banners, banner)
    return replace(state, banners=banners)


def clear_banner(state: LiveViewState, kind: str) -> LiveViewState:
    """指定 kind のバナーを消した新しい状態を返す(無ければ同じオブジェクト)"""
    banners = tuple(b for b in state.banners if b.kind != kind)
    if len(banners) == len(state.banners):
        return state
    return replace(state, banners=banners)


def set_phase(state: LiveViewState, phase: LivePhase) -> LiveViewState:
    """ライブ段階を変更した新しい状態を返す(同じ段階なら同じオブジェクト)"""
    if not isinstance(phase, LivePhase):
        raise TypeError(f"phase は LivePhase が必要です: {phase!r}")
    if phase is state.phase:
        return state
    return replace(state, phase=phase)


def format_segment_line(segment: LiveSegment) -> str:
    """区間を `[MM:SS] テキスト` の1行に整形する(改行・連続空白は1つの空白にまとめる)"""
    text = _WHITESPACE.sub(" ", segment.text).strip()
    return f"[{format_timestamp(segment.start_sec)}] {text}"


def new_display_lines(state: LiveViewState, rendered_count: int) -> DisplayDelta:
    """描画済み区間数 rendered_count 以降の差分行と、上限超過時の削除行数を返す

    テキストボックスには常に最大 MAX_DISPLAY_LINES 行だけ残す前提で trim_count を計算する。
    """
    total = len(state.segments)
    if rendered_count < 0 or rendered_count > total:
        raise ValueError(f"rendered_count が範囲外です: {rendered_count} (区間数 {total})")
    pending = state.segments[rendered_count:][-MAX_DISPLAY_LINES:]
    lines = tuple(format_segment_line(s) for s in pending)
    displayed = min(rendered_count, MAX_DISPLAY_LINES)
    trim_count = max(0, displayed + len(lines) - MAX_DISPLAY_LINES)
    return DisplayDelta(lines=lines, new_count=total, trim_count=trim_count)


def _validate_count(name: str, value) -> None:
    """非負の int(bool を除く)であることを確認する"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} は 0 以上の整数が必要です: {value!r}")


def to_live_result(
    state: LiveViewState,
    dropped_chunks: int,
    complete: bool,
    dropped_segments: int = 0,
) -> LiveResult:
    """現在の状態から LiveResult を作る(ドレイン中断時の部分ノート用)"""
    _validate_count("dropped_chunks", dropped_chunks)
    _validate_count("dropped_segments", dropped_segments)
    return LiveResult(
        segments=state.segments,
        final_summary=state.summary,
        dropped_chunks=dropped_chunks,
        dropped_segments=dropped_segments,
        complete=complete,
    )
