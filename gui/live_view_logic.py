"""GUI ライブ画面の表示内容を決める純関数(Tk 非依存)

LiveViewState / MeterReading から、ボタン・状態文言・バナー・メーター・区間テキストの
差分を求める。ウィジェットへの反映は gui/live_panel.py が行う。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .constants import strip_emoji
from .live_state import DisplayDelta, LiveBanner, LivePhase, LiveViewState, new_display_lines
from .live_workflow import BANNER_MODEL_DOWNLOAD

LEVEL_FLOOR_DB = -60.0  # メーターの左端(これ以下は 0 として描く)
BACKLOG_WARN_SEC = 5.0  # 文字起こしの遅れがこれ以上なら強調表示する

_PERCENT = re.compile(r"(\d+)%")


class ButtonAction(Enum):
    """ライブ用ボタンを押した時の動作"""

    START = "start"
    STOP = "stop"  # PREPARING は取り消し、RECORDING は停止
    ABORT = "abort"  # ドレインを中断して部分ノートを保存
    NONE = "none"


class ButtonStyle(Enum):
    PRIMARY = "primary"
    DANGER = "danger"
    SECONDARY = "secondary"


@dataclass(frozen=True)
class ButtonSpec:
    text: str
    style: ButtonStyle
    enabled: bool
    action: ButtonAction


_BUTTONS = {
    LivePhase.IDLE: ButtonSpec("ライブ開始", ButtonStyle.PRIMARY, True, ButtonAction.START),
    LivePhase.PREPARING: ButtonSpec("停止", ButtonStyle.DANGER, True, ButtonAction.STOP),
    LivePhase.RECORDING: ButtonSpec("停止", ButtonStyle.DANGER, True, ButtonAction.STOP),
    LivePhase.FINALIZING: ButtonSpec("保存中...", ButtonStyle.SECONDARY, False, ButtonAction.NONE),
    LivePhase.DRAINING: ButtonSpec("中断して保存", ButtonStyle.SECONDARY, True, ButtonAction.ABORT),
    LivePhase.SAVING: ButtonSpec("保存中...", ButtonStyle.SECONDARY, False, ButtonAction.NONE),
}

_PHASE_STATUS = {
    LivePhase.IDLE: "待機中",
    LivePhase.PREPARING: "ライブの準備中...",
    LivePhase.FINALIZING: "録音データを確定しています...",
    LivePhase.DRAINING: "残りの区間を文字起こし・最終要約を作成中...",
    LivePhase.SAVING: "ライブノートを保存しています...",
}


def _require_phase(phase: Any) -> LivePhase:
    if not isinstance(phase, LivePhase):
        raise TypeError(f"phase は LivePhase が必要です: {phase!r}")
    return phase


def button_spec(phase: LivePhase) -> ButtonSpec:
    """段階ごとのボタン(ライブ開始 → 停止 → ドレイン中は中断して保存)"""
    return _BUTTONS[_require_phase(phase)]


def status_text(phase: LivePhase, status: str) -> str:
    """状態カードの文言。録音中だけセッションの状態メッセージを使う"""
    if _require_phase(phase) is LivePhase.RECORDING:
        return status or "ライブ録音中"
    return _PHASE_STATUS[phase]


def controls_locked(phase: LivePhase) -> bool:
    """モード選択・デバイス・設定を操作不可にするか(ライブ処理中は固定する)"""
    return _require_phase(phase) is not LivePhase.IDLE


@dataclass(frozen=True)
class SegmentRender:
    """区間テキストボックスの更新指示。reset なら全消去してから delta を適用する"""

    reset: bool
    delta: DisplayDelta


def plan_segment_render(state: LiveViewState, rendered_count: int) -> SegmentRender:
    """描画済み区間数から差分を求める。新しい実行で区間が減っていたら描き直す"""
    if rendered_count < 0 or rendered_count > len(state.segments):
        return SegmentRender(True, new_display_lines(state, 0))
    return SegmentRender(False, new_display_lines(state, rendered_count))


def download_fraction(text: str) -> float | None:
    """モデル取得バナーの「（NN%）」から進捗 0..1 を取り出す。総量不明なら None"""
    match = _PERCENT.search(text or "")
    if match is None:
        return None
    return min(1.0, int(match.group(1)) / 100)


@dataclass(frozen=True)
class BannerView:
    text: str
    show_progress: bool
    fraction: float | None  # None はプログレスバーを不定モードにする

    @property
    def visible(self) -> bool:
        return bool(self.text)


def banner_view(banners: Iterable[LiveBanner]) -> BannerView:
    """案内バナーをまとめた表示内容(絵文字は macOS Tk 対策で除く)"""
    items = tuple(banners)
    text = "\n".join(t for t in (strip_emoji(b.text) for b in items) if t)
    download = next((b for b in items if b.kind == BANNER_MODEL_DOWNLOAD), None)
    if download is None:
        return BannerView(text, False, None)
    return BannerView(text, True, download_fraction(download.text))


def level_fraction(level_db: float) -> float:
    """入力レベル(dBFS)をメーターの 0..1 に変換する"""
    if math.isnan(level_db):
        return 0.0
    return max(0.0, min(1.0, (level_db - LEVEL_FLOOR_DB) / -LEVEL_FLOOR_DB))


def format_level(level_db: float) -> str:
    return f"{level_db:.0f} dB"


def format_backlog(backlog_sec: float | None) -> str:
    if backlog_sec is None:
        return "遅延 -"
    return f"遅延 {max(0.0, backlog_sec):.1f}秒"


def format_dropped(dropped: int | None) -> str:
    if dropped is None:
        return "破棄 -"
    return f"破棄 {max(0, dropped)}"


@dataclass(frozen=True)
class MeterView:
    fraction: float
    level_text: str
    backlog_text: str
    dropped_text: str
    low_input: bool
    backlog_high: bool


def meter_view(reading) -> MeterView:
    """poll_meter() の結果(録音中以外は None)からメーター表示を作る"""
    if reading is None:
        return MeterView(0.0, "- dB", format_backlog(None), format_dropped(None), False, False)
    return MeterView(
        fraction=level_fraction(reading.level_db),
        level_text=format_level(reading.level_db),
        backlog_text=format_backlog(reading.backlog_sec),
        dropped_text=format_dropped(reading.dropped_chunks),
        low_input=bool(reading.input_level.is_low),
        backlog_high=reading.backlog_sec >= BACKLOG_WARN_SEC,
    )


def new_warning(state: LiveViewState, last_seen: str | None) -> str | None:
    """最新の警告がまだ表示・記録していないものなら返す"""
    if not state.warnings:
        return None
    latest = state.warnings[-1]
    return None if latest == last_seen else latest


def result_text(wav: Path, note: Path | None) -> str:
    """完了時に表示する保存ファイル名(既存の完了表示と同じく名前だけ。フルパスは詳細ログへ)"""
    lines = [f"WAV: {wav.name}"]
    if note is not None:
        lines.append(f"ノート: {note.name}")
    return "\n".join(lines)


TRANSCRIPT_LINES = 3  # 文字起こし(副表示)の高さ。要約に縦幅を回すため数行だけにする
SUMMARY_MIN_LINES = 3  # 要約(主役)はウィンドウが狭くてもこの行数分の高さを保つ
_LINE_SPACING = 1.4  # フォントサイズに対する1行の高さの目安
_TEXTBOX_PAD_PX = 12  # テキストボックス上下の内側余白
_TOGGLE_HIDE = "文字起こしを隠す"
_TOGGLE_SHOW = "文字起こしを表示"
_TOGGLE_SHOW_AUTO = "文字起こしを表示（狭いため自動で非表示）"
_TOGGLE_REFUSED = "狭いため文字起こしを表示できません。ウィンドウを広げてください"
# 表示を断った理由を詳細ログにも残す(本文は含めない)
TRANSCRIPT_REFUSED_LOG = "警告: " + _TOGGLE_REFUSED


def transcript_toggle_text(collapsed: bool, *, auto: bool = False, refused: bool = False) -> str:
    """文字起こしの折りたたみボタンの文言

    auto: 窓が狭いため自動で隠している / refused: 押されたが1行も入らず表示できなかった
    """
    if not collapsed:
        return _TOGGLE_HIDE
    if refused:
        return _TOGGLE_REFUSED
    return _TOGGLE_SHOW_AUTO if auto else _TOGGLE_SHOW


def textbox_height(lines: int, font_size: int) -> int:
    """lines 行を表示するテキストボックスの要求高さ(px)"""
    if lines <= 0 or font_size <= 0:
        raise ValueError(f"lines と font_size は正の値が必要です: {lines}, {font_size}")
    return round(lines * font_size * _LINE_SPACING) + _TEXTBOX_PAD_PX


def summary_min_height(font_size: int) -> int:
    """要約欄が保つ最小の高さ(px)。SUMMARY_MIN_LINES 行分"""
    return textbox_height(SUMMARY_MIN_LINES, font_size)


def is_body_cramped(
    body_height: int, *, fixed_height: int, summary_min: int, transcript_height: int
) -> bool:
    """要約・文字起こし欄に割り当てられた高さでは、文字起こしを出すと要約が最小を下回るか

    body_height: 要約・文字起こし欄全体に割り当てられた高さ(折りたたみ状態に依らない)
    fixed_height: 見出し行など常に出す部分の高さ
    summary_min: 要約欄の最小の高さ(余白込み)
    transcript_height: 文字起こし欄(見出し・余白込み)の要求高さ
    """
    if body_height < 0 or fixed_height < 0 or transcript_height < 0 or summary_min <= 0:
        raise ValueError(
            f"高さが不正です: {body_height}, {fixed_height}, {summary_min}, {transcript_height}"
        )
    return body_height < fixed_height + summary_min + transcript_height


def transcript_min_height(*, section_height: int, box_height: int, font_size: int) -> int:
    """文字起こし欄を見出し+1行分だけ出すのに要る高さ(px。上の余白は含めない)

    section_height / box_height: 文字起こし欄全体・テキストボックスの要求高さ
    """
    if box_height < 0 or section_height < box_height:
        raise ValueError(f"高さが不正です: section={section_height}, box={box_height}")
    return section_height - box_height + textbox_height(1, font_size)


def has_room_for_line(
    body_height: int, *, fixed_height: int, summary_min: int, transcript_min: int
) -> bool:
    """要約の最小高さを保ったまま、文字起こし欄の見出し+1行分が入るか

    transcript_min: transcript_min_height に上の余白を足した高さ
    """
    return not is_body_cramped(
        body_height,
        fixed_height=fixed_height,
        summary_min=summary_min,
        transcript_height=transcript_min,
    )


class TranscriptPreference(Enum):
    """文字起こし欄の表示の希望(折りたたみボタンで切り替える)"""

    AUTO = "auto"  # 既定: 窓が狭い時だけ自動で隠す
    HIDDEN = "hidden"  # 利用者が隠した(実行をまたいで保持する)
    # 利用者が狭い窓で表示を選んだ(要約の最小高さは保ち、文字起こしは下から削られる)。
    # 一時的な希望で、窓が広がるか1行も入らなくなれば AUTO に戻す(settle_preference)
    SHOWN = "shown"


@dataclass(frozen=True)
class TranscriptView:
    collapsed: bool
    auto: bool  # 窓が狭いため自動で隠している
    toggle_text: str


@dataclass(frozen=True)
class TranscriptToggle:
    """折りたたみボタンを押した結果"""

    preference: TranscriptPreference
    refused: bool  # 表示を求められたが、狭くて1行も入らないため表示しなかった


def _require_preference(pref: Any) -> TranscriptPreference:
    if not isinstance(pref, TranscriptPreference):
        raise TypeError(f"TranscriptPreference が必要です: {pref!r}")
    return pref


def transcript_view(
    pref: TranscriptPreference, *, cramped: bool, refused: bool = False
) -> TranscriptView:
    """表示の希望と窓の狭さから、文字起こし欄を隠すか・ボタンの文言を決める

    refused: 直前に表示を断った(隠れている間だけ理由をボタンに出す)
    """
    pref = _require_preference(pref)
    auto = pref is TranscriptPreference.AUTO and cramped
    collapsed = pref is TranscriptPreference.HIDDEN or auto
    text = transcript_toggle_text(collapsed, auto=auto, refused=refused)
    return TranscriptView(collapsed, auto, text)


def toggle_preference(
    pref: TranscriptPreference, *, cramped: bool, room_for_line: bool
) -> TranscriptToggle:
    """折りたたみボタンを押した後の希望

    表示中 → 隠す(HIDDEN)。隠れている時 → 広ければ既定(AUTO)に戻して表示、
    狭くても見出し+1行分が入るなら強制表示(SHOWN)。1行も入らないなら表示せず
    AUTO のまま(HIDDEN からなら AUTO にして、窓が広がれば自動で表示する)、refused にする。
    """
    if not transcript_view(pref, cramped=cramped).collapsed:
        return TranscriptToggle(TranscriptPreference.HIDDEN, False)
    if not cramped:
        return TranscriptToggle(TranscriptPreference.AUTO, False)
    if room_for_line:
        return TranscriptToggle(TranscriptPreference.SHOWN, False)
    return TranscriptToggle(TranscriptPreference.AUTO, True)


def settle_preference(
    pref: TranscriptPreference, *, cramped: bool, room_for_line: bool
) -> TranscriptPreference:
    """窓の大きさが変わった時の希望。SHOWN は狭くなくなるか1行も入らなくなれば AUTO に戻す

    SHOWN が AUTO を恒久的に上書きしないため。HIDDEN・AUTO はそのまま。
    """
    pref = _require_preference(pref)
    if pref is TranscriptPreference.SHOWN and (not cramped or not room_for_line):
        return TranscriptPreference.AUTO
    return pref
