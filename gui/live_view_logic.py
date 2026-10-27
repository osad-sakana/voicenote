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
