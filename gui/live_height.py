"""ライブ窓の高さ見積り(Tk 非依存の純関数)。要約欄の最小3行分を守れるかを判定する

Tk・customtkinter を読み込まない。数値はすべて実App計測(macOS・幅760・CustomTkinter 倍率1)で較正した px。
gui/live_layout.py・gui/live_context_form.py の余白や行数を変えたら、計測し直して更新する
(tests/test_gui_live_height.py が 760/790/800 の重い状態の退行を防ぐ)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .live_view_logic import summary_min_height

# theme.LIVE_SUMMARY_FONT_SIZE と同じ値(theme は customtkinter を読み込むので参照しない。
# 一致はテストで確認する)
SUMMARY_FONT_SIZE = 16

# 窓の高さ - 要約・文字起こし欄(body)の高さ。案内・警告・完了表示が無い時
_CHROME = 515
# 小さい画面(compact)ではメモ欄2行・クリア上の余白・各間隔を詰めて 19px 減る
_COMPACT_CHROME = 496
# body の中で常に出す部分: 見出し行(28) + 要約欄の上の余白(4)
_BODY_FIXED = 28 + 4
# 案内バナー: ラベルは最小 28px、1行約 15px。カード内の上下余白 12px
_BANNER_LABEL_MIN = 28
_BANNER_LINE = 15
_BANNER_CARD_PAD = 12


@dataclass(frozen=True)
class _Gaps:
    banner: int
    warning: int
    notice: int  # 上限案内(3項目すべて)の高さ+上の余白
    result: int  # 完了表示の高さ+上の余白


_NORMAL = _Gaps(banner=8, warning=20 + 4, notice=30 + 2, result=46 + 6)
# compact は上限案内を1行に省略する(gui/live_context_logic.limit_notice)
_COMPACT = _Gaps(banner=4, warning=20 + 2, notice=18 + 2, result=46 + 2)


@dataclass(frozen=True)
class LiveExtras:
    """body の外に出る案内・警告・完了表示"""

    banner_lines: int = 0  # 案内バナーの行数(バナー1件で1行)
    warning: bool = False  # 入力レベル警告などの警告行
    limit_notice: bool = False  # 参加者・メモの上限案内(3項目すべての最悪ケース)
    result: bool = False  # 完了表示(高精度で文字起こしボタンつき)

    def __post_init__(self) -> None:
        if self.banner_lines < 0:
            raise ValueError(f"banner_lines は0以上が必要です: {self.banner_lines}")


# 重い状態: 案内バナー1件+入力レベル警告+上限案内(3項目)+完了表示
HEAVY_EXTRAS = LiveExtras(banner_lines=1, warning=True, limit_notice=True, result=True)


def _banner_height(lines: int, gap: int) -> int:
    if lines == 0:
        return 0
    return max(_BANNER_LABEL_MIN, lines * _BANNER_LINE) + _BANNER_CARD_PAD + gap


def _extras_height(extras: LiveExtras, gaps: _Gaps) -> int:
    return (
        _banner_height(extras.banner_lines, gaps.banner)
        + (gaps.warning if extras.warning else 0)
        + (gaps.notice if extras.limit_notice else 0)
        + (gaps.result if extras.result else 0)
    )


def estimated_body_height(window_height: int, extras: LiveExtras, *, compact: bool) -> int:
    """窓の高さから、要約・文字起こし欄(body)に割り当てられる高さを見積もる"""
    if window_height < 0:
        raise ValueError(f"window_height は0以上が必要です: {window_height}")
    return window_height - _outside_body(extras, compact=compact)


def _outside_body(extras: LiveExtras, *, compact: bool) -> int:
    """body の外で使う高さ(窓の枠・操作部・案内・警告・完了表示)"""
    chrome = _COMPACT_CHROME if compact else _CHROME
    return chrome + _extras_height(extras, _COMPACT if compact else _NORMAL)


def _required_body() -> int:
    """文字起こしを畳んだ状態で、要約欄の最小3行分を出すのに要る body の高さ"""
    return _BODY_FIXED + summary_min_height(SUMMARY_FONT_SIZE)


def summary_shortfall(window_height: int, extras: LiveExtras, *, compact: bool) -> int:
    """要約欄の最小3行分に足りない高さ(px)。足りていれば 0"""
    body = estimated_body_height(window_height, extras, compact=compact)
    return max(0, _required_body() - body)


def min_window_height(extras: LiveExtras, *, compact: bool) -> int:
    """要約欄の最小3行分を保てる最小の窓の高さ"""
    return _outside_body(extras, compact=compact) + _required_body()
