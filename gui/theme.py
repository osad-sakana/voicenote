"""GUI のデザイントークン (色・角丸・フォント・余白)。

色は (ライト, ダーク) のタプルで持ち、CustomTkinter の外観モードに追従させる。
macOS はシステムカラー、Windows は Fluent 系のアクセント色に寄せてOSらしさを出す。
"""

import sys

import customtkinter as ctk

_IS_MAC = sys.platform == "darwin"
_IS_WINDOWS = sys.platform == "win32"

# ──────────────── 色 ────────────────

ACCENT = ("#007AFF", "#0A84FF") if _IS_MAC else ("#0067C0", "#4CC2FF")
ACCENT_HOVER = ("#0062CC", "#409CFF") if _IS_MAC else ("#005A9E", "#6CCBFF")
RECORDING = ("#FF3B30", "#FF453A")
RECORDING_HOVER = ("#D93025", "#FF6961")
SUCCESS = ("#248A3D", "#30D158")

WINDOW_BG = ("#F5F5F7", "#1E1E1E")
CARD_BG = ("#FFFFFF", "#2C2C2E")
BORDER = ("#D1D1D6", "#3A3A3C")
TEXT = ("#1D1D1F", "#F5F5F7")
TEXT_SECONDARY = ("#6E6E73", "#98989D")
SECONDARY_BTN = ("#E5E5EA", "#3A3A3C")
SECONDARY_BTN_HOVER = ("#D1D1D6", "#48484A")

# ボタンの見た目 (`widget.configure(**STYLE)` でまとめて切り替える)
PRIMARY_STYLE = {"fg_color": ACCENT, "hover_color": ACCENT_HOVER, "text_color": "white"}
DANGER_STYLE = {"fg_color": RECORDING, "hover_color": RECORDING_HOVER, "text_color": "white"}
SECONDARY_STYLE = {
    "fg_color": SECONDARY_BTN,
    "hover_color": SECONDARY_BTN_HOVER,
    "text_color": TEXT,
}

# ──────────────── 形・余白 ────────────────

CARD_RADIUS = 12
CONTROL_RADIUS = 8
SCREEN_PAD = 20
CARD_PAD = 14

# ──────────────── フォント ────────────────

if _IS_MAC:
    FONT_FAMILY: str | None = ".AppleSystemUIFont"
elif _IS_WINDOWS:
    FONT_FAMILY = "Segoe UI"
else:
    FONT_FAMILY = None

# ライブ画面: 要約を主役として大きめに、速報の文字起こしは副として小さめに出す
LIVE_SUMMARY_FONT_SIZE = 16
LIVE_TRANSCRIPT_FONT_SIZE = 12


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    """OS 標準のシステムフォントを返す。"""
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)


def apply_global_theme() -> None:
    """外観モードとデフォルトテーマを設定する。ウィンドウ生成前に呼ぶ。"""
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
