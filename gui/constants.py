"""GUI で使用する定数とユーティリティ。"""

import re

MODE_RECORD_TRANSCRIBE = "録音して文字起こしする"
MODE_RECORD_ONLY = "録音だけする"
MODE_TRANSCRIBE_ONLY = "文字起こしだけする"
MODES = [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY, MODE_TRANSCRIBE_ONLY]

PRIMARY_BUTTON_COLOR = ["#3B8ED0", "#1F6AA5"]

_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001ffff⌀-⟿⤀-⯿​-‏︀-️]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """文字列から絵文字を取り除く (macOS Tk クラッシュ対策)。"""
    return _EMOJI_RE.sub("", text).strip()
