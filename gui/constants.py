"""GUI で使用する定数とユーティリティ。"""

import re

MODE_RECORD_TRANSCRIBE = "録音して文字起こしする"
MODE_RECORD_ONLY = "録音だけする"
MODE_TRANSCRIBE_ONLY = "文字起こしだけする"
MODES = [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY, MODE_TRANSCRIBE_ONLY]

_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001ffff⌀-⟿⤀-⯿​-‏︀-️]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """文字列から絵文字を取り除く (macOS Tk クラッシュ対策)。"""
    return _EMOJI_RE.sub("", text).strip()


def format_elapsed(seconds: float) -> str:
    """経過秒数を `mm:ss` (1時間以上は `h:mm:ss`) に整形する。"""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
