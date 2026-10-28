"""gui.constants のユーティリティのテスト。"""

import pytest

from gui.constants import (
    DEFAULT_MODE,
    MODE_LIVE,
    MODE_RECORD_ONLY,
    MODE_RECORD_TRANSCRIBE,
    MODE_TRANSCRIBE_ONLY,
    MODES,
    format_elapsed,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (5.9, "00:05"),
        (65, "01:05"),
        (3599, "59:59"),
        (3661, "1:01:01"),
        (-3, "00:00"),
    ],
)
def test_format_elapsed(seconds, expected):
    assert format_elapsed(seconds) == expected


def test_modes_keep_existing_order_and_append_live():
    assert MODES == [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY, MODE_TRANSCRIBE_ONLY, MODE_LIVE]


def test_existing_mode_values_unchanged():
    assert MODE_RECORD_TRANSCRIBE == "録音して文字起こしする"
    assert MODE_RECORD_ONLY == "録音だけする"
    assert MODE_TRANSCRIBE_ONLY == "文字起こしだけする"


def test_live_mode_label():
    assert MODE_LIVE == "ライブ"
    assert len(set(MODES)) == len(MODES)


def test_default_mode_is_live_and_selectable():
    # 起動時に選ばれるモードはライブ。選択肢に含まれていること
    assert DEFAULT_MODE == MODE_LIVE
    assert DEFAULT_MODE in MODES
