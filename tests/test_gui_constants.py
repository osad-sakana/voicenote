"""gui.constants のユーティリティのテスト。"""

import pytest

from gui.constants import format_elapsed


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (5.9, "00:05"), (65, "01:05"), (3599, "59:59"), (3661, "1:01:01"), (-3, "00:00")],
)
def test_format_elapsed(seconds, expected):
    assert format_elapsed(seconds) == expected
