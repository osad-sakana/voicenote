"""gui.devices モジュールのユニットテスト (Tkinter 非依存の純関数)。"""

from gui.devices import build_device_labels, format_device_label, parse_device_id


class TestFormatDeviceLabel:
    def test_formats_id_and_name(self):
        assert format_device_label(3, "MacBook Air マイク") == "[3] MacBook Air マイク"

    def test_strips_emoji_from_name(self):
        assert format_device_label(1, "🎤 マイク") == "[1] マイク"


class TestBuildDeviceLabels:
    def test_builds_labels_from_devices(self):
        devices = [
            {"id": 0, "name": "内蔵マイク", "input_channels": 1},
            {"id": 2, "name": "USB マイク", "input_channels": 2},
        ]
        assert build_device_labels(devices) == ["[0] 内蔵マイク", "[2] USB マイク"]

    def test_returns_placeholder_when_empty(self):
        assert build_device_labels([]) == ["デバイスなし"]


class TestParseDeviceId:
    def test_parses_single_digit_id(self):
        assert parse_device_id("[3] マイク") == 3

    def test_parses_multi_digit_id(self):
        assert parse_device_id("[12] マイク") == 12

    def test_returns_none_when_not_bracketed(self):
        assert parse_device_id("デバイスなし") is None

    def test_returns_none_when_id_not_numeric(self):
        assert parse_device_id("[x] マイク") is None
