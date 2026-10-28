"""gui/mode_logic.py のテスト(モード切替で決まるウィンドウ・ボタン・排他)"""

import pytest

from gui.constants import MODE_LIVE, MODE_RECORD_ONLY, MODE_RECORD_TRANSCRIBE, MODE_TRANSCRIBE_ONLY
from gui.mode_logic import (
    COMPACT_HEIGHT,
    EXPANDED_HEIGHT,
    FIT_FLOOR_HEIGHT,
    LIVE_HEIGHT,
    LIVE_MIN_HEIGHT,
    LIVE_WIDTH,
    NORMAL_WIDTH,
    SCREEN_RESERVED_HEIGHT,
    fit_to_screen,
    normal_exec_spec,
    other_mode_rejection,
    resize_on_log_toggle,
    window_spec,
)

_NORMAL_MODES = [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY, MODE_TRANSCRIBE_ONLY]


class TestWindowSpec:
    @pytest.mark.parametrize("mode", _NORMAL_MODES)
    def test_normal_compact(self, mode):
        spec = window_spec(mode, log_visible=False)
        assert (spec.width, spec.height) == (NORMAL_WIDTH, COMPACT_HEIGHT)
        assert spec.resizable is False

    @pytest.mark.parametrize("mode", _NORMAL_MODES)
    def test_normal_expanded(self, mode):
        spec = window_spec(mode, log_visible=True)
        assert (spec.width, spec.height) == (NORMAL_WIDTH, EXPANDED_HEIGHT)
        assert spec.resizable is False

    @pytest.mark.parametrize("log_visible", [False, True])
    def test_live_resizable(self, log_visible):
        spec = window_spec(MODE_LIVE, log_visible=log_visible)
        assert (spec.width, spec.height) == (LIVE_WIDTH, LIVE_HEIGHT)
        assert spec.resizable is True
        assert 0 < spec.min_width <= LIVE_WIDTH
        assert 0 < spec.min_height <= LIVE_HEIGHT

    def test_geometry_string(self):
        assert window_spec(MODE_LIVE, log_visible=False).geometry == f"{LIVE_WIDTH}x{LIVE_HEIGHT}"

    def test_live_size(self):
        # 幅は従来どおり。既定の高さは13インチ+ドックの画面(縦 約800px)に収める
        assert LIVE_WIDTH == 760
        assert LIVE_MIN_HEIGHT <= LIVE_HEIGHT <= 800

    def test_live_min_height_fits_worst_case(self):
        # 実App計測: バナー+入力レベル警告+上限案内(3項目)+完了表示を全部出し、文字起こしを
        # 自動で折りたたんだ時の要求高さが 782px(要約欄は最小の3行分)。それを下回らない
        assert LIVE_MIN_HEIGHT >= 782

    def test_previous_sizes_kept(self):
        # 従来の固定サイズを壊さない
        assert (NORMAL_WIDTH, COMPACT_HEIGHT, EXPANDED_HEIGHT) == (520, 380, 560)


class TestResizeOnLogToggle:
    @pytest.mark.parametrize("mode", _NORMAL_MODES)
    def test_normal_modes_resize(self, mode):
        assert resize_on_log_toggle(mode) is True

    def test_live_keeps_user_size(self):
        assert resize_on_log_toggle(MODE_LIVE) is False


class TestNormalExecSpec:
    @pytest.mark.parametrize("mode", [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY])
    def test_record_idle(self, mode):
        assert normal_exec_spec(mode, recording=False) == ("実行", False)

    @pytest.mark.parametrize("mode", [MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY])
    def test_record_recording_shows_stop(self, mode):
        assert normal_exec_spec(mode, recording=True) == ("録音停止", True)

    def test_transcribe_only(self):
        assert normal_exec_spec(MODE_TRANSCRIBE_ONLY, recording=False) == (
            "文字起こし開始",
            False,
        )

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            normal_exec_spec("不明", recording=False)

    def test_live_mode_not_handled(self):
        with pytest.raises(ValueError):
            normal_exec_spec(MODE_LIVE, recording=False)


class TestOtherModeRejection:
    def test_allowed_when_live_inactive(self):
        assert other_mode_rejection(live_active=False) is None

    def test_rejected_when_live_active(self):
        rejection = other_mode_rejection(live_active=True)
        assert rejection is not None
        title, message = rejection
        assert "ライブ" in title or "ライブ" in message
        assert message

    def test_rejected_while_live_stream_closing(self):
        rejection = other_mode_rejection(live_active=False, live_stream_closing=True)
        assert rejection is not None
        assert "解放" in rejection[1]

    def test_live_active_takes_priority_over_stream_closing(self):
        rejection = other_mode_rejection(live_active=True, live_stream_closing=True)
        assert rejection == other_mode_rejection(live_active=True)


class TestFitToScreen:
    """画面の縦が小さいとき、ウィンドウの高さを画面に合わせて縮める"""

    def test_keeps_spec_when_screen_is_large(self):
        spec = window_spec(MODE_LIVE, log_visible=False)
        assert fit_to_screen(spec, 1050) is spec

    def test_shrinks_height_and_min_height_on_small_screen(self):
        spec = window_spec(MODE_LIVE, log_visible=False)
        fitted = fit_to_screen(spec, 900)  # 13インチ標準の論理解像度
        available = 900 - SCREEN_RESERVED_HEIGHT
        assert fitted.height == available
        assert fitted.min_height == available
        assert fitted.width == spec.width
        assert fitted.resizable is True

    def test_min_height_not_raised_above_requested(self):
        # 最小の高さは元の最小値を超えない(画面が少し足りないだけの場合)
        spec = window_spec(MODE_LIVE, log_visible=False)
        fitted = fit_to_screen(spec, LIVE_HEIGHT + SCREEN_RESERVED_HEIGHT - 5)
        assert fitted.height == LIVE_HEIGHT - 5
        assert fitted.min_height == min(LIVE_MIN_HEIGHT, LIVE_HEIGHT - 5)

    def test_never_below_floor(self):
        spec = window_spec(MODE_LIVE, log_visible=False)
        assert fit_to_screen(spec, 300).height == FIT_FLOOR_HEIGHT

    @pytest.mark.parametrize("height", [0, -1])
    def test_unknown_screen_height_is_ignored(self, height):
        spec = window_spec(MODE_LIVE, log_visible=False)
        assert fit_to_screen(spec, height) is spec

    @pytest.mark.parametrize("mode", _NORMAL_MODES)
    def test_normal_modes_unchanged_on_normal_screens(self, mode):
        spec = window_spec(mode, log_visible=True)
        assert fit_to_screen(spec, 900) is spec
