"""gui/live_height.py のテスト(ライブ窓の高さ見積り。要約欄の最小3行を守れるかの判定)"""

import pytest

from gui.constants import MODE_LIVE
from gui.live_height import (
    HEAVY_EXTRAS,
    LiveExtras,
    estimated_body_height,
    min_window_height,
    summary_shortfall,
)
from gui.mode_logic import (
    LIVE_HEIGHT,
    LIVE_MIN_HEIGHT,
    SCREEN_RESERVED_HEIGHT,
    fit_to_screen,
    live_screen_compact,
    window_spec,
)


def _fitted_height(screen_height: int) -> int:
    return fit_to_screen(window_spec(MODE_LIVE, log_visible=False), screen_height).height


class TestCalibration:
    """実App計測(幅760・圧縮なし)と一致すること"""

    def test_plain_body_at_760(self):
        assert estimated_body_height(760, LiveExtras(), compact=False) == 245

    def test_heavy_body_at_760(self):
        # 案内バナー1件+入力レベル警告+上限案内(3項目・2行)+完了表示
        assert estimated_body_height(760, HEAVY_EXTRAS, compact=False) == 89

    def test_heavy_without_compaction_is_22px_short_at_760(self):
        # 2回目レビューの実測: 要約欄が 22px はみ出し、見える高さは約57px
        assert summary_shortfall(760, HEAVY_EXTRAS, compact=False) == 22

    def test_heavy_needs_782_without_compaction(self):
        assert min_window_height(HEAVY_EXTRAS, compact=False) == 782


class TestSummaryKeepsMinimum:
    def test_13inch_standard_screen_heavy_fits_with_compaction(self):
        # 縦900pt の画面: 窓は 760 に縮み、圧縮して要約欄の3行分を守る(退行防止)
        height = _fitted_height(900)
        assert height == 900 - SCREEN_RESERVED_HEIGHT
        assert live_screen_compact(900) is True
        assert summary_shortfall(height, HEAVY_EXTRAS, compact=True) == 0

    def test_large_screen_heavy_fits_at_default(self):
        assert live_screen_compact(1050) is False
        assert summary_shortfall(_fitted_height(1050), HEAVY_EXTRAS, compact=False) == 0

    def test_min_height_keeps_summary_in_heavy_state(self):
        assert summary_shortfall(LIVE_MIN_HEIGHT, HEAVY_EXTRAS, compact=False) == 0

    def test_three_banners_at_min_height_are_9px_short(self):
        # 大きい画面でも最小の高さ(790)まで縮めてバナー3件が同時に出ると数px割る(文書と一致)
        extras = LiveExtras(banner_lines=3, warning=True, limit_notice=True, result=True)
        assert summary_shortfall(LIVE_MIN_HEIGHT, extras, compact=False) == 9

    def test_three_banners_fit_default_height(self):
        extras = LiveExtras(banner_lines=3, warning=True, limit_notice=True, result=True)
        assert summary_shortfall(LIVE_HEIGHT, extras, compact=False) == 0

    def test_compaction_never_needs_more_height(self):
        for extras in (LiveExtras(), HEAVY_EXTRAS, LiveExtras(banner_lines=3)):
            assert min_window_height(extras, compact=True) <= min_window_height(
                extras, compact=False
            )

    def test_shortfall_zero_when_tall(self):
        assert summary_shortfall(2000, HEAVY_EXTRAS, compact=False) == 0

    def test_shortfall_consistent_with_min_window_height(self):
        need = min_window_height(HEAVY_EXTRAS, compact=True)
        assert summary_shortfall(need, HEAVY_EXTRAS, compact=True) == 0
        assert summary_shortfall(need - 1, HEAVY_EXTRAS, compact=True) == 1


class TestCompactCalibration:
    """実App計測(幅760・compact)と一致すること"""

    def test_heavy_body_at_760(self):
        assert estimated_body_height(760, HEAVY_EXTRAS, compact=True) == 130

    def test_three_banners_body_at_760(self):
        extras = LiveExtras(banner_lines=3, warning=True, limit_notice=True, result=True)
        assert estimated_body_height(760, extras, compact=True) == 113
        assert summary_shortfall(760, extras, compact=True) == 0

    def test_plain_body_at_760(self):
        assert estimated_body_height(760, LiveExtras(), compact=True) == 264

    def test_screen_thresholds(self):
        # 文書(README・CLAUDE.md)の「縦が約900pt未満」の根拠
        three = LiveExtras(banner_lines=3, warning=True, limit_notice=True, result=True)
        assert min_window_height(HEAVY_EXTRAS, compact=True) + SCREEN_RESERVED_HEIGHT == 881
        assert min_window_height(three, compact=True) + SCREEN_RESERVED_HEIGHT == 898
        assert min_window_height(HEAVY_EXTRAS, compact=False) + SCREEN_RESERVED_HEIGHT == 922


class TestCalibratedLayout:
    """見積りは下の配置値で計測・較正した。変えたら実Appで計測し直して live_height を更新する"""

    _MSG = "レイアウトが変わりました。実Appで計測し直して gui/live_height.py を更新してください"

    def test_summary_font_matches_theme(self):
        from gui import theme
        from gui.live_height import SUMMARY_FONT_SIZE

        assert SUMMARY_FONT_SIZE == theme.LIVE_SUMMARY_FONT_SIZE, self._MSG

    def test_panel_gaps(self):
        from gui import live_layout

        normal, compact = live_layout.NORMAL_GAPS, live_layout.COMPACT_GAPS
        assert (normal.section, normal.warning, normal.result) == (8, 4, 6), self._MSG
        assert (compact.section, compact.warning, compact.result) == (4, 2, 2), self._MSG
        assert live_layout.SUMMARY_PAD_TOP == 4, self._MSG

    def test_context_form_compaction(self):
        from gui import live_context_form as form

        assert (form._MEMO_LINES, form._COMPACT_MEMO_LINES) == (3, 2), self._MSG
        assert (form._CLEAR_PAD_TOP, form._COMPACT_CLEAR_PAD_TOP) == (6, 2), self._MSG


class TestInvalid:
    def test_negative_banner_lines(self):
        with pytest.raises(ValueError):
            LiveExtras(banner_lines=-1)

    def test_negative_window_height(self):
        with pytest.raises(ValueError):
            estimated_body_height(-1, LiveExtras(), compact=False)
