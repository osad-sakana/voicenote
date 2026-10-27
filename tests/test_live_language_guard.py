"""live.language_guard のユニットテスト。"""

import pytest

from live.language_guard import (
    SIMPLIFIED_LINE_RATIO,
    SIMPLIFIED_ONLY_CHARS,
    is_plausible_japanese,
    simplified_ratio,
)

# 実機（qwen2.5:7b）で出力された中国語混入の要約
REAL_BROKEN_SUMMARY = "現在の話題: とても簡単だった\n進発:\n- 过去的会议和讲义非常容易理解"


class TestSimplifiedOnlyChars:
    def test_no_char_exists_in_jis_x_0208(self):
        """日本語の文章で普通に使われる字形（JIS X 0208 収録字）を含めない。"""
        encodable = [c for c in SIMPLIFIED_ONLY_CHARS if _encodes_in_shift_jis(c)]
        assert encodable == []

    @pytest.mark.parametrize("char", list("会来与号区参体的了没着后里学数据"))
    def test_japanese_shared_glyphs_are_excluded(self, char):
        assert char not in SIMPLIFIED_ONLY_CHARS

    @pytest.mark.parametrize("char", list("过议讲义题进说这们时现对"))
    def test_typical_simplified_chars_are_included(self, char):
        assert char in SIMPLIFIED_ONLY_CHARS

    def test_is_immutable(self):
        assert isinstance(SIMPLIFIED_ONLY_CHARS, frozenset)


class TestSimplifiedRatio:
    def test_no_han_is_zero(self):
        assert simplified_ratio("GPT-4o / AWS / API v2") == 0.0

    def test_empty_is_zero(self):
        assert simplified_ratio("") == 0.0

    def test_pure_japanese_is_zero(self):
        assert simplified_ratio("会議と講義の議題を整理した") == 0.0

    def test_ratio_over_han_chars(self):
        # 漢字: 过 去 的 会 议 = 5、簡体字専用: 过 议 = 2
        assert simplified_ratio("过去的会议") == pytest.approx(2 / 5)


class TestIsPlausibleJapanese:
    def test_rejects_real_broken_summary(self):
        assert is_plausible_japanese(REAL_BROKEN_SUMMARY) is False

    def test_rejects_chinese_without_kana(self):
        assert is_plausible_japanese("当前话题: 会议\n进度:\n- 讨论了预算") is False

    def test_rejects_han_only_text_without_kana(self):
        """簡体字専用字形が無くても、漢字だけで仮名が全く無ければ日本語とみなさない。"""
        assert is_plausible_japanese("非常容易理解\n过去会议") is False

    def test_accepts_kanji_dense_japanese(self):
        text = (
            "現在の話題: 定例会議の議題整理\n"
            "進捗:\n"
            "- 前回会議の議事録を確認し、講義資料の改訂方針と進捗管理表の更新期限を決定\n"
            "- 来期予算案は経理部門と調整中"
        )
        assert is_plausible_japanese(text) is True

    def test_accepts_alnum_and_proper_noun_lines(self):
        text = (
            "現在の話題: AWS 移行\n"
            "進捗:\n"
            "- GPT-4o / AWS / API v2\n"
            "- Kubernetes 1.30, Terraform, GitHub Actions"
        )
        assert is_plausible_japanese(text) is True

    def test_accepts_ascii_only_text(self):
        """漢字を含まない英数字のみの出力は中国語ではないので弾かない。"""
        assert is_plausible_japanese("a" * 20) is True

    def test_accepts_empty(self):
        assert is_plausible_japanese("") is True

    def test_single_stray_simplified_char_in_long_line_is_tolerated(self):
        # 漢字 16 字中 1 字（約 6%）は閾値未満
        line = "- 来期予算案の確認と採用計画の見直しと議事進行の改善を行った时"
        assert simplified_ratio(line) < SIMPLIFIED_LINE_RATIO
        assert is_plausible_japanese(f"現在の話題: 予算\n{line}") is True

    def test_line_at_threshold_is_rejected(self):
        # 漢字 5 字中 1 字（20%）は閾値以上
        line = "- 会议日程決"
        assert simplified_ratio(line) >= SIMPLIFIED_LINE_RATIO
        assert is_plausible_japanese(f"現在の話題: 予定\n{line}") is False

    def test_katakana_only_counts_as_kana(self):
        assert is_plausible_japanese("現在ノ話題: 予算") is True


def _encodes_in_shift_jis(char: str) -> bool:
    try:
        char.encode("shift_jis")
    except UnicodeEncodeError:
        return False
    return True
