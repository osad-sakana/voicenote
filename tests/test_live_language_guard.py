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


# 実機（qwen2.5:7b）で出力された中国語混入の要約（2例目）。
# 行内の簡体字比率は 0.12〜0.17 で閾値未満、かつ別の行にかながあるため従来は採用されていた。
REAL_BROKEN_SUMMARY_MIXED = (
    "現在の話題: 非エンジニア向けGit入門\n"
    "\n"
    "進捗:\n"
    "- Gitについて简述，指出其为版本管理工具。\n"
    "- 借助GitHub Desktop进行演示，因为其界面友好适合非技术人员使用。通常操作是通过命令行方式进行。"
)

# かなを含む日本語の文脈。節単位の判定だけで弾けることを示すために使う
_JA_CONTEXT = "現在の話題: 非エンジニア向けGit入門\n進捗:\n- Gitについて確認した。"


class TestClauseLevelDetection:
    def test_rejects_real_mixed_summary(self):
        assert is_plausible_japanese(REAL_BROKEN_SUMMARY_MIXED) is False

    @pytest.mark.parametrize("line", REAL_BROKEN_SUMMARY_MIXED.splitlines()[3:])
    def test_rejects_each_broken_line_in_japanese_context(self, line):
        assert is_plausible_japanese(f"{_JA_CONTEXT}\n{line}") is False

    @pytest.mark.parametrize(
        "clause",
        ["指出其为版本管理工具", "借助GitHub Desktop进行演示", "因为其界面友好适合非技术人员使用"],
    )
    def test_rejects_each_broken_clause_in_japanese_context(self, clause):
        """かなは他の行にあり、行比率も閾値未満でも、かな無し＋簡体字の節で弾く。"""
        line = f"- 本日の会議では次の点を確認した。{clause}。以上を共有した"
        assert simplified_ratio(line) < SIMPLIFIED_LINE_RATIO
        assert is_plausible_japanese(f"{_JA_CONTEXT}\n{line}") is False

    def test_accepts_kana_clause_with_single_simplified_char_in_long_line(self):
        """仕様: 節にかながあれば簡体字専用字が混じっても節単位では許容する（行比率ルールのみ適用）。"""
        line = "- 来期予算案の確認と採用計画の見直しを経てプロジェクト进行中の議事録を共有した"
        assert simplified_ratio(line) < SIMPLIFIED_LINE_RATIO
        assert is_plausible_japanese(f"現在の話題: 予算\n{line}") is True

    def test_short_kana_clause_with_simplified_char_still_hits_line_ratio(self):
        """仕様: 「プロジェクト进行中」単独の行は漢字 3 字中 1 字（約 33%）で行比率ルールにより不採用。"""
        line = "- プロジェクト进行中"
        assert simplified_ratio(line) >= SIMPLIFIED_LINE_RATIO
        assert is_plausible_japanese(f"現在の話題: 予算\n{line}") is False

    def test_accepts_kanji_noun_list_clauses_without_kana(self):
        """かな無しの節でも簡体字専用字が無ければ弾かない（名詞の列挙など）。"""
        text = "現在の話題: 定例会議\n進捗:\n- 定例会議、議題整理、予算確認\n- 講義資料を更新した"
        assert is_plausible_japanese(text) is True

    def test_accepts_realistic_japanese_summary(self):
        text = (
            "現在の話題: 新規サービスのランディングページ（LP）改善\n"
            "\n"
            "進捗:\n"
            "- 前回の会議で決めた議題に沿って、講義形式で進捗を共有した。\n"
            "- GPT-4o を使った要約機能を AWS 上で検証中。\n"
            "- LP の CVR は 2.5% から 3.1% に改善。次回は A/B テストを実施する。\n"
            "- 次回定例: 10/30 15:00\n"
            "- 来期予算案、採用計画、評価制度改定方針を経営会議で承認"
        )
        assert is_plausible_japanese(text) is True


class TestFullwidthCommaDetection:
    def test_rejects_comma_line_with_kanji_only_clause(self):
        """「，」は中国語の強い手がかり。かな無しの漢字節があれば簡体字が無くても弾く。"""
        line = "- 本件について確認した，借助工具演示"
        assert is_plausible_japanese(f"{_JA_CONTEXT}\n{line}") is False

    def test_accepts_japanese_sentence_with_fullwidth_comma(self):
        """全ての節にかながある日本語文は「，」を含んでいても採用する。"""
        text = "現在の話題: 予算\n- 予算について，来週までに確認します．担当は経理部門です"
        assert is_plausible_japanese(text) is True

    def test_accepts_many_fullwidth_commas_in_japanese(self):
        """学術・技術文書風の「，．」を多用した日本語は過検出しない（個数だけでは弾かない）。"""
        text = "- まず要件を整理し，次に設計を見直し，さらに実装を進め，最後にテストを行った．"
        assert is_plausible_japanese(text) is True

    def test_accepts_comma_between_alnum_clauses(self):
        """漢字を含まない節（英数字のみ）はかな無しでも弾かない。"""
        assert is_plausible_japanese("- AWS，GCP，Azure を比較した") is True


class TestAddedSimplifiedChars:
    @pytest.mark.parametrize("char", list("继续确划适术员简"))
    def test_review_gaps_are_included(self, char):
        assert char in SIMPLIFIED_ONLY_CHARS

    @pytest.mark.parametrize("char", list("决准境范源写并尽据数从况段档号"))
    def test_jis_x_0208_lookalikes_are_excluded(self, char):
        """JIS X 0208 で符号化できる字は簡体字由来でも含めない。"""
        assert char not in SIMPLIFIED_ONLY_CHARS
