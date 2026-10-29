"""live.asr_hint（Whisper 用語ヒント）のテスト。"""

import pytest

from live.asr_hint import (
    HINT_MIN_SEGMENT_SEC,
    MAX_HINT_CHARS,
    MAX_TERM_CHARS,
    build_asr_hint,
    extract_memo_terms,
    is_hint_echo,
)
from live.context import LiveContext


class TestConstants:
    def test_values(self):
        assert MAX_HINT_CHARS == 120
        assert MAX_TERM_CHARS == 20
        assert HINT_MIN_SEGMENT_SEC == 1.0


class TestExtractMemoTerms:
    def test_splits_on_newline_comma_and_space(self):
        memo = "予算、スケジュール,Kubernetes，ArgoCD\nOKR 四半期"
        assert extract_memo_terms(memo) == (
            "予算",
            "スケジュール",
            "Kubernetes",
            "ArgoCD",
            "OKR",
            "四半期",
        )

    def test_nfkc_normalizes(self):
        assert extract_memo_terms("ＡＢＣ社　ｶﾀｶﾅ") == ("ABC社", "カタカナ")

    @pytest.mark.parametrize(
        "line",
        [
            "- 予算",
            "* 予算",
            "• 予算",
            "# 予算",
            "> 予算",
            "・予算",
            "1. 予算",
            "2) 予算",
            "(3) 予算",
        ],
    )
    def test_strips_line_markers(self, line):
        assert extract_memo_terms(line) == ("予算",)

    def test_strips_circled_number_before_nfkc(self):
        # NFKC で「①」は「1」になるため、正規化前に除く
        assert extract_memo_terms("①議題\n② 予算") == ("議題", "予算")

    def test_keeps_terms_starting_with_digit(self):
        assert extract_memo_terms("5G 3D") == ("5G", "3D")

    def test_drops_urls_symbols_and_numbers(self):
        memo = "https://example.com/x www.example.com --- ★★ 2026 正規表現"
        assert extract_memo_terms(memo) == ("正規表現",)

    def test_drops_long_sentences(self):
        sentence = "あ" * (MAX_TERM_CHARS + 1)
        assert extract_memo_terms(f"{sentence}\n短い語") == ("短い語",)

    def test_term_of_max_length_is_kept(self):
        term = "あ" * MAX_TERM_CHARS
        assert extract_memo_terms(term) == (term,)

    def test_strips_trailing_punctuation(self):
        assert extract_memo_terms("議題: 予算。決定!") == ("議題", "予算", "決定")

    def test_dedupes_preserving_order(self):
        assert extract_memo_terms("B a B c a") == ("B", "a", "c")

    @pytest.mark.parametrize("memo", ["", "   ", "\n\n", "- \n* "])
    def test_empty(self, memo):
        assert extract_memo_terms(memo) == ()


class TestBuildAsrHint:
    def test_none_and_empty(self):
        assert build_asr_hint(None) is None
        assert build_asr_hint(LiveContext()) is None
        assert build_asr_hint(LiveContext(memo="https://example.com ---")) is None

    def test_format_terms_then_participants(self):
        ctx = LiveContext(participants=("田中", "佐藤"), memo="予算\nKubernetes")
        assert build_asr_hint(ctx) == "用語: 予算、Kubernetes。参加者: 田中、佐藤。"

    def test_participants_only(self):
        assert build_asr_hint(LiveContext(participants=("田中",))) == "参加者: 田中。"

    def test_terms_only(self):
        assert build_asr_hint(LiveContext(memo="予算")) == "用語: 予算。"

    def test_no_newline(self):
        ctx = LiveContext(participants=("田中",), memo="a\nb\nc")
        assert "\n" not in build_asr_hint(ctx)

    def test_term_equal_to_participant_is_not_duplicated(self):
        ctx = LiveContext(participants=("田中",), memo="田中 予算")
        assert build_asr_hint(ctx) == "用語: 予算。参加者: 田中。"

    def test_long_memo_drops_leading_terms_and_keeps_participants(self):
        terms = [f"用語{i:02d}" for i in range(60)]
        ctx = LiveContext(participants=("田中", "佐藤"), memo="\n".join(terms))
        hint = build_asr_hint(ctx)
        assert len(hint) <= MAX_HINT_CHARS
        assert hint.endswith("参加者: 田中、佐藤。")
        assert "用語59" in hint  # 末尾の用語は残る
        assert "用語00" not in hint  # 先頭から削る

    def test_many_long_participants_respect_cap(self):
        # 36文字×3人で「参加者: 」と区切りを含め116文字。用語の入る余地がない
        names = tuple(f"{i:02d}" + "名" * 34 for i in range(30))
        ctx = LiveContext(participants=names, memo="予算")
        hint = build_asr_hint(ctx)
        assert hint is not None
        assert len(hint) <= MAX_HINT_CHARS
        assert hint.startswith("参加者: ")  # 用語は全て削られる
        assert names[-1] in hint  # 末尾側（Whisper が残す側）の参加者を残す
        assert names[0] not in hint

    def test_short_term_fills_remaining_room_after_participants(self):
        names = tuple(f"{i:02d}" + "名" * 38 for i in range(30))
        hint = build_asr_hint(LiveContext(participants=names, memo="予算"))
        assert len(hint) <= MAX_HINT_CHARS
        assert hint.startswith("用語: 予算。参加者: ")

    @pytest.mark.parametrize("n", range(0, 40, 3))
    def test_cap_is_always_respected(self, n):
        ctx = LiveContext(
            participants=tuple(f"人{i}" * 5 for i in range(n)),
            memo=" ".join(f"語{i}" * 4 for i in range(n * 2)),
        )
        hint = build_asr_hint(ctx)
        assert hint is None or len(hint) <= MAX_HINT_CHARS


HINT = "用語: 予算、Kubernetes、四半期決算。参加者: 田中、佐藤。"


class TestIsHintEcho:
    @pytest.mark.parametrize(
        "text",
        [
            HINT,
            "用語: 予算、Kubernetes",
            "予算 Kubernetes 四半期",
            "参加者 田中 佐藤。",
            "ｋｕｂｅｒｎｅｔｅｓ、四半期",  # 全角でも NFKC 後に一致
            "田中、佐藤",  # 6文字未満だがヒント由来の2語で全体を占める
            "佐藤、予算、田中",  # 並びが違っても、ヒントの語だけでできている
        ],
    )
    def test_echo(self, text):
        assert is_hint_echo(text, HINT) is True

    @pytest.mark.parametrize(
        "text",
        [
            "田中",  # 1語の名前だけの発話は捨てない
            "田中です",
            "田中さん、佐藤さん",  # ヒントの語が80%未満
            "田中、田中",  # 異なる語が2つ以上ない
            "佐藤さん",
            "予算について田中さんから説明してください",
            "Kubernetesの移行は来月から始めます",
            "今日はいい天気ですね",
            "",
        ],
    )
    def test_not_echo(self, text):
        assert is_hint_echo(text, HINT) is False

    @pytest.mark.parametrize("hint", [None, ""])
    def test_no_hint(self, hint):
        assert is_hint_echo("用語: 予算、Kubernetes", hint) is False
