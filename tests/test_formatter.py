"""formatter モジュールのユニットテスト。

LLM 整形 (`_apply_llm_format`) は外部 API 呼び出しのためスコープ外。
ルールベース整形 (`_apply_rule_based_format`) と `format_transcription` の
分岐 (none/rule) のみをテストする。
"""

from config import VoiceNoteConfig
from formatter import _apply_rule_based_format, format_transcription


class TestApplyRuleBasedFormat:
    def test_empty_string_returns_empty(self):
        assert _apply_rule_based_format("") == ""

    def test_collapses_multiple_spaces(self):
        result = _apply_rule_based_format("これは    テストです。")
        assert "   " not in result
        assert "これは テストです。" in result

    def test_inserts_newline_after_japanese_period(self):
        result = _apply_rule_based_format("一文目です。二文目です。")
        assert "一文目です。\n" in result
        assert "二文目です。" in result

    def test_inserts_newline_after_exclamation_and_question(self):
        result = _apply_rule_based_format("驚いた！本当ですか？はい")
        assert "驚いた！\n" in result
        assert "本当ですか？\n" in result

    def test_removes_filler_eeto(self):
        result = _apply_rule_based_format("えーと 今日は晴れです")
        assert "えーと" not in result

    def test_removes_filler_etto(self):
        result = _apply_rule_based_format("えっと 今日は晴れです")
        assert "えっと" not in result

    def test_removes_filler_anoo(self):
        result = _apply_rule_based_format("あのー それで")
        assert "あのー" not in result
        assert "あの" not in result

    def test_does_not_remove_filler_substring_inside_word(self):
        # フィラー除去は前後が空白/句読点/行頭末でないと作用しないこと
        result = _apply_rule_based_format("まあまあです")  # 「まあ」は文末・読点前のみ
        assert "まあまあです" in result

    def test_compresses_repeated_phrases(self):
        result = _apply_rule_based_format("ありがとうありがとうありがとうありがとう")
        # 3回以上の連続は 1 回に圧縮される
        assert result.count("ありがとう") < 4

    def test_normalizes_consecutive_newlines(self):
        result = _apply_rule_based_format("一文目。\n\n\n\n二文目。")
        assert "\n\n\n" not in result

    def test_strips_leading_and_trailing_whitespace(self):
        result = _apply_rule_based_format("   テストです   ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestFormatTranscription:
    def test_returns_text_unchanged_when_mode_none(self):
        text = "えーと これは整形されない。"
        assert format_transcription(text, VoiceNoteConfig(format_mode="none")) == text

    def test_returns_empty_for_empty_input(self):
        assert format_transcription("", VoiceNoteConfig(format_mode="rule")) == ""

    def test_rule_mode_applies_rule_based_format(self):
        result = format_transcription(
            "えーと 一文目です。二文目です。", VoiceNoteConfig(format_mode="rule")
        )
        assert "えーと" not in result
        assert "一文目です。\n" in result

    def test_unknown_mode_returns_input_unchanged(self):
        text = "そのまま"
        assert format_transcription(text, VoiceNoteConfig(format_mode="unknown")) == text

    def test_default_mode_is_rule(self):
        text = "えーと 整形されるはず"
        result = format_transcription(text, VoiceNoteConfig())
        assert "えーと" not in result
