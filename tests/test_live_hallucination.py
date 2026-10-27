"""live.hallucination のテスト"""

import pytest

from live.hallucination import is_hallucination


@pytest.mark.parametrize(
    "text",
    [
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございました。",
        " ご視聴ありがとうございました! ",
        "チャンネル登録よろしくお願いします",
        "チャンネル登録と高評価をお願いします",
        "高評価よろしくお願いします",
        "最後までご視聴いただきありがとうございます",
        "Thank you for watching.",
    ],
)
def test_known_phrases_are_hallucinations(text):
    """Whisper の定型幻覚句は幻覚と判定する"""
    assert is_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "今日の会議では来期の予算について話し合います。",
        "ありがとうございます、助かりました。",
        "登録の手続きは来週までにお願いします。",
        "はい、そうですね。",
        "顧客から高評価をいただきました。",
        "お客様の番号は000000で登録されています。",
    ],
)
def test_ordinary_sentences_are_not_hallucinations(text):
    """通常の発話は幻覚と判定しない(過剰フィルタしない)"""
    assert is_hallucination(text) is False


@pytest.mark.parametrize("text", ["", "   ", "。、"])
def test_empty_text_is_hallucination(text):
    """空・記号のみのテキストは捨てる対象として True"""
    assert is_hallucination(text) is True


def test_high_no_speech_prob_is_hallucination():
    """no_speech_prob が閾値超なら幻覚"""
    assert is_hallucination("今日はいい天気", no_speech_prob=0.9) is True


def test_low_no_speech_prob_is_not_hallucination():
    """no_speech_prob が低ければ幻覚ではない"""
    assert is_hallucination("今日はいい天気", no_speech_prob=0.1) is False


def test_high_no_speech_with_confident_logprob_is_not_hallucination():
    """no_speech が高くても avg_logprob が十分高ければ幻覚としない(Whisper 準拠)"""
    assert is_hallucination("今日はいい天気", no_speech_prob=0.9, avg_logprob=-0.2) is False


def test_high_no_speech_with_low_logprob_is_hallucination():
    """no_speech が高く avg_logprob も低ければ幻覚"""
    assert is_hallucination("今日はいい天気", no_speech_prob=0.9, avg_logprob=-1.5) is True


def test_repeated_sentence_is_hallucination():
    """同一文の反復は幻覚"""
    assert (
        is_hallucination("よろしくお願いします。よろしくお願いします。よろしくお願いします。")
        is True
    )


def test_sentence_repeated_twice_is_not_hallucination():
    """2回程度の繰り返しは通常の発話として扱う"""
    assert is_hallucination("そうですね。そうですね。") is False


def test_repeated_short_pattern_is_hallucination():
    """短いパターンの連続反復は幻覚"""
    assert is_hallucination("あのあのあのあのあのあのあの") is True
