"""
幻覚判定モジュール
無音・雑音区間で Whisper が出力しがちな定型句や反復を検出する
"""

import re
from collections import Counter

# 部分一致で幻覚とみなす定型句(正規化後の文字列で比較する)
HALLUCINATION_PHRASES: tuple[str, ...] = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご視聴いただきありがとうございます",
    "ご視聴いただきありがとうございました",
    "最後までご視聴",
    "チャンネル登録",
    "高評価お願い",
    "高評価をお願い",
    "高評価よろしく",
    "字幕作成",
    "字幕は視聴者",
    "thankyouforwatching",
    "pleasesubscribe",
)

# 正規化時に除去する文字(空白・句読点・記号)
STRIP_CHARS_PATTERN = re.compile(r"[\s。、，,．.！!？?・…「」『』()（）\"'〜~]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?\n]+")
# 1〜10文字のパターンが6回以上連続し、かつテキストの過半を占めたら反復とみなす
REPEATED_PATTERN = re.compile(r"(.{1,10}?)\1{5,}")
REPEATED_PATTERN_MIN_RATIO = 0.6

SENTENCE_REPEAT_THRESHOLD = 3
NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0


def _normalize(text: str) -> str:
    """空白・句読点を除去し小文字化した比較用文字列を返す"""
    return STRIP_CHARS_PATTERN.sub("", text).lower()


def _contains_known_phrase(normalized: str) -> bool:
    return any(phrase in normalized for phrase in HALLUCINATION_PHRASES)


def _is_no_speech(no_speech_prob: float | None, avg_logprob: float | None) -> bool:
    """Whisper と同じ基準: no_speech が高く、かつ(指定時は)avg_logprob が低い"""
    if no_speech_prob is None or no_speech_prob <= NO_SPEECH_PROB_THRESHOLD:
        return False
    return avg_logprob is None or avg_logprob < AVG_LOGPROB_THRESHOLD


def _has_repeated_sentence(text: str) -> bool:
    sentences = [_normalize(s) for s in SENTENCE_SPLIT_PATTERN.split(text)]
    counts = Counter(s for s in sentences if s)
    return any(c >= SENTENCE_REPEAT_THRESHOLD for c in counts.values())


def _has_repeated_pattern(normalized: str) -> bool:
    """反復部分がテキストの大半を占めるか(「000000」等を含む通常文は除外)"""
    longest = max((len(m.group(0)) for m in REPEATED_PATTERN.finditer(normalized)), default=0)
    return longest >= len(normalized) * REPEATED_PATTERN_MIN_RATIO


def is_hallucination(
    text: str,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
) -> bool:
    """
    文字起こし結果が幻覚(破棄すべきテキスト)かを判定する。
    空文字・記号のみのテキストも破棄対象として True を返す。
    """
    normalized = _normalize(text)
    if not normalized:
        return True
    return (
        _is_no_speech(no_speech_prob, avg_logprob)
        or _contains_known_phrase(normalized)
        or _has_repeated_sentence(text)
        or _has_repeated_pattern(normalized)
    )
