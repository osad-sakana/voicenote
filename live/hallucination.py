"""
幻覚判定モジュール
無音・雑音区間で Whisper が出力しがちな定型句や反復を検出する
"""

import re
from collections import Counter

# 幻覚とみなす定型句(正規化後の文字列で比較する)。
# 定型句以外の残りが KNOWN_PHRASE_MAX_EXTRA_CHARS 文字以下(文全体がほぼ定型句)のときだけ幻覚とする
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
# 反復パターン・同一文の反復の両方で「テキストの大半」とみなす比率
REPEATED_PATTERN_MIN_RATIO = 0.6

SENTENCE_REPEAT_THRESHOLD = 3
# 「チャンネル登録よろしくお願いします」の「よろしくお願いします」程度の付け足しまでは許容する
KNOWN_PHRASE_MAX_EXTRA_CHARS = 10
NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0


def normalize_for_compare(text: str) -> str:
    """空白・句読点を除去し小文字化した比較用文字列を返す"""
    return STRIP_CHARS_PATTERN.sub("", text).lower()


# 互換のための旧名(モジュール内の既存呼び出しもこちらを使う)
_normalize = normalize_for_compare


def _phrase_covered_indices(normalized: str) -> frozenset[int]:
    """定型句に一致する文字位置の集合(定型句どうしの重なりも考慮する)"""
    return frozenset(
        i
        for phrase in HALLUCINATION_PHRASES
        for m in re.finditer(re.escape(phrase), normalized)
        for i in range(m.start(), m.end())
    )


def _is_known_phrase(normalized: str) -> bool:
    """定型句を含み、かつ定型句以外の残りがごく短い(文全体がほぼ定型句)か"""
    covered = _phrase_covered_indices(normalized)
    if not covered:
        return False
    return len(normalized) - len(covered) <= KNOWN_PHRASE_MAX_EXTRA_CHARS


def _is_no_speech(no_speech_prob: float | None, avg_logprob: float | None) -> bool:
    """Whisper と同じ基準: no_speech が高く、かつ(指定時は)avg_logprob が低い"""
    if no_speech_prob is None or no_speech_prob <= NO_SPEECH_PROB_THRESHOLD:
        return False
    return avg_logprob is None or avg_logprob < AVG_LOGPROB_THRESHOLD


def _has_repeated_sentence(text: str, normalized: str) -> bool:
    """
    同一文が SENTENCE_REPEAT_THRESHOLD 回以上現れ、かつ反復文がテキストの大半を占めるか。
    「はい。〜。はい。〜。はい。」のように相づちの間に本物の内容がある区間は捨てない。
    """
    sentences = [_normalize(s) for s in SENTENCE_SPLIT_PATTERN.split(text)]
    counts = Counter(s for s in sentences if s)
    repeated_chars = sum(
        len(sentence) * count
        for sentence, count in counts.items()
        if count >= SENTENCE_REPEAT_THRESHOLD
    )
    return repeated_chars > 0 and repeated_chars >= len(normalized) * REPEATED_PATTERN_MIN_RATIO


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
        or _is_known_phrase(normalized)
        or _has_repeated_sentence(text, normalized)
        or _has_repeated_pattern(normalized)
    )
