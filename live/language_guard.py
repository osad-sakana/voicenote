"""要約出力が日本語として妥当かを判定する純粋関数群。

ローカルLLM（qwen2.5 等）は入力の文字起こしが崩れていると、指示に反して中国語で
要約を書くことがある。ここでは次のいずれかに当たる出力を「日本語として不正」とみなす。

- 漢字を含むのに、ひらがな・カタカナが全く無い（中国語の文章とみなす）
- 簡体字専用字形の割合（行内の漢字に対する比率）が SIMPLIFIED_LINE_RATIO 以上の行がある

英数字・固有名詞だけの行（漢字を含まない行）は判定対象にしない。
"""

from __future__ import annotations

# 行内の漢字に占める簡体字専用字形の割合がこれ以上なら、その行は中国語混入とみなす
SIMPLIFIED_LINE_RATIO = 0.2

# 簡体字専用字形（日本語の新字体と字形が異なるもの）のみの集合。
# 会・来・与・号・区・参・体 などの日本語と同形の字は含めない。
# 誤検出を避けるため、JIS X 0208（Shift_JIS で符号化できる字）に含まれる字は除外している
# （个・无・从・广・况・价・网 など。テストで機械的に検査する）。
SIMPLIFIED_ONLY_CHARS: frozenset[str] = frozenset(
    "过议讲义题进说这们时现对开关问实动为术发经样业种东书车长门间见记让认请谁转须质选际"
    "给级构总产员阳乐红约线终组织绝统页项顺预领频类调该复单济办热压离严积极优众传伤么吗"
    "还话语讨论设计习应电脑处结报务识难简"
)

_HIRAGANA = ("぀", "ゟ")
# 中黒「・」(U+30FB) は中国語の文中にも現れうるため仮名として数えない
_KATAKANA = ("ァ", "ヺ")
_KATAKANA_MARKS = ("ー", "ヿ")
_KATAKANA_PHONETIC_EXT = ("ㇰ", "ㇿ")
_HALFWIDTH_KATAKANA = ("ｦ", "ﾟ")
_KANA_RANGES = (
    _HIRAGANA,
    _KATAKANA,
    _KATAKANA_MARKS,
    _KATAKANA_PHONETIC_EXT,
    _HALFWIDTH_KATAKANA,
)

# CJK 統合漢字（基本・拡張A）と互換漢字
_HAN_RANGES = (("一", "鿿"), ("㐀", "䶿"), ("豈", "﫿"))


def _in_ranges(char: str, ranges: tuple[tuple[str, str], ...]) -> bool:
    return any(low <= char <= high for low, high in ranges)


def _is_kana(char: str) -> bool:
    return _in_ranges(char, _KANA_RANGES)


def _is_han(char: str) -> bool:
    return _in_ranges(char, _HAN_RANGES)


def simplified_ratio(line: str) -> float:
    """行内の漢字に占める簡体字専用字形の割合を返す。漢字が無ければ 0.0。"""
    han = [c for c in line if _is_han(c)]
    if not han:
        return 0.0
    simplified = sum(1 for c in han if c in SIMPLIFIED_ONLY_CHARS)
    return simplified / len(han)


def is_plausible_japanese(text: str) -> bool:
    """要約本文が日本語として妥当なら True を返す（中国語混入なら False）。"""
    has_han = any(_is_han(c) for c in text)
    if has_han and not any(_is_kana(c) for c in text):
        return False
    return all(simplified_ratio(line) < SIMPLIFIED_LINE_RATIO for line in text.splitlines())
