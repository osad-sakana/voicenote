"""要約出力が日本語として妥当かを判定する純粋関数群。

ローカルLLM（qwen2.5 等）は入力の文字起こしが崩れていると、指示に反して中国語で
要約を書くことがある。ここでは次のいずれかに当たる出力を「日本語として不正」とみなす。

- 漢字を含むのに、ひらがな・カタカナが全く無い（中国語の文章とみなす）
- 簡体字専用字形の割合（行内の漢字に対する比率）が SIMPLIFIED_LINE_RATIO 以上の行がある
- 簡体字専用字形を 1 文字でも含み、かつひらがな・カタカナを含まない節がある
  （節 = 改行と句読点 CLAUSE_DELIMITERS で区切った単位。ただし数字に挟まれた半角の
  「,」「.」は桁区切り・小数点とみなして区切らない）。節にかなが 1 文字でもあれば
  （日本語文中に引用された中国語の固有名詞など）節単位では許容する

全角カンマ「，」や漢字だけの節（かな無し）は、それだけでは根拠にしない。
「背景，目的，手法」「担当：田中，佐藤」のような正当な日本語と区別できないため。
「的」「是」「了」などの中国語の機能語も、日本語の熟語（具体的・是正・完了）と衝突する
ので判定に使わない。

英数字・固有名詞だけの行（漢字を含まない行）は判定対象にしない。
"""

from __future__ import annotations

import re

# 行内の漢字に占める簡体字専用字形の割合がこれ以上なら、その行は中国語混入とみなす
SIMPLIFIED_LINE_RATIO = 0.2

# 節の区切りとみなす句読点（全角・半角）
CLAUSE_DELIMITERS: frozenset[str] = frozenset("。、，,．.！？!?；;：:")

# 数字に挟まれると桁区切り・小数点になる半角の区切り文字（1,000 / 1.5 / v1.2 は分割しない）
_NUMERIC_SEPARATORS = ",."

_CLAUSE_SPLIT_PATTERN = re.compile(
    "["
    + re.escape("".join(sorted(CLAUSE_DELIMITERS - frozenset(_NUMERIC_SEPARATORS))))
    + "]"
    + f"|(?<!\\d)[{re.escape(_NUMERIC_SEPARATORS)}]"
    + f"|[{re.escape(_NUMERIC_SEPARATORS)}](?!\\d)"
)

# 簡体字専用字形（日本語の新字体と字形が異なるもの）のみの集合。
# 会・来・与・号・区・参・体 などの日本語と同形の字は含めない。
# 誤検出を避けるため、JIS X 0208（Shift_JIS で符号化できる字）に含まれる字は除外している
# （个・无・从・广・况・价・网 など。テストで機械的に検査する）。
SIMPLIFIED_ONLY_CHARS: frozenset[str] = frozenset(
    "过议讲义题进说这们时现对开关问实动为术发经样业种东书车长门间见记让认请谁转须质选际"
    "给级构总产员阳乐红约线终组织绝统页项顺预领频类调该复单济办热压离严积极优众传伤么吗"
    "还话语讨论设计习应电脑处结报务识难简"
    "继续确划适虑态运输标备链节库仓缓测试验证环维护规则围载视图层码执资变换读错误虽两录"
    "启删览户账战灵势边园显张拥归纳师阶历岁举营销钱购买卖费满"
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


def split_clauses(line: str) -> tuple[str, ...]:
    """行を CLAUSE_DELIMITERS で節に分割する（空の節は含めない）。

    数字に挟まれた半角の「,」「.」（1,000 や 1.5）では分割しない。
    """
    return tuple(c for c in _CLAUSE_SPLIT_PATTERN.split(line) if c.strip())


def _has_kana(text: str) -> bool:
    return any(_is_kana(c) for c in text)


def _is_simplified_clause(clause: str) -> bool:
    """簡体字専用字形を含み、かなを含まない節なら True。"""
    return any(c in SIMPLIFIED_ONLY_CHARS for c in clause) and not _has_kana(clause)


def _is_plausible_line(line: str) -> bool:
    """1 行が日本語として妥当なら True（行比率・節の各ルールで判定）。"""
    if simplified_ratio(line) >= SIMPLIFIED_LINE_RATIO:
        return False
    return not any(_is_simplified_clause(c) for c in split_clauses(line))


def is_plausible_japanese(text: str) -> bool:
    """要約本文が日本語として妥当なら True を返す（中国語混入なら False）。"""
    has_han = any(_is_han(c) for c in text)
    if has_han and not _has_kana(text):
        return False
    return all(_is_plausible_line(line) for line in text.splitlines())
