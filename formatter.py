"""
文字起こしテキスト整形モジュール
ルールベース整形とLLM（GPT-4o-mini）による整形を提供する
"""

import os
import re
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# 日本語フィラー語パターン（単独出現かつ文脈に依存しない語）
_FILLER_PATTERNS = [
    r"(?<![^\s。！？])えーと(?=[^\s。！？]|\s|$)",
    r"(?<![^\s。！？])えっと(?=[^\s。！？]|\s|$)",
    r"(?<![^\s。！？])あー(?=[^\s。！？]|\s|$)",
    r"(?<![^\s。！？])あのー?(?=[^\s。！？]|\s|$)",
    r"(?<![^\s。！？])うーん(?=[^\s。！？]|\s|$)",
    r"(?<![^\s。！？])まあ(?=\s|$|、|。)",
    r"(?<![^\s。！？])なんか(?=\s|$|、|。)",
]


def _apply_rule_based_format(text: str) -> str:
    """
    ルールベースで文字起こしテキストを整形する。

    処理内容:
    1. 連続スペースを1つに正規化
    2. 句点・感嘆符・疑問符の後に改行を挿入
    3. フィラー語（えーと、あー等）を除去
    4. 連続する同一フレーズを圧縮
    5. 連続改行を正規化

    Args:
        text: 整形対象のテキスト

    Returns:
        整形後のテキスト
    """
    if not text:
        return text

    # 1. 連続スペースを1つに正規化
    result = re.sub(r" {2,}", " ", text)

    # 2. 句点・感嘆符・疑問符の後に改行を挿入（既に改行がある場合はスキップ）
    result = re.sub(r"([。！？])\s*(?!\n)", r"\1\n", result)

    # 3. フィラー語を除去
    for pattern in _FILLER_PATTERNS:
        result = re.sub(pattern, "", result)

    # 4. 連続する同一フレーズの圧縮（3回以上の繰り返しを1回に）
    result = re.sub(r"(.{2,}?)\1{2,}", r"\1", result)

    # 5. 連続改行を最大2つに正規化
    result = re.sub(r"\n{3,}", "\n\n", result)

    # 先頭・末尾の余分な空白・改行を除去
    return result.strip()


def _apply_llm_format(text: str, api_key: str) -> str:
    """
    OpenAI GPT-4o-miniで文字起こしテキストを整形する。

    Args:
        text: 整形対象のテキスト（ルールベース整形済みを想定）
        api_key: OpenAI APIキー

    Returns:
        整形後のテキスト。エラー時は入力テキストをそのまま返す。
    """
    from openai import OpenAI

    system_prompt = (
        "あなたは音声文字起こしのテキスト整形の専門家です。\n"
        "以下のルールに従って、入力テキストを整形してください:\n\n"
        "1. 句読点（、。！？）を適切に補完し、読みやすい日本語文章にする\n"
        "2. 文の区切りを明確にし、段落を適切に分ける\n"
        "3. フィラー語（えーと、あー、えっと等）が残っていれば除去する\n"
        "4. 内容の追加・削除・要約は一切行わない（整形のみ）\n"
        "5. 後続のLLMがこのテキストを分析・要約する場合も想定して、"
        "構造が明確で論理の流れが追いやすい形に整形する\n\n"
        "整形後のテキストのみを出力してください。説明文は不要です。"
    )

    # 長いテキストは段落単位でチャンク分割して処理
    paragraphs = text.split("\n\n")
    if len(text) > 3000 and len(paragraphs) > 1:
        formatted_parts = []
        for para in paragraphs:
            if para.strip():
                formatted_parts.append(_format_chunk_with_llm(para, api_key, system_prompt))
        return "\n\n".join(formatted_parts)

    return _format_chunk_with_llm(text, api_key, system_prompt)


def _format_chunk_with_llm(text: str, api_key: str, system_prompt: str) -> str:
    """1チャンクをLLMで整形する。エラー時は元テキストを返す。"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        result = response.choices[0].message.content
        return result.strip() if result else text
    except Exception as e:
        console.print(f"[yellow]⚠ LLM整形でエラーが発生しました（ルールベース結果を使用）: {e}[/yellow]")
        return text


def format_transcription(text: str, config: dict) -> str:
    """
    設定に基づいて文字起こしテキストを整形する。

    Args:
        text: 整形対象のテキスト
        config: 設定辞書（format_mode キーを参照）

    Returns:
        整形後のテキスト
    """
    format_mode = config.get("format_mode", "none")

    if format_mode == "none" or not text:
        return text

    if format_mode == "rule":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("テキストを整形中...", total=None)
            result = _apply_rule_based_format(text)
        console.print("[green]✓ テキスト整形完了（ルールベース）[/green]")
        return result

    if format_mode == "llm":
        api_key = _resolve_api_key(config)
        if not api_key:
            console.print("[yellow]⚠ OPENAI_API_KEYが設定されていません。ルールベース整形を使用します。[/yellow]")
            return _apply_rule_based_format(text)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("LLMでテキストを整形中...", total=None)
            # ルールベースで前処理してからLLMに渡す（トークン節約）
            preprocessed = _apply_rule_based_format(text)
            result = _apply_llm_format(preprocessed, api_key)
        console.print("[green]✓ テキスト整形完了（LLM）[/green]")
        return result

    return text


def _resolve_api_key(config: dict) -> Optional[str]:
    """環境変数または設定からOpenAI APIキーを取得する。"""
    return os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
