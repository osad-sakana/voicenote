"""
ライブ文字起こしノートの本文生成(純粋関数)
本文 = 最終要約 + 参加者・メモ(入力があれば) + タイムスタンプ付きの区間テキスト
"""

from __future__ import annotations

from pathlib import Path

from live.context import MAX_MEMO_CHARS, LiveContext
from live.session_models import LiveResult, LiveSegment, RefinedTranscript

_NO_SUMMARY = "（要約なし）"
_NO_SEGMENTS = "（文字起こし結果なし）"
_EXTERNAL_SEND_WARNING = (
    "この実行は音声/文字起こし本文を OpenAI に送信します。"
    "ローカルだけで処理するには --config で local / rule を選んでください"
)


def external_send_warning(transcription_mode: str, format_mode: str) -> str | None:
    """`--file` 再文字起こしが OpenAI へ送信する設定なら警告文を返す。ローカル設定なら None"""
    if transcription_mode == "openai" or format_mode == "llm":
        return _EXTERNAL_SEND_WARNING
    return None


def format_timestamp(seconds: float) -> str:
    """録音開始からの秒数を `MM:SS`(1時間以上は `H:MM:SS`)に整形する"""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _quality_note(result: LiveResult, send_warning: str | None) -> str | None:
    """音声の欠落・未処理があれば注意書きを返す(再文字起こしの外部送信警告を併記)"""
    if result.dropped_chunks == 0 and result.dropped_segments == 0 and result.complete:
        return None
    note = (
        "> [!warning] 処理が追いつかず、ライブ文字起こしに欠落があります"
        f"（破棄チャンク: {result.dropped_chunks} / 破棄区間: {result.dropped_segments}"
        f" / 未処理あり: {'いいえ' if result.complete else 'はい'}）。"
        "正確な全文は WAV ファイルを `--file` で再文字起こししてください。"
    )
    if send_warning is None:
        return note
    return f"{note}\n> {send_warning}"


def _context_sections(context: LiveContext | None) -> list[str]:
    """参加者・メモの節(入力が空の項目は出さない)"""
    if context is None:
        return []
    sections = []
    if context.participants:
        names = "\n".join(f"- {name}" for name in context.participants)
        sections.append(f"## 参加者\n\n{names}")
    if context.memo:
        notice = (
            f"> [!note] メモが長いため、先頭 {MAX_MEMO_CHARS} 文字のみ記録しています\n\n"
            if context.memo_truncated
            else ""
        )
        sections.append(f"## メモ\n\n{notice}{context.memo}")
    return sections


def _transcript_lines(segments: tuple[LiveSegment, ...]) -> str:
    lines = [f"- [{format_timestamp(s.start_sec)}] {s.text}" for s in segments]
    return "\n".join(lines) if lines else _NO_SEGMENTS


def _transcript_sections(result: LiveResult, refined: RefinedTranscript | None) -> list[str]:
    """文字起こしの節。高精度版があれば先に置き、ライブ版は速報として後ろに残す"""
    live = _transcript_lines(result.segments)
    if refined is None:
        return [f"## 文字起こし\n\n{live}"]
    return [
        f"## 文字起こし（高精度: {refined.model}）\n\n{_transcript_lines(refined.segments)}",
        f"## ライブ文字起こし（速報）\n\n{live}",
    ]


def render_live_note(
    result: LiveResult,
    wav_path: Path | None = None,
    send_warning: str | None = None,
    context: LiveContext | None = None,
    refined: RefinedTranscript | None = None,
) -> str:
    """ライブ文字起こし結果を Markdown 本文(frontmatter なし)に変換する

    Args:
        result: ライブ文字起こし結果
        wav_path: 録音 WAV のパス(指定時は本文に記載する)
        send_warning: `--file` 再文字起こしが外部送信する場合の警告(external_send_warning の戻り値)
        context: 参加者・メモ。空でない項目だけ「## 参加者」「## メモ」節として要約の後に出す
            (None・空なら従来と同一の出力)
        refined: 停止後の高精度再文字起こしの結果。あれば「## 文字起こし（高精度: モデル）」の後に
            「## ライブ文字起こし（速報）」を出す。WAV 全体から作るため、ライブの欠落の注意書きは省く
    """
    summary = result.final_summary.strip() or _NO_SUMMARY
    sections = [
        f"## 要約\n\n{summary}",
        *_context_sections(context),
        *_transcript_sections(result, refined),
    ]
    if wav_path is not None:
        sections.append(f"## 音声ファイル\n\n{wav_path}")
    note = None if refined is not None else _quality_note(result, send_warning)
    if note is not None:
        sections.insert(0, note)
    return "\n\n".join(sections) + "\n"
