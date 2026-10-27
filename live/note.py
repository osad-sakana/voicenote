"""
ライブ文字起こしノートの本文生成(純粋関数)
本文 = 最終要約 + タイムスタンプ付きの区間テキスト
"""

from pathlib import Path

from live.session_models import LiveResult

_NO_SUMMARY = "（要約なし）"
_NO_SEGMENTS = "（文字起こし結果なし）"


def format_timestamp(seconds: float) -> str:
    """録音開始からの秒数を `MM:SS`(1時間以上は `H:MM:SS`)に整形する"""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _quality_note(result: LiveResult) -> str | None:
    """音声の欠落・未処理があれば注意書きを返す"""
    if result.dropped_chunks == 0 and result.dropped_segments == 0 and result.complete:
        return None
    return (
        "> [!warning] 処理が追いつかず、ライブ文字起こしに欠落があります"
        f"（破棄チャンク: {result.dropped_chunks} / 破棄区間: {result.dropped_segments}"
        f" / 未処理あり: {'いいえ' if result.complete else 'はい'}）。"
        "正確な全文は WAV ファイルを `--file` で再文字起こししてください。"
    )


def render_live_note(result: LiveResult, wav_path: Path | None = None) -> str:
    """ライブ文字起こし結果を Markdown 本文(frontmatter なし)に変換する"""
    summary = result.final_summary.strip() or _NO_SUMMARY
    lines = [f"- [{format_timestamp(s.start_sec)}] {s.text}" for s in result.segments]
    transcript = "\n".join(lines) if lines else _NO_SEGMENTS
    sections = [f"## 要約\n\n{summary}", f"## 文字起こし\n\n{transcript}"]
    if wav_path is not None:
        sections.append(f"## 音声ファイル\n\n{wav_path}")
    note = _quality_note(result)
    if note is not None:
        sections.insert(0, note)
    return "\n\n".join(sections) + "\n"
