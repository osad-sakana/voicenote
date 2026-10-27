"""
ノート保存モジュール。

文字起こし結果を YAML frontmatter 付きの Markdown ファイルとして指定フォルダに保存する。
frontmatter は Obsidian の規約に準拠しているが、他の Markdown ベースのノートツールでも
そのまま利用できる。
"""

from datetime import datetime
from pathlib import Path


def save_transcript(save_folder: Path, transcription: str, format_mode: str = "none") -> Path:
    """文字起こし結果を `save_folder/YYYY-MM-DD_HHMMSS_raw.md` として保存する。

    Args:
        save_folder: 保存先フォルダの絶対パス
        transcription: 文字起こしされたテキスト
        format_mode: 使用した整形モード（"none"/"rule"/"llm"）

    Returns:
        保存されたファイルのパス

    Raises:
        RuntimeError: 保存失敗時
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = save_folder / f"{timestamp}_raw.md"

    now = datetime.now().isoformat()
    content = f"""---
created: {now}
type: transcription
format_mode: {format_mode}
tags:
  - recording
  - raw
---
{transcription}
"""

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
    except Exception as e:
        raise RuntimeError(f"保存エラー: {e}") from e


def save_live_note(save_folder: Path, body: str) -> Path:
    """ライブ文字起こし結果を `save_folder/YYYY-MM-DD_HHMMSS_live.md` として保存する。

    Args:
        save_folder: 保存先フォルダの絶対パス
        body: ノート本文（frontmatter を除く Markdown）

    Returns:
        保存されたファイルのパス

    Raises:
        RuntimeError: 保存失敗時
    """
    save_folder = Path(save_folder)
    now = datetime.now()
    filepath = save_folder / f"{now.strftime('%Y-%m-%d_%H%M%S')}_live.md"
    content = f"""---
created: {now.isoformat()}
type: live-transcription
tags:
  - recording
  - live
---
{body}
"""
    try:
        save_folder.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return filepath
    except Exception as e:
        raise RuntimeError(f"保存エラー: {e}") from e
