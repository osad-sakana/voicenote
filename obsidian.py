"""
Obsidian保存機能モジュール
文字起こし結果をObsidian Vaultに保存
"""

from datetime import datetime
from pathlib import Path


def save_to_obsidian(save_folder: Path, transcription: str) -> Path:
    """
    文字起こし結果を指定フォルダに保存する

    Args:
        save_folder: 保存先フォルダの絶対パス
        transcription: 文字起こしされたテキスト

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
