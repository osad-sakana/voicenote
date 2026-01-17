"""
Obsidian保存機能モジュール
文字起こし結果をObsidian Vaultに保存
"""

import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()


def save_to_obsidian(vault_path: Path, save_folder: str, transcription: str) -> Path:
    """
    文字起こし結果をObsidianに保存する

    Args:
        vault_path: Obsidian Vaultのパス
        save_folder: 保存先フォルダ名（Vault内の相対パス）
        transcription: 文字起こしされたテキスト

    Returns:
        保存されたファイルのパス
    """
    # 保存先ディレクトリの作成
    save_dir = vault_path / save_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名生成
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{timestamp}_raw.md"
    filepath = save_dir / filename

    # フロントマターとコンテンツ
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
        console.print(f"[green]✓ 保存完了[/green]")
        return filepath
    except Exception as e:
        console.print(f"[red]保存エラー: {e}[/red]")
        sys.exit(1)
