"""
設定管理モジュール
設定ファイルの読み込み、保存、対話的設定を提供
"""

import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def load_config(config_path: Path) -> Optional[dict]:
    """設定ファイルを読み込む"""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]設定ファイルの読み込みエラー: {e}[/red]")
            return None
    return None


def save_config(config_path: Path, config: dict):
    """設定ファイルを保存する"""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        console.print(f"[green]設定を保存しました: {config_path}[/green]")
    except Exception as e:
        console.print(f"[red]設定ファイルの保存エラー: {e}[/red]")
        sys.exit(1)


def configure_interactive() -> dict:
    """対話的に設定を入力する"""
    console.print(Panel.fit(
        "[bold cyan]初回設定[/bold cyan]\n設定項目を入力してください。",
        border_style="cyan"
    ))

    # Obsidian Vaultのパス
    while True:
        vault_path = Prompt.ask("[bold]Obsidian Vaultの絶対パス[/bold]")
        vault_path = Path(vault_path).expanduser().resolve()

        if vault_path.exists() and vault_path.is_dir():
            console.print(f"[green]✓ Vaultパスを確認しました: {vault_path}[/green]")
            break
        else:
            console.print("[red]✗ 指定されたパスが存在しないか、ディレクトリではありません。[/red]")

    # 保存先フォルダ名
    save_folder = Prompt.ask(
        "[bold]保存先フォルダ名[/bold]（Vault内の相対パス）",
        default="recordings"
    )

    # Whisperモデル選択
    console.print("\n[bold]使用するWhisperモデルを選択してください:[/bold]")
    console.print("  1. tiny   (最速・精度低)")
    console.print("  2. base   (高速・精度中)")
    console.print("  3. small  (標準)")
    console.print("  4. medium (精度高・時間かかる)")
    console.print("  5. large-v3 (最高精度・最も時間かかる)")

    model_map = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large-v3"
    }

    while True:
        choice = Prompt.ask("[bold]選択[/bold]", default="3")
        if choice in model_map:
            whisper_model = model_map[choice]
            console.print(f"[green]✓ モデル '{whisper_model}' を選択しました[/green]")
            break
        else:
            console.print("[red]✗ 1-5の数字を入力してください。[/red]")

    config = {
        "vault_path": str(vault_path),
        "save_folder": save_folder,
        "whisper_model": whisper_model
    }

    return config
