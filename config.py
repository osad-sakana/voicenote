"""
設定管理モジュール
設定ファイルの読み込み、保存、対話的設定を提供
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

CONFIG_PATH = Path.home() / ".config" / "voicenote" / "config.json"


class InvalidConfigError(Exception):
    """設定ファイルが存在するが読み込み・パースに失敗した場合に送出する。"""


@dataclass(frozen=True)
class VoiceNoteConfig:
    """アプリケーション設定。全フィールドのデフォルト値をここに集約する。"""

    save_folder: str = ""
    whisper_model: str = "small"
    transcription_mode: str = "local"
    vad_filter: bool = True
    format_mode: str = "rule"
    openai_api_key: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceNoteConfig":
        """dict から生成する。旧フォーマット（vault_path）のマイグレーションも行う。"""
        migrated = _migrate_legacy(data)
        known_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in migrated.items() if k in known_keys})

    def to_dict(self) -> dict:
        """JSON保存用の dict に変換する。未設定の openai_api_key は含めない。"""
        data = asdict(self)
        if not data.get("openai_api_key"):
            del data["openai_api_key"]
        return data


def _migrate_legacy(config: dict) -> dict:
    """旧フォーマット（vault_path + save_folder）を新フォーマットに変換する。"""
    if "vault_path" in config and "save_folder" in config:
        vault_path = config["vault_path"]
        old_save_folder = config["save_folder"]
        config = {k: v for k, v in config.items() if k != "vault_path"}
        # 既に絶対パスなら変換不要
        if not Path(old_save_folder).is_absolute():
            config = {
                **config,
                "save_folder": str(Path(vault_path) / old_save_folder),
            }
    return config


def load_config(config_path: Path) -> VoiceNoteConfig | None:
    """設定ファイルを読み込む。存在しない場合はNoneを返す。

    Raises:
        InvalidConfigError: ファイルは存在するが読み込み・パースに失敗した場合。
    """
    if not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        return VoiceNoteConfig.from_dict(data)
    except Exception as e:
        raise InvalidConfigError(f"設定ファイルの読み込みエラー: {e}") from e


def save_config(config_path: Path, config: VoiceNoteConfig):
    """設定ファイルを保存する。失敗した場合はRuntimeErrorを送出。"""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"設定ファイルの保存エラー: {e}") from e


def resolve_api_key(config: VoiceNoteConfig) -> str | None:
    """環境変数を優先し、なければ設定からOpenAI APIキーを取得する。"""
    return os.environ.get("OPENAI_API_KEY") or config.openai_api_key


def configure_interactive() -> VoiceNoteConfig:
    """対話的に設定を入力する（CLI用）"""
    console.print(
        Panel.fit(
            "[bold cyan]初回設定[/bold cyan]\n設定項目を入力してください。",
            border_style="cyan",
        )
    )

    # 保存先フォルダ（絶対パス）
    while True:
        save_folder = Prompt.ask(
            "[bold]保存先フォルダの絶対パス[/bold]（例: /Users/xxx/Obsidian/recordings）"
        )
        save_folder_path = Path(save_folder).expanduser().resolve()
        if save_folder_path.parent.exists():
            console.print(f"[green]✓ 保存先フォルダ: {save_folder_path}[/green]")
            break
        else:
            console.print("[red]✗ 親ディレクトリが存在しません。絶対パスを確認してください。[/red]")

    # 文字起こしモード選択
    console.print("\n[bold]文字起こしモードを選択してください:[/bold]")
    console.print("  1. local  (ローカル実行 - faster-whisper)")
    console.print("  2. openai (OpenAI API - 高速・高精度)")

    while True:
        mode_choice = Prompt.ask("[bold]選択[/bold]", default="1")
        if mode_choice == "1":
            transcription_mode = "local"
            console.print("[green]✓ ローカルモード(faster-whisper)を選択しました[/green]")
            break
        elif mode_choice == "2":
            transcription_mode = "openai"
            console.print("[green]✓ OpenAI APIモードを選択しました[/green]")
            break
        else:
            console.print("[red]✗ 1または2を入力してください。[/red]")

    # Whisperモデル選択（ローカルモード時のみ）
    whisper_model = "small"
    if transcription_mode == "local":
        console.print("\n[bold]使用するWhisperモデルを選択してください:[/bold]")
        console.print("  1. tiny     (最速・精度低)")
        console.print("  2. base     (高速・精度中)")
        console.print("  3. small    (標準)")
        console.print("  4. medium   (精度高・時間かかる)")
        console.print("  5. large-v3 (最高精度・最も時間かかる)")

        model_map = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large-v3"}
        while True:
            choice = Prompt.ask("[bold]選択[/bold]", default="3")
            if choice in model_map:
                whisper_model = model_map[choice]
                console.print(f"[green]✓ モデル '{whisper_model}' を選択しました[/green]")
                break
            else:
                console.print("[red]✗ 1-5の数字を入力してください。[/red]")

    # OpenAI APIキー設定（openaiモード選択時）
    openai_api_key = None
    if transcription_mode == "openai":
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            console.print("[dim]OPENAI_API_KEYが環境変数で設定されています。[/dim]")
            use_env = Prompt.ask(
                "[bold]環境変数のキーを使用しますか？[/bold]",
                choices=["y", "n"],
                default="y",
            )
            if use_env != "y":
                openai_api_key = Prompt.ask("[bold]OpenAI APIキー[/bold]", password=True)
        else:
            openai_api_key = Prompt.ask("[bold]OpenAI APIキー[/bold]", password=True)
            if openai_api_key:
                console.print("[green]✓ APIキーを設定しました[/green]")

    # VADフィルタ設定（ローカルモード時のみ有効）
    vad_filter = True
    if transcription_mode == "local":
        console.print("\n[bold]VAD（音声区間検出）フィルタを有効にしますか？[/bold]")
        console.print(
            "  有効にすると無音・ノイズ区間を除去し、ループ（Hallucination）を抑制します。"
        )
        vad_choice = Prompt.ask("[bold]VADフィルタ[/bold]", choices=["y", "n"], default="y")
        vad_filter = vad_choice == "y"
        console.print(f"[green]✓ VADフィルタ: {'有効' if vad_filter else '無効'}[/green]")

    # 整形モード選択
    console.print("\n[bold]文字起こし結果の整形モードを選択してください:[/bold]")
    console.print("  1. rule  （ルールベース整形 - 句読点補完・フィラー語除去）")
    console.print("  2. llm   （GPT-4o-miniで高品質整形 - OPENAI_API_KEY必要）")
    console.print("  3. none  （整形なし - 生テキスト）")

    format_mode = "rule"
    while True:
        fmt_choice = Prompt.ask("[bold]選択[/bold]", default="1")
        if fmt_choice == "1":
            format_mode = "rule"
            console.print("[green]✓ ルールベース整形を選択しました[/green]")
            break
        elif fmt_choice == "2":
            api_key_available = os.environ.get("OPENAI_API_KEY") or openai_api_key
            if not api_key_available:
                console.print(
                    "[yellow]⚠ OPENAI_API_KEYが設定されていません。ルールベース整形を使用します。[/yellow]"
                )
                format_mode = "rule"
            else:
                format_mode = "llm"
                console.print("[green]✓ LLM整形（GPT-4o-mini）を選択しました[/green]")
            break
        elif fmt_choice == "3":
            format_mode = "none"
            console.print("[green]✓ 整形なしを選択しました[/green]")
            break
        else:
            console.print("[red]✗ 1・2・3のいずれかを入力してください。[/red]")

    return VoiceNoteConfig(
        save_folder=str(save_folder_path),
        whisper_model=whisper_model,
        transcription_mode=transcription_mode,
        vad_filter=vad_filter,
        format_mode=format_mode,
        openai_api_key=openai_api_key,
    )
