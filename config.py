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

from live.local_llm import is_loopback_url

console = Console()

CONFIG_PATH = Path.home() / ".config" / "voicenote" / "config.json"

# ライブ文字起こし（--live）の既定値
DEFAULT_LIVE_ENGINE = "faster-whisper"
DEFAULT_LIVE_WHISPER_MODEL = "small"
# --config で mlx を選んだときだけ提示する既定モデル（M1 実測で small より明確に高精度・
# 区間遅延 p50 約1.5秒で実用範囲）。VoiceNoteConfig / 旧 config.json の既定は small のまま
DEFAULT_LIVE_MLX_WHISPER_MODEL = "large-v3-turbo"
_LIVE_ENGINE_DEFAULT_MODELS = {"mlx": DEFAULT_LIVE_MLX_WHISPER_MODEL}
# 入力が小さい環境(RMS -47dB 程度)でもエネルギーVADより取りこぼしが大幅に少ないため Silero を既定にする
DEFAULT_LIVE_VAD = "silero"
# session_models が config を import するため、循環を避けて選択肢はここで定義する
LIVE_VAD_CHOICES = ("silero", "energy")
DEFAULT_LIVE_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LIVE_LLM_MODEL = "qwen2.5:7b"
DEFAULT_LIVE_SUMMARY_INTERVAL_SEC = 45
LIVE_ENGINE_CHOICES = {"1": "mlx", "2": "faster-whisper"}


class InvalidConfigError(Exception):
    """設定ファイルが存在するが読み込み・パースに失敗した場合に送出する。"""


@dataclass(frozen=True)
class VoiceNoteConfig:
    """アプリケーション設定。全フィールドのデフォルト値をここに集約する。"""

    save_folder: str = ""
    recording_folder: str = ""
    whisper_model: str = "small"
    transcription_mode: str = "local"
    vad_filter: bool = True
    format_mode: str = "rule"
    openai_api_key: str | None = None
    # ライブ文字起こし設定（旧 config.json にキーが無ければ既定値が使われる）
    live_engine: str = DEFAULT_LIVE_ENGINE
    live_whisper_model: str = DEFAULT_LIVE_WHISPER_MODEL
    live_language: str = ""  # 空文字は自動判定
    live_vad: str = DEFAULT_LIVE_VAD
    live_llm_enabled: bool = True
    live_llm_base_url: str = DEFAULT_LIVE_LLM_BASE_URL
    live_llm_model: str = DEFAULT_LIVE_LLM_MODEL
    live_summary_interval_sec: int = DEFAULT_LIVE_SUMMARY_INTERVAL_SEC

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
            api_key_available = resolve_api_key(VoiceNoteConfig(openai_api_key=openai_api_key))
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

    live_overrides = configure_live_interactive()

    return VoiceNoteConfig(
        save_folder=str(save_folder_path),
        whisper_model=whisper_model,
        transcription_mode=transcription_mode,
        vad_filter=vad_filter,
        format_mode=format_mode,
        openai_api_key=openai_api_key,
        **live_overrides,
    )


def configure_live_interactive() -> dict:
    """ライブ文字起こし設定を対話的に入力する（任意）。

    Returns:
        VoiceNoteConfig に渡す live_* フィールドの dict。スキップ時は空 dict（既定値を使う）。
    """
    answer = Prompt.ask(
        "\n[bold]ライブ文字起こし設定を行いますか？[/bold]", choices=["y", "n"], default="n"
    )
    if answer != "y":
        return {}
    engine = _ask_live_engine()
    return {
        "live_engine": engine,
        "live_whisper_model": _ask_live_whisper_model(engine),
        "live_vad": _ask_live_vad(),
        "live_language": _ask_live_language(),
        "live_llm_model": _ask_live_llm_model(),
        "live_llm_base_url": _ask_live_base_url(),
    }


def _ask_live_engine() -> str:
    """ライブ用の音声認識エンジンを選ばせる"""
    console.print("\n[bold]ライブ文字起こしのエンジンを選択してください:[/bold]")
    console.print(
        "  1. mlx            （Apple Silicon 向け・高速 / uv sync --extra live-mlx が必要）"
    )
    console.print("  2. faster-whisper （CPU・どの環境でも動作）")
    while True:
        choice = Prompt.ask("[bold]選択[/bold]", default="2")
        if choice in LIVE_ENGINE_CHOICES:
            engine = LIVE_ENGINE_CHOICES[choice]
            console.print(f"[green]✓ エンジン: {engine}[/green]")
            return engine
        console.print("[red]✗ 1または2を入力してください。[/red]")


def default_live_model_for(engine: str) -> str:
    """エンジンごとの推奨（対話設定の既定）モデル名を返す。未知のエンジンは全体の既定"""
    return _LIVE_ENGINE_DEFAULT_MODELS.get(engine, DEFAULT_LIVE_WHISPER_MODEL)


def _ask_live_whisper_model(engine: str) -> str:
    """ライブ用の Whisper モデルを入力させる。既定値はエンジン別。空欄は既定値"""
    default = default_live_model_for(engine)
    model = Prompt.ask(
        "[bold]ライブ用のWhisperモデル[/bold]（例: small, medium, large-v3-turbo）",
        default=default,
    ).strip()
    chosen = model or default
    console.print(f"[green]✓ モデル: {chosen}[/green]")
    return chosen


def _ask_live_vad() -> str:
    """ライブ用の VAD（音声区間検出）を選ばせる。不正な入力は再入力"""
    console.print("\n[bold]ライブ文字起こしの VAD（音声区間検出）を選択してください:[/bold]")
    console.print("  silero: 入力が小さくても安定(推奨) / energy: 軽量だが入力音量に敏感")
    while True:
        choice = (
            Prompt.ask(f"[bold]VAD ({'/'.join(LIVE_VAD_CHOICES)})[/bold]", default=DEFAULT_LIVE_VAD)
            .strip()
            .lower()
        )
        if choice in LIVE_VAD_CHOICES:
            console.print(f"[green]✓ VAD: {choice}[/green]")
            return choice
        console.print(f"[red]✗ {' または '.join(LIVE_VAD_CHOICES)} を入力してください。[/red]")


def _ask_live_language() -> str:
    """認識言語を入力させる。空または auto は自動判定（空文字）"""
    language = Prompt.ask(
        "[bold]認識言語[/bold]（例: ja, en / 空欄または auto で自動判定）", default=""
    ).strip()
    return "" if language.lower() == "auto" else language


def _ask_live_llm_model() -> str:
    """要約に使うローカルLLMのモデル名を入力させる"""
    return Prompt.ask(
        "[bold]要約に使うローカルLLMモデル[/bold]", default=DEFAULT_LIVE_LLM_MODEL
    ).strip()


def _ask_live_base_url() -> str:
    """ローカルLLMの接続先URLを入力させる。ループバック以外は再入力"""
    while True:
        url = Prompt.ask(
            "[bold]ローカルLLMの接続先URL[/bold]", default=DEFAULT_LIVE_LLM_BASE_URL
        ).strip()
        if is_loopback_url(url):
            console.print(f"[green]✓ 接続先: {url}[/green]")
            return url
        console.print(
            "[red]✗ ループバックアドレス（localhost / 127.0.0.1 / ::1）のみ指定できます。[/red]"
        )
