#!/usr/bin/env python3
"""
録音・文字起こしツール
ローカルで音声を録音し、faster-whisperで文字起こしを行い、Obsidianに保存する。
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "faster-whisper",
#     "sounddevice",
#     "scipy",
#     "numpy",
#     "rich",
# ]
# ///

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from scipy.io import wavfile

console = Console()

# グローバル変数（録音データ格納用）
recording_data = []
is_recording = False
sample_rate = 16000


def signal_handler(sig, frame):
    """Ctrl+Cでの録音停止ハンドラー"""
    global is_recording
    is_recording = False
    console.print("\n[yellow]録音を停止しています...[/yellow]")


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


def audio_callback(indata, frames, time, status):
    """録音コールバック（sounddeviceで使用）"""
    if status:
        console.print(f"[yellow]Warning: {status}[/yellow]")
    if is_recording:
        recording_data.append(indata.copy())


def record_audio() -> np.ndarray:
    """音声を録音する"""
    global is_recording, recording_data

    recording_data = []
    is_recording = True

    # Ctrl+Cのシグナルハンドラーを設定
    signal.signal(signal.SIGINT, signal_handler)

    console.print(Panel.fit(
        "[bold green]録音を開始します[/bold green]\n[yellow]Ctrl+C[/yellow] で録音を終了します",
        border_style="green"
    ))

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            callback=audio_callback
        ):
            while is_recording:
                sd.sleep(100)
    except Exception as e:
        console.print(f"[red]録音エラー: {e}[/red]")
        sys.exit(1)

    console.print("[green]✓ 録音完了[/green]")

    # 録音データを結合
    if len(recording_data) == 0:
        console.print("[red]録音データがありません。[/red]")
        sys.exit(1)

    audio_data = np.concatenate(recording_data, axis=0)
    return audio_data.flatten()


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    """faster-whisperで音声を文字起こしする"""
    console.print(f"\n[cyan]Whisperモデル '{model_name}' をロード中...[/cyan]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # モデルのロード
            task = progress.add_task("モデルをロード中...", total=None)
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            progress.update(task, completed=True)

            # 文字起こし
            task = progress.add_task("文字起こし中...", total=None)
            segments, info = model.transcribe(
                str(audio_path),
                language="ja",
                beam_size=5
            )

            # セグメントを結合
            transcription = ""
            for segment in segments:
                transcription += segment.text

            progress.update(task, completed=True)

        console.print("[green]✓ 文字起こし完了[/green]")
        return transcription.strip()

    except Exception as e:
        console.print(f"[red]文字起こしエラー: {e}[/red]")
        sys.exit(1)


def save_to_obsidian(
    vault_path: Path,
    save_folder: str,
    transcription: str
) -> Path:
    """文字起こし結果をObsidianに保存する"""
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

# 録音文字起こし

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


def main():
    parser = argparse.ArgumentParser(
        description="録音・文字起こしツール"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="設定を再入力する"
    )
    args = parser.parse_args()

    # 設定ファイルパス
    config_path = Path(__file__).parent / "config.json"

    # 設定の読み込みまたは作成
    config = None
    if not args.config:
        config = load_config(config_path)

    if config is None or args.config:
        config = configure_interactive()
        save_config(config_path, config)
    else:
        console.print("[cyan]設定を読み込みました。[/cyan]")

    # 設定の取得
    vault_path = Path(config["vault_path"])
    save_folder = config["save_folder"]
    whisper_model = config["whisper_model"]

    # 録音
    audio_data = record_audio()

    # 一時ファイルに保存（文字起こし用）
    temp_wav = Path(__file__).parent / "temp_recording.wav"
    console.print(f"\n[cyan]音声データを一時保存中...[/cyan]")

    # float32からint16に変換
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(temp_wav, sample_rate, audio_int16)

    # 文字起こし
    transcription = transcribe_audio(temp_wav, whisper_model)

    # 一時ファイル削除
    temp_wav.unlink()

    # Obsidianに保存
    console.print(f"\n[cyan]Obsidianに保存中...[/cyan]")
    saved_path = save_to_obsidian(vault_path, save_folder, transcription)

    # 完了メッセージ
    console.print(Panel.fit(
        f"[bold green]完了![/bold green]\n\n"
        f"[bold]保存先:[/bold]\n{saved_path.absolute()}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
