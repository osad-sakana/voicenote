"""
ライブ文字起こし CLI(`main_cli.py --live`)

録音しながらローカルで逐次文字起こしし、ローカルLLMで要約を更新しつつ表示する。
WAV ファイルが正であり、ライブのテキストは best-effort。終了後に `--file` での
高精度な再文字起こしを案内する。

停止順序の不変条件: 「WAV確定 → ストリーム停止 → ライブのドレイン」
- record_audio() は内部で WAV を確定(on_saved)してから PortAudio ストリームを止めて戻る
- その後で session.stop() を呼び、ライブ側の残り区間を処理する
- ライブ側がハング・遅延しても WAV は既にディスク上に確定している
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from config import VoiceNoteConfig

# FALLBACK_ENGINE などは live_cli からも参照できるよう再エクスポートする
from live.bootstrap import (  # noqa: F401
    CPU_UNSUITABLE_MODELS,
    FALLBACK_ENGINE,
    VAD_FALLBACK_WARNING,
    FallbackOnLoadEngine,
    LiveSetupError,
    build_engine,
    build_vad,
    check_backend,
    fallback_model,
)
from live.local_llm import LocalLLMClient, probe
from live.note import external_send_warning, render_live_note
from live.session import LiveCallbacks, LiveSession, LiveSettings
from live.vad import create_vad, vad_kind
from live.view import LiveView
from note_writer import save_live_note
from recorder import record_audio

console = Console()

SESSION_STOP_TIMEOUT_SEC = 120.0
ERROR_STOP_TIMEOUT_SEC = 2.0
PROBE_TIMEOUT_SEC = 3.0
MLX_INSTALL_HINT = "uv sync --extra live-mlx"


def _engine_api():
    """live.engines を遅延 import する(mlx 等の重い依存を必要時まで読み込まない)"""
    from live.engines import EngineUnavailableError, create_engine

    return create_engine, EngineUnavailableError


def _fatal(message: str) -> None:
    console.print(f"[red]エラー: {message}[/red]")
    sys.exit(1)


def _notify(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def _warn(message: str) -> None:
    console.print(f"[yellow]⚠ {message}[/yellow]")


def _create_engine(settings: LiveSettings):
    """設定のエンジンを生成する。使えなければ faster-whisper にフォールバックする

    依存はテストで差し替えられるよう、呼び出し時に本モジュールの名前から解決して渡す。

    Returns:
        (engine, 実際に使う設定) のタプル
    """
    create_engine, unavailable_error = _engine_api()
    try:
        return build_engine(
            settings,
            create_engine=create_engine,
            unavailable_error=unavailable_error,
            notify=_notify,
            warn=_warn,
            install_hint=MLX_INSTALL_HINT,
        )
    except LiveSetupError as e:
        _fatal(str(e))
        raise  # _fatal は終了するため到達しない(型チェッカー向け)


def _create_vad(settings: LiveSettings):
    """設定の VAD を生成する。Silero が使えず energy に落ちた場合は画面に警告する"""
    return build_vad(settings, create_vad=create_vad, vad_kind=vad_kind, warn=_warn)


def _prepare_backend(settings: LiveSettings):
    """ローカルLLMの疎通を確認し、使えるならバックエンドを返す。使えなければ None"""
    check = check_backend(
        settings,
        client_factory=LocalLLMClient,
        probe_fn=probe,
        probe_timeout=PROBE_TIMEOUT_SEC,
    )
    if check.status == "disabled":
        return None
    if check.status == "invalid_url":
        console.print(f"[yellow]⚠ {check.message}。要約なしで続行します。[/yellow]")
        return None
    if check.status != "ok":
        console.print(f"[yellow]⚠ {check.message} 要約なしで続行します。[/yellow]")
        return None
    console.print(f"[green]✓ {check.message}[/green]")
    return check.backend


def _make_callbacks(view: LiveView) -> LiveCallbacks:
    """セッションの通知を画面状態へ反映するコールバック(描画は Live に任せる)"""

    return LiveCallbacks(
        on_segment=view.add_segment,
        on_summary=view.set_summary,
        on_status=view.set_status,
        on_warning=_warn,
    )


def _record(session: LiveSession, view: LiveView, device: str | None, dest_dir: Path) -> Path:
    """Rich Live で画面を更新しながら録音する。戻った時点で WAV は確定・ストリームは停止済み"""

    def on_start(device_name: str) -> None:
        console.print(
            Panel.fit(
                f"[bold green]ライブ文字起こしを開始します[/bold green]\n"
                f"[dim]デバイス: {device_name}[/dim]\n"
                f"[yellow]Ctrl+C[/yellow] で録音を終了します",
                border_style="green",
            )
        )

    def on_stop() -> None:
        view.set_status("録音を停止しています...")

    def on_saved(path: Path) -> None:
        console.print(f"[green]✓ 録音を保存しました: {path.absolute()}[/green]")

    with Live(get_renderable=view.render, console=console, refresh_per_second=4):
        return record_audio(
            device=device,
            dest_dir=dest_dir,
            on_start=on_start,
            on_stop=on_stop,
            on_saved=on_saved,
            chunk_sink=session.feed,
        )


def run_live(config: VoiceNoteConfig, device: str | None, dest_dir: Path) -> None:
    """ライブ文字起こしを実行し、録音 WAV とライブノートを保存する"""
    try:
        settings = LiveSettings.from_config(config)
    except ValueError as e:
        _fatal(f"ライブ文字起こしの設定が不正です: {e}")
    engine, settings = _create_engine(settings)
    backend = _prepare_backend(settings)
    view = LiveView()
    session = LiveSession(
        settings,
        engine=engine,
        vad=_create_vad(settings),
        backend=backend,
        callbacks=_make_callbacks(view),
    )
    # モデル読み込み中も録音チャンクはキューに溜まるよう、録音より先に開始する
    session.start()
    try:
        audio_file = _record(session, view, device, dest_dir)
    except (ValueError, RuntimeError) as e:
        session.stop(timeout=ERROR_STOP_TIMEOUT_SEC)
        _fatal(str(e))
    # ここに到達した時点で「WAV確定 → ストリーム停止」は完了済み。最後にライブ側をドレインする
    send_warning = external_send_warning(config.transcription_mode, config.format_mode)
    console.print(
        "[cyan]残りの区間を文字起こしし、最終要約を作成しています...（Ctrl+C で中断）[/cyan]"
    )
    try:
        result = session.stop(timeout=SESSION_STOP_TIMEOUT_SEC)
    except KeyboardInterrupt:
        # record_audio は戻る前に SIGINT ハンドラを復元するため、ここでの Ctrl+C は
        # KeyboardInterrupt になる。WAV は確定済みなのでノート生成だけを諦める
        _print_interrupted(audio_file, send_warning)
        return
    try:
        body = render_live_note(result, audio_file, send_warning=send_warning)
        note_path = save_live_note(Path(config.save_folder), body, audio_file=audio_file)
    except RuntimeError as e:
        _fatal(f"{e}（録音は {audio_file.absolute()} に保存済みです）")
    _print_done(audio_file, note_path, send_warning)


def _retranscribe_hint(audio_file: Path, send_warning: str | None) -> str:
    """`--file` での再文字起こし案内。外部送信する設定なら警告を併記する"""
    hint = (
        "[dim]ライブのテキストは速報版です。高精度な全文が必要な場合は次を実行してください:[/dim]\n"
        f'uv run main_cli.py --file "{audio_file.absolute()}"'
    )
    if send_warning is None:
        return hint
    return f"{hint}\n[yellow]⚠ {send_warning}[/yellow]"


def _print_interrupted(audio_file: Path, send_warning: str | None) -> None:
    console.print(
        Panel.fit(
            f"[bold yellow]ライブノート作成を中断しました（WAVは保存済み）[/bold yellow]\n\n"
            f"[bold]音声ファイル:[/bold]\n{audio_file.absolute()}\n\n"
            f"{_retranscribe_hint(audio_file, send_warning)}",
            border_style="yellow",
        )
    )


def _print_done(audio_file: Path, note_path: Path, send_warning: str | None) -> None:
    console.print(
        Panel.fit(
            f"[bold green]完了![/bold green]\n\n"
            f"[bold]音声ファイル:[/bold]\n{audio_file.absolute()}\n\n"
            f"[bold]ライブノート:[/bold]\n{note_path.absolute()}\n\n"
            f"{_retranscribe_hint(audio_file, send_warning)}",
            border_style="green",
        )
    )
