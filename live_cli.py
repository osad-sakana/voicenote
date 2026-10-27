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
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from config import DEFAULT_LIVE_WHISPER_MODEL, VoiceNoteConfig
from live.local_llm import LocalLLMClient, probe
from live.note import external_send_warning, render_live_note
from live.session import LiveCallbacks, LiveSession, LiveSettings
from live.vad import create_vad, vad_kind
from live.view import LiveView
from note_writer import save_live_note
from recorder import record_audio

console = Console()

FALLBACK_ENGINE = "faster-whisper"
SESSION_STOP_TIMEOUT_SEC = 120.0
ERROR_STOP_TIMEOUT_SEC = 2.0
PROBE_TIMEOUT_SEC = 3.0
MLX_INSTALL_HINT = "uv sync --extra live-mlx"
VAD_FALLBACK_WARNING = (
    "Silero VADを読み込めなかったため energy VAD にフォールバックしました"
    "（入力が小さいと取りこぼしが増えます）"
)
# mlx 向けには実用的だが、faster-whisper(CPU, int8) では実時間に追いつかないモデル
# (M1 実測2回: large-v3-turbo は区間遅延 p50 5.4〜9.7秒 / RTF 3.0〜4.1。small は p50 2.0〜2.9秒)
CPU_UNSUITABLE_MODELS = frozenset({"large-v3-turbo", "turbo"})


def _engine_api():
    """live.engines を遅延 import する(mlx 等の重い依存を必要時まで読み込まない)"""
    from live.engines import EngineUnavailableError, create_engine

    return create_engine, EngineUnavailableError


def _fatal(message: str) -> None:
    console.print(f"[red]エラー: {message}[/red]")
    sys.exit(1)


def _notify(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


class FallbackOnLoadEngine:
    """load() が EngineUnavailableError を出したら代替エンジンに切り替えるラッパー。

    mlx はモデルのダウンロード失敗などで生成後の load() 時にも利用不可になりうる。
    load() は LiveSession の asr スレッドで呼ばれ、transcribe() も同じスレッドから呼ばれる。
    """

    def __init__(self, primary, make_fallback, unavailable_error, on_fallback):
        self._primary = primary
        self._make_fallback = make_fallback
        self._unavailable_error = unavailable_error
        self._on_fallback = on_fallback
        self._active = primary

    def load(self) -> None:
        try:
            self._primary.load()
            return
        except self._unavailable_error as e:
            self._on_fallback(
                f"{e}。{FALLBACK_ENGINE} に切り替えます（`{MLX_INSTALL_HINT}` を確認してください）"
            )
        fallback = self._make_fallback()
        fallback.load()
        self._active = fallback

    def transcribe(self, samples, language):
        return self._active.transcribe(samples, language)


def _warn(message: str) -> None:
    console.print(f"[yellow]⚠ {message}[/yellow]")


def _create_engine(settings: LiveSettings):
    """設定のエンジンを生成する。使えなければ faster-whisper にフォールバックする

    Returns:
        (engine, 実際に使う設定) のタプル
    """
    create_engine, unavailable_error = _engine_api()
    try:
        primary = create_engine(settings.engine, settings.model, notify=_notify)
    except unavailable_error as e:
        if settings.engine == FALLBACK_ENGINE:
            _fatal(f"音声認識エンジンを利用できません: {e}")
        _warn(
            f"{settings.engine} エンジンを利用できません（{e}）。{FALLBACK_ENGINE} で続行します。"
            f"MLX を使うには `{MLX_INSTALL_HINT}` を実行してください。"
        )
        return _create_fallback(create_engine, unavailable_error, settings)
    if settings.engine == FALLBACK_ENGINE:
        return primary, settings
    # 生成できても load() 時に利用不可になりうるため、読み込み時のフォールバックも備える
    fallback_model = _fallback_model(settings.model)
    engine = FallbackOnLoadEngine(
        primary,
        lambda: create_engine(FALLBACK_ENGINE, fallback_model, notify=_notify),
        unavailable_error,
        _warn,
    )
    return engine, settings


def _fallback_model(model: str) -> str:
    """フォールバック先 faster-whisper で使うモデル名を返す

    mlx 向けの HuggingFace リポジトリ指定("/" を含む)と、CPU では遅すぎるモデル
    (CPU_UNSUITABLE_MODELS)は faster-whisper の既定モデルに戻す
    """
    if "/" in model or model in CPU_UNSUITABLE_MODELS:
        return DEFAULT_LIVE_WHISPER_MODEL
    return model


def _create_fallback(create_engine, unavailable_error, settings: LiveSettings):
    """faster-whisper エンジンを生成する。それも使えなければ致命的エラーで終了する"""
    fallback = replace(settings, engine=FALLBACK_ENGINE, model=_fallback_model(settings.model))
    try:
        return create_engine(fallback.engine, fallback.model, notify=_notify), fallback
    except unavailable_error as e:
        _fatal(f"音声認識エンジンを利用できません: {e}")
        raise  # _fatal は終了するため到達しない(型チェッカー向け)


def _create_vad(settings: LiveSettings):
    """設定の VAD を生成する。Silero が使えず energy に落ちた場合は画面に警告する"""
    vad = create_vad(settings.vad)
    if settings.vad != "energy" and vad_kind(vad) == "energy":
        _warn(VAD_FALLBACK_WARNING)
    return vad


def _prepare_backend(settings: LiveSettings):
    """ローカルLLMの疎通を確認し、使えるならバックエンドを返す。使えなければ None"""
    if not settings.llm_enabled:
        return None
    try:
        # 疎通確認は録音開始前に行うため短いタイムアウトの別クライアントを使う。
        # 要約は生成に時間がかかるので、本番のクライアントは既定のタイムアウトにする
        probe_client = LocalLLMClient(
            settings.llm_base_url, settings.llm_model, timeout=PROBE_TIMEOUT_SEC
        )
    except ValueError as e:
        console.print(f"[yellow]⚠ {e}。要約なしで続行します。[/yellow]")
        return None
    result = probe(probe_client, settings.llm_model)
    if result.status != "ok":
        console.print(f"[yellow]⚠ {result.message} 要約なしで続行します。[/yellow]")
        return None
    console.print(f"[green]✓ {result.message}[/green]")
    return LocalLLMClient(settings.llm_base_url, settings.llm_model)


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
