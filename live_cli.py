"""
ライブ文字起こし CLI(`main_cli.py --live`)

録音しながらローカルで逐次文字起こしし、ローカルLLMで要約を更新しつつ表示する。
WAV ファイルが正であり、ライブのテキストは best-effort。終了後に `--file` での
高精度な再文字起こしを案内する。
再文字起こしを有効にしていれば(mlx 使用時のみ)、速報版ノートを保存した後に WAV 全体を
large-v3 で文字起こしし直し、要約も作り直して同じノートを上書きする(Ctrl+C で中断できる)。

停止順序の不変条件: 「WAV確定 → ストリーム停止 → ライブのドレイン」
- record_audio() は内部で WAV を確定(on_saved)してから PortAudio ストリームを止めて戻る
- その後で session.stop() を呼び、ライブ側の残り区間を処理する
- ライブ側がハング・遅延しても WAV は既にディスク上に確定している
"""

import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from config import VoiceNoteConfig
from live import model_fetch

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
    effective_engine_name,
    fallback_model,
)
from live.full_summary import summarize_full
from live.local_llm import LocalLLMClient, probe
from live.note import external_send_warning, render_live_note
from live.refine import REFINE_ENGINE, decide_refine, format_refine_progress, refine_wav
from live.session import LiveCallbacks, LiveSession, LiveSettings
from live.session_models import LiveResult, RefinedTranscript
from live.vad import create_vad, vad_kind
from live.view import LiveView
from note_writer import overwrite_live_note, save_live_note
from recorder import record_audio

console = Console()

SESSION_STOP_TIMEOUT_SEC = 120.0
ERROR_STOP_TIMEOUT_SEC = 2.0
PROBE_TIMEOUT_SEC = 3.0
MLX_INSTALL_HINT = "uv sync --extra live-mlx"
REFINE_ASR_IDLE_TIMEOUT_SEC = 30.0  # 再文字起こし前に asr スレッドの終了を待つ上限
REFINE_SUMMARY_TIMEOUT_SEC = 180.0  # 高精度版からの要約の上限
REFINE_POLL_SEC = 0.25  # 再文字起こし中に Ctrl+C を受けるための join 間隔
REFINE_PROGRESS_PRINT_SEC = 10.0  # 進捗表示の間隔


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


def _settings_or_exit(config: VoiceNoteConfig) -> LiveSettings:
    try:
        return LiveSettings.from_config(config)
    except ValueError as e:
        _fatal(f"ライブ文字起こしの設定が不正です: {e}")
        raise  # _fatal は終了するため到達しない(型チェッカー向け)


def run_live(config: VoiceNoteConfig, device: str | None, dest_dir: Path) -> None:
    """ライブ文字起こしを実行し、録音 WAV とライブノートを保存する"""
    engine, settings = _create_engine(_settings_or_exit(config))
    settings = _prepare_refine_model(settings)
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
    job = _RefineJob(settings, engine, backend, session, audio_file, send_warning)
    refine = _plan_refine(job)
    result = _drain(session, rebuild_summary=not refine)
    if result is None:
        _print_interrupted(audio_file, send_warning)
        return
    note_path = _save_fast_note(config, result, audio_file, send_warning)
    refined = refine and _refine_and_overwrite(job, result, note_path)
    _print_done(audio_file, note_path, send_warning, refined=refined)


def _drain(session: LiveSession, *, rebuild_summary: bool) -> LiveResult | None:
    """残り区間を文字起こしして結果を返す。Ctrl+C なら None"""
    console.print(
        "[cyan]残りの区間を文字起こしし、最終要約を作成しています...（Ctrl+C で中断）[/cyan]"
    )
    try:
        return session.stop(timeout=SESSION_STOP_TIMEOUT_SEC, rebuild_summary=rebuild_summary)
    except KeyboardInterrupt:
        # record_audio は戻る前に SIGINT ハンドラを復元するため、ここでの Ctrl+C は
        # KeyboardInterrupt になる。WAV は確定済みなのでノート生成だけを諦める
        return None


def _save_fast_note(
    config: VoiceNoteConfig, result: LiveResult, audio_file: Path, send_warning: str | None
) -> Path:
    """速報版のライブノートを保存する。失敗は終了(録音は保存済み)"""
    try:
        body = render_live_note(result, audio_file, send_warning=send_warning)
        return save_live_note(Path(config.save_folder), body, audio_file=audio_file)
    except RuntimeError as e:
        _fatal(f"{e}（録音は {audio_file.absolute()} に保存済みです）")
        raise  # _fatal は終了するため到達しない(型チェッカー向け)


# ------------------------------------------------------------ 高精度の再文字起こし


@dataclass(frozen=True)
class _RefineJob:
    """停止後の再文字起こしに必要な、1回の実行の文脈"""

    settings: LiveSettings
    engine: object
    backend: object | None
    session: LiveSession
    audio_file: Path
    send_warning: str | None


def _refine_model_cached(settings: LiveSettings) -> bool:
    """高精度モデルがローカルにあるか(リポジトリ名が解決できなければ False)"""
    try:
        repo = model_fetch.resolve_repo(REFINE_ENGINE, settings.refine_model)
    except ValueError:
        return False
    return repo is None or model_fetch.is_cached(repo)


def _print_fetch_progress(model: str):
    """ダウンロード進捗を間引いて表示する関数を返す"""
    last = {"at": None}

    def on_progress(done: int, total: int) -> None:
        now = time.monotonic()
        finished = total > 0 and done >= total
        if last["at"] is not None and now - last["at"] < REFINE_PROGRESS_PRINT_SEC and not finished:
            return
        last["at"] = now
        percent = f"（{min(100, done * 100 // total)}%）" if total > 0 else ""
        _notify(f"高精度モデル {model} をダウンロードしています... {done // 1_000_000} MB{percent}")

    return on_progress


def _prepare_refine_model(settings: LiveSettings) -> LiveSettings:
    """再文字起こしを予定していれば、録音前に高精度モデルを取得する

    取得できなければ警告し、再文字起こしなし(refine_enabled=False)で続行する。
    """
    if not settings.refine_enabled or settings.engine != REFINE_ENGINE:
        return settings
    if _refine_model_cached(settings):
        return settings
    model = settings.refine_model
    console.print(f"[cyan]高精度モデル {model} を取得します（初回のみ）[/cyan]")
    try:
        repo = model_fetch.resolve_repo(REFINE_ENGINE, model)
        model_fetch.fetch(repo, _print_fetch_progress(model))
    except Exception as e:
        _warn(
            f"{model_fetch.describe_fetch_error(e).message}高精度の再文字起こしなしで続行します。"
        )
        return replace(settings, refine_enabled=False)
    return settings


def _plan_refine(job: _RefineJob) -> bool:
    """停止前に、再文字起こしを行うか決める(行わない理由があれば表示する)"""
    decision = decide_refine(
        job.settings,
        effective_engine_name(job.engine, job.settings),
        job.settings.refine_enabled and _refine_model_cached(job.settings),
    )
    if decision.reason:
        _warn(decision.reason)
    return decision.run


def _progress_printer():
    """再文字起こしの進捗を間引いて表示する関数を返す(完了時は必ず表示)"""
    last = {"at": None}

    def on_progress(done: float, total: float, eta: float | None) -> None:
        now = time.monotonic()
        recent = last["at"] is not None and now - last["at"] < REFINE_PROGRESS_PRINT_SEC
        if recent and done < total:
            return
        last["at"] = now
        _notify(format_refine_progress(done, total, eta))

    return on_progress


def _join_once(worker: threading.Thread) -> None:
    """Ctrl+C(KeyboardInterrupt)を受けられるよう短い間隔で join する"""
    worker.join(REFINE_POLL_SEC)


def _wait_worker(worker: threading.Thread, cancel: threading.Event) -> bool:
    """worker の終了を待つ。Ctrl+C なら cancel して現在の窓の終わりまで待ち、False を返す

    mlx の処理中にプロセスを終えると Metal の後始末で落ちうるため、もう一度 Ctrl+C されるまでは
    窓の区切りで止まるのを待つ。
    """
    try:
        while worker.is_alive():
            _join_once(worker)
        return True
    except KeyboardInterrupt:
        cancel.set()
    console.print(
        "[yellow]高精度の再文字起こしを中断しています...（現在の区間が終わり次第。"
        "もう一度 Ctrl+C で即時終了）[/yellow]"
    )
    try:
        while worker.is_alive():
            _join_once(worker)
    except KeyboardInterrupt:
        pass
    return False


def _run_refine(job: _RefineJob):
    """worker スレッドで refine_wav を実行する。(結果 | None, 中断されたか) を返す"""
    create_engine, _ = _engine_api()
    cancel = threading.Event()
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["result"] = refine_wav(
                job.audio_file,
                make_engine=lambda: create_engine(
                    REFINE_ENGINE, job.settings.refine_model, notify=_notify
                ),
                release_live=getattr(job.engine, "unload", lambda: None),
                language=job.settings.language,
                hint=None,
                cancel=cancel,
                on_progress=_progress_printer(),
            )
        except Exception as e:
            outcome["error"] = e

    worker = threading.Thread(target=target, name="live-refine", daemon=True)
    worker.start()
    if not _wait_worker(worker, cancel):
        return None, True
    if "error" in outcome:
        # 例外のメッセージは文字起こし本文を含みうるため型名だけを出す
        _warn(f"高精度の再文字起こしに失敗しました（{type(outcome['error']).__name__}）")
        return None, False
    return outcome["result"], False


def _refined_summary(job: _RefineJob, refined: RefinedTranscript, fallback: str) -> str:
    """高精度版の全文から要約を作り直す。作れなければ速報版の要約"""
    if job.backend is None or not refined.segments:
        return fallback
    console.print("[cyan]高精度版の全文から要約を作り直しています...[/cyan]")
    summary = summarize_full(
        job.backend,
        tuple(s.text for s in refined.segments),
        None,
        deadline=time.monotonic() + REFINE_SUMMARY_TIMEOUT_SEC,
    )
    return summary or fallback


def _refine_and_overwrite(job: _RefineJob, result: LiveResult, note_path: Path) -> bool:
    """WAV 全体を再文字起こしし、要約を作り直して速報版ノートを上書きする。上書きしたら True

    中断・失敗・時間切れのときは速報版ノートをそのまま残す。
    """
    if not job.session.wait_asr_idle(REFINE_ASR_IDLE_TIMEOUT_SEC):
        _warn("ライブの文字起こしが終わらないため、高精度の再文字起こしをスキップしました")
        return False
    console.print(
        f"[cyan]高精度モデル {job.settings.refine_model} で録音全体を再文字起こししています..."
        "（Ctrl+C で中断。速報版のノートは保存済みです）[/cyan]"
    )
    refine_result, interrupted = _run_refine(job)
    if interrupted:
        _warn("高精度の再文字起こしを中断しました。速報版のノートを残します")
        return False
    if refine_result is None:
        return False  # 失敗(_run_refine が表示済み)
    if not refine_result.complete:
        _warn("高精度の再文字起こしが時間内に終わらなかったため、速報版のままにします")
        return False
    return _overwrite_refined(job, result, note_path, refine_result.segments)


def _overwrite_refined(job: _RefineJob, result: LiveResult, note_path: Path, segments) -> bool:
    refined = RefinedTranscript(segments, job.settings.refine_model)
    try:
        summary = _refined_summary(job, refined, result.final_summary)
        body = render_live_note(
            replace(result, final_summary=summary),
            job.audio_file,
            send_warning=job.send_warning,
            refined=refined,
        )
        overwrite_live_note(note_path, body, refined=True)
    except KeyboardInterrupt:
        _warn("要約の作り直しを中断しました。速報版のノートを残します")
        return False
    except RuntimeError as e:
        _warn(f"高精度版でノートを上書きできませんでした（速報版のノートは残っています）: {e}")
        return False
    return True


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


def _print_done(
    audio_file: Path, note_path: Path, send_warning: str | None, *, refined: bool = False
) -> None:
    footer = (
        "[green]高精度版で上書きしました（速報版の文字起こしも残しています）[/green]"
        if refined
        else _retranscribe_hint(audio_file, send_warning)
    )
    console.print(
        Panel.fit(
            f"[bold green]完了![/bold green]\n\n"
            f"[bold]音声ファイル:[/bold]\n{audio_file.absolute()}\n\n"
            f"[bold]ライブノート:[/bold]\n{note_path.absolute()}\n\n"
            f"{footer}",
            border_style="green",
        )
    )
