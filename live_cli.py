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

import logging
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
from live.refine import (
    REFINE_ASR_IDLE_TIMEOUT_SEC,
    REFINE_ENGINE,
    decide_refine,
    format_refine_progress,
    refine_wav,
    summarize_refined,
)
from live.refine_fetch import BackgroundModelFetch
from live.session import LiveCallbacks, LiveSession, LiveSettings
from live.session_models import LiveResult, RefinedTranscript
from live.vad import create_vad, vad_kind
from live.view import LiveView
from note_writer import overwrite_live_note, save_live_note
from recorder import record_audio

console = Console()
_logger = logging.getLogger("voicenote")

SESSION_STOP_TIMEOUT_SEC = 120.0
ERROR_STOP_TIMEOUT_SEC = 2.0
PROBE_TIMEOUT_SEC = 3.0
MLX_INSTALL_HINT = "uv sync --extra live-mlx"
REFINE_POLL_SEC = 0.25  # 再文字起こし中に Ctrl+C を受けるための join 間隔
REFINE_PROGRESS_PRINT_SEC = 10.0  # 進捗表示の間隔
# 高精度モデルのバックグラウンド取得スレッドの生成(テストで差し替える)
_fetch_thread_factory = threading.Thread


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


def _record(
    session: LiveSession,
    view: LiveView,
    device: str | None,
    dest_dir: Path,
    on_recording=lambda: None,
) -> Path:
    """Rich Live で画面を更新しながら録音する。戻った時点で WAV は確定・ストリームは停止済み

    on_recording は録音が始まった時に呼ぶ(高精度モデルのバックグラウンド取得の開始など)。
    """

    def on_start(device_name: str) -> None:
        console.print(
            Panel.fit(
                f"[bold green]ライブ文字起こしを開始します[/bold green]\n"
                f"[dim]デバイス: {device_name}[/dim]\n"
                f"[yellow]Ctrl+C[/yellow] で録音を終了します",
                border_style="green",
            )
        )
        on_recording()

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
    backend = _prepare_backend(settings)
    fetch = BackgroundModelFetch(thread_factory=lambda **kw: _fetch_thread_factory(**kw))
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
        audio_file = _record(
            session,
            view,
            device,
            dest_dir,
            on_recording=lambda: _start_refine_check(settings, fetch),
        )
    except (ValueError, RuntimeError) as e:
        session.stop(timeout=ERROR_STOP_TIMEOUT_SEC, summary_timeout=0.0)
        _fatal(str(e))
    # ここに到達した時点で「WAV確定 → ストリーム停止」は完了済み。最後にライブ側をドレインする
    send_warning = external_send_warning(config.transcription_mode, config.format_mode)
    result = _drain(session)
    if result is None:
        _print_interrupted(audio_file, send_warning)
        return
    note_path = _save_fast_note(config, result, audio_file, send_warning)
    job = _RefineJob(settings, engine, backend, session, audio_file, send_warning, fetch)
    refined = _refine_phase(job, result, note_path)
    _print_done(audio_file, note_path, send_warning, refined=refined)


def _drain(session: LiveSession) -> LiveResult | None:
    """残り区間を文字起こしして結果を返す

    Ctrl+C なら残りの文字起こし・要約を打ち切り(cancel_summary)、それまでの区間と直前の要約で
    結果を作る(GUI の「中断して保存」の部分ノートと同じ)。その間にもう一度 Ctrl+C されたら
    None(ノートを作らない)。

    GUI の中断と違い、CLI は中断後も速報版ノートを保存して _refine_phase(停止後の高精度
    再文字起こし。有効なら)へ進む。再文字起こしも Ctrl+C で中断できる。
    """
    console.print(
        "[cyan]残りの区間を文字起こしし、最終要約を作成しています..."
        "（Ctrl+C で要約を打ち切って保存）[/cyan]"
    )
    try:
        # 再文字起こしの有無に関係なく全文要約まで作る。要約は stop の中で終わるか打ち切られる
        # (時間切れなら専用の cancel で以降の LLM 呼び出しを止める。送信中の1回だけは LLM 側の
        # timeout まで続きうる)。上限は文字起こしの量に比例する
        return session.stop(timeout=SESSION_STOP_TIMEOUT_SEC)
    except KeyboardInterrupt:
        # record_audio は戻る前に SIGINT ハンドラを復元するため、ここでの Ctrl+C は
        # KeyboardInterrupt になる。WAV は確定済み
        pass
    _warn("文字起こし・要約を中断しました。ここまでの区間と直前の要約でノートを保存します")
    session.cancel_summary()
    try:
        return session.stop(timeout=ERROR_STOP_TIMEOUT_SEC, summary_timeout=0.0)
    except KeyboardInterrupt:
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
    fetch: BackgroundModelFetch


def _refine_model_cached(settings: LiveSettings) -> bool:
    """高精度モデルがローカルにあるか(確認できなければ False。型名だけログに残す)"""
    try:
        repo = model_fetch.resolve_repo(REFINE_ENGINE, settings.refine_model)
        return repo is None or model_fetch.is_cached(repo)
    except Exception as e:
        _logger.warning("高精度モデルを確認できません: %s", type(e).__name__)
        return False


def _start_refine_check(settings: LiveSettings, fetch: BackgroundModelFetch) -> None:
    """録音開始時に呼ぶ。取得要否の確認(ディスク参照)ごと daemon スレッドへ回し、録音を待たせない

    record_audio の on_start から呼ばれるため、ここでは例外を一切外へ出さない
    (出すと録音エラー扱いで終了してしまう)。
    """
    if not settings.refine_enabled or settings.engine != REFINE_ENGINE:
        return
    try:
        _fetch_thread_factory(
            target=_start_refine_fetch,
            args=(settings, fetch),
            daemon=True,
            name="refine-model-check",
        ).start()
    except Exception as e:
        _logger.warning("高精度モデルの確認を開始できません: %s", type(e).__name__)
        _warn(
            f"高精度モデルの確認を開始できません（{type(e).__name__}）。再文字起こしなしで続行します"
        )


def _start_refine_fetch(settings: LiveSettings, fetch: BackgroundModelFetch) -> None:
    """再文字起こしを予定していて高精度モデルが未取得なら、録音中にバックグラウンドで取得する

    録音の開始は待たせない。失敗は警告だけ(型名のみ)で、その回の再文字起こしはしない。
    """
    if not settings.refine_enabled or settings.engine != REFINE_ENGINE:
        return
    model = settings.refine_model
    status = fetch.start(
        model_fetch,
        REFINE_ENGINE,
        model,
        on_failed=lambda msg: _warn(f"{msg}。高精度の再文字起こしなしで続行します"),
    )
    if status == "started":
        _notify(
            f"高精度モデル {model} をバックグラウンドで取得しています"
            "（取得が終わる前に停止した場合、再文字起こしはしません）"
        )


def _plan_refine(job: _RefineJob) -> bool:
    """停止後に、再文字起こしを行うか決める(行わない理由があれば表示する)"""
    settings = job.settings
    fetching = settings.refine_enabled and job.fetch.is_fetching(
        model_fetch, REFINE_ENGINE, settings.refine_model
    )
    decision = decide_refine(
        settings,
        effective_engine_name(job.engine, settings),
        settings.refine_enabled and not fetching and _refine_model_cached(settings),
        fetching=fetching,
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
    if job.backend is not None and refined.segments:
        console.print("[cyan]高精度版の全文から要約を作り直しています...[/cyan]")
    return summarize_refined(
        job.backend, refined.segments, None, fallback, summarize=summarize_full
    )


def _refine_phase(job: _RefineJob, result: LiveResult, note_path: Path) -> bool:
    """停止後の再文字起こし全体。Ctrl+C・想定外の例外は速報版を残して正常に終える"""
    try:
        return _plan_refine(job) and _refine_and_overwrite(job, result, note_path)
    except KeyboardInterrupt:
        _warn("高精度の再文字起こしを中断しました。速報版のノートを残します")
    except Exception as e:
        _logger.error("高精度の再文字起こしで未捕捉の例外: %s", type(e).__name__)
        _warn(f"高精度の再文字起こしに失敗しました（{type(e).__name__}）。速報版のノートを残します")
    return False


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
