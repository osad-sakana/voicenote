"""
ライブ文字起こしの準備処理(CLI と GUI で共有する)

エンジン・VAD・ローカルLLM バックエンドの生成とフォールバック判断をまとめる。
表示(Rich / Tk)には依存せず、通知は呼び出し側が渡すコールバックで行い、
致命的なエラーは LiveSetupError として送出する。

依存(create_engine, create_vad, LocalLLMClient, probe など)はすべて引数で受け取る。
重い依存(mlx 等)を読み込まないこと、呼び出し側でテスト用に差し替えられることが目的。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from config import DEFAULT_LIVE_WHISPER_MODEL
from live.session_models import LiveSettings

FALLBACK_ENGINE = "faster-whisper"
DEFAULT_INSTALL_HINT = "uv sync --extra live-mlx"
VAD_FALLBACK_WARNING = (
    "Silero VADを読み込めなかったため energy VAD にフォールバックしました"
    "（入力が小さいと取りこぼしが増えます）"
)
# mlx 向けには実用的だが、faster-whisper(CPU, int8) では実時間に追いつかないモデル
# (M1 実測2回: large-v3-turbo は区間遅延 p50 5.4〜9.7秒 / RTF 3.0〜4.1。small は p50 2.0〜2.9秒)
CPU_UNSUITABLE_MODELS = frozenset({"large-v3-turbo", "turbo"})

BackendStatus = Literal["ok", "server_down", "model_missing", "invalid_url", "disabled"]


class LiveSetupError(Exception):
    """ライブ文字起こしを開始できない致命的なエラー。メッセージはそのまま利用者に表示できる"""


@dataclass(frozen=True)
class BackendCheck:
    """ローカルLLM の疎通確認結果

    Attributes:
        backend: 要約に使うクライアント。使えない場合は None
        status: ok / server_down / model_missing / invalid_url / disabled
        message: 利用者向けの説明(disabled のときは空文字)
    """

    backend: Any | None
    status: BackendStatus
    message: str


class FallbackOnLoadEngine:
    """load() が EngineUnavailableError を出したら代替エンジンに切り替えるラッパー。

    mlx はモデルのダウンロード失敗などで生成後の load() 時にも利用不可になりうる。
    load() は LiveSession の asr スレッドで呼ばれ、transcribe() も同じスレッドから呼ばれる。
    """

    def __init__(
        self,
        primary,
        make_fallback,
        unavailable_error,
        on_fallback,
        *,
        install_hint: str = DEFAULT_INSTALL_HINT,
    ):
        self._primary = primary
        self._make_fallback = make_fallback
        self._unavailable_error = unavailable_error
        self._on_fallback = on_fallback
        self._install_hint = install_hint
        self._active = primary

    def load(self) -> None:
        """主エンジンを読み込む。利用不可なら代替エンジンを生成して読み込む"""
        try:
            self._primary.load()
            return
        except self._unavailable_error as e:
            self._on_fallback(
                f"{e}。{FALLBACK_ENGINE} に切り替えます（`{self._install_hint}` を確認してください）"
            )
        fallback = self._make_fallback()
        fallback.load()
        self._active = fallback

    def transcribe(self, samples, language):
        """現在有効なエンジンで文字起こしする"""
        return self._active.transcribe(samples, language)


def fallback_model(model: str) -> str:
    """フォールバック先 faster-whisper で使うモデル名を返す

    mlx 向けの HuggingFace リポジトリ指定("/" を含む)と、CPU では遅すぎるモデル
    (CPU_UNSUITABLE_MODELS)は faster-whisper の既定モデルに戻す
    """
    if "/" in model or model in CPU_UNSUITABLE_MODELS:
        return DEFAULT_LIVE_WHISPER_MODEL
    return model


def _engine_error(cause: Exception) -> LiveSetupError:
    return LiveSetupError(f"音声認識エンジンを利用できません: {cause}")


def build_engine(
    settings: LiveSettings,
    *,
    create_engine: Callable[..., Any],
    unavailable_error: type[Exception],
    notify: Callable[[str], None],
    warn: Callable[[str], None],
    install_hint: str,
) -> tuple[Any, LiveSettings]:
    """設定のエンジンを生成する。使えなければ faster-whisper にフォールバックする

    Args:
        settings: ライブ設定
        create_engine: create_engine(name, model, notify=...) 互換の生成関数
        unavailable_error: エンジン利用不可を表す例外型
        notify: エンジンからの進捗通知の表示先
        warn: フォールバック時の警告の表示先
        install_hint: MLX を使えるようにする手順(CLI と配布版で文言が異なる)

    Returns:
        (engine, 実際に使う設定) のタプル

    Raises:
        LiveSetupError: faster-whisper も利用できない場合
    """
    try:
        primary = create_engine(settings.engine, settings.model, notify=notify)
    except unavailable_error as e:
        if settings.engine == FALLBACK_ENGINE:
            raise _engine_error(e) from e
        warn(
            f"{settings.engine} エンジンを利用できません（{e}）。{FALLBACK_ENGINE} で続行します。"
            f"MLX を使うには `{install_hint}` を実行してください。"
        )
        return _create_fallback(settings, create_engine, unavailable_error, notify)
    if settings.engine == FALLBACK_ENGINE:
        return primary, settings
    # 生成できても load() 時に利用不可になりうるため、読み込み時のフォールバックも備える
    model = fallback_model(settings.model)
    engine = FallbackOnLoadEngine(
        primary,
        lambda: create_engine(FALLBACK_ENGINE, model, notify=notify),
        unavailable_error,
        warn,
        install_hint=install_hint,
    )
    return engine, settings


def _create_fallback(
    settings: LiveSettings,
    create_engine: Callable[..., Any],
    unavailable_error: type[Exception],
    notify: Callable[[str], None],
) -> tuple[Any, LiveSettings]:
    """faster-whisper エンジンを生成する。それも使えなければ LiveSetupError を送出する"""
    fallback = replace(settings, engine=FALLBACK_ENGINE, model=fallback_model(settings.model))
    try:
        return create_engine(fallback.engine, fallback.model, notify=notify), fallback
    except unavailable_error as e:
        raise _engine_error(e) from e


def build_vad(
    settings: LiveSettings,
    *,
    create_vad: Callable[[str], Any],
    vad_kind: Callable[[Any], str | None],
    warn: Callable[[str], None],
):
    """設定の VAD を生成する。Silero が使えず energy に落ちた場合は warn で警告する"""
    vad = create_vad(settings.vad)
    if settings.vad != "energy" and vad_kind(vad) == "energy":
        warn(VAD_FALLBACK_WARNING)
    return vad


def check_backend(
    settings: LiveSettings,
    *,
    client_factory: Callable[..., Any],
    probe_fn: Callable[[Any, str], Any],
    probe_timeout: float,
) -> BackendCheck:
    """ローカルLLM の疎通を確認し、結果を返す(表示はしない)

    疎通確認は録音開始前に行うため短いタイムアウトの別クライアントを使う。
    要約は生成に時間がかかるので、本番のクライアントは既定のタイムアウトで生成する。
    """
    if not settings.llm_enabled:
        return BackendCheck(backend=None, status="disabled", message="")
    try:
        probe_client = client_factory(
            settings.llm_base_url, settings.llm_model, timeout=probe_timeout
        )
    except ValueError as e:
        return BackendCheck(backend=None, status="invalid_url", message=str(e))
    result = probe_fn(probe_client, settings.llm_model)
    if result.status != "ok":
        return BackendCheck(backend=None, status=result.status, message=result.message)
    backend = client_factory(settings.llm_base_url, settings.llm_model)
    return BackendCheck(backend=backend, status="ok", message=result.message)
