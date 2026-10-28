"""設定ダイアログ「ライブ」タブのロジック（Tk 非依存の純関数）。

UI から分離して単体テストできるようにしている。入力値は `LiveFormValues` に
文字列のまま集め、`validate_live_form` で VoiceNoteConfig の live_* フィールドへの
上書き dict か、フィールド名付きの `FormError` に変換する。
"""

from __future__ import annotations

import importlib.util
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace

from config import DEFAULT_LIVE_REFINE_MODEL, VoiceNoteConfig, default_live_model_for
from live.bootstrap import CPU_UNSUITABLE_MODELS
from live.engines import ENGINE_FASTER_WHISPER, ENGINE_MLX, MLX_INSTALL_HINT, MLX_MODEL_REPOS
from live.local_llm import is_loopback_url
from live.session_models import LiveSettings

LANGUAGE_AUTO = "自動"
LANGUAGE_CHOICES = (LANGUAGE_AUTO, "ja", "en")
LOOPBACK_ONLY_MESSAGE = "ローカルのアドレス（localhost / 127.0.0.1）のみ指定できます"
REFINE_ONLY_MLX_NOTE = (
    "mlx 使用時のみ有効。停止後に録音と同程度の待ち時間と、モデル分のメモリが必要です"
)
PRIVACY_NOTE = "音声と文字起こしはこのPCの外に送信されません（要約はlocalhostのLLMのみ）"

# 表示順（CLI の --config と同じく mlx を先に出す）
ENGINE_ORDER = (ENGINE_MLX, ENGINE_FASTER_WHISPER)
# faster-whisper(CPU, int8) で実用的な候補。CPU 不向きモデルは候補に出さない（手入力は可）
FASTER_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
MLX_PACKAGE = "mlx_whisper"

# Whisper の言語コード（ISO 639-1 / 一部 3文字: yue, haw など）の簡易検証
_LANGUAGE_RE = re.compile(r"[a-z]{2,3}")
# int() は "+45" / "4_5" / 全角数字も受け付けるため ASCII 数字だけに限定する
_POSITIVE_INT_RE = re.compile(r"[0-9]+")

_FIELD_LABELS = {
    "engine": "エンジン",
    "model": "Whisperモデル",
    "vad": "VAD",
    "language": "言語",
    "llm_enabled": "要約（ローカルLLM）",
    "llm_model": "LLMモデル",
    "base_url": "LLMの接続先URL",
    "summary_interval": "要約間隔（秒）",
    "refine_enabled": "停止後の高精度再文字起こし",
    "refine_model": "高精度モデル",
    "settings": "ライブ設定",
}


@dataclass(frozen=True)
class LiveFormValues:
    """ライブタブの入力値。テキスト入力は検証前の文字列のまま保持する"""

    engine: str
    model: str
    vad: str
    language: str
    llm_enabled: bool
    llm_model: str
    base_url: str
    summary_interval: str
    refine_enabled: bool = False
    refine_model: str = DEFAULT_LIVE_REFINE_MODEL


@dataclass(frozen=True)
class FormError:
    """入力エラー。field は LiveFormValues のフィールド名（全体検証の失敗は "settings"）"""

    field: str
    message: str


@dataclass(frozen=True)
class EngineChoices:
    """エンジン選択肢と、利用できないエンジン・その理由の注記"""

    values: tuple[str, ...]
    disabled: frozenset[str]
    note: str | None

    def is_enabled(self, engine: str) -> bool:
        """そのエンジンを選択できるか"""
        return engine not in self.disabled


def field_label(field: str) -> str:
    """エラー表示用のフィールド名（日本語）を返す。未知の名前はそのまま返す"""
    return _FIELD_LABELS.get(field, field)


def form_values_from_config(config: VoiceNoteConfig) -> LiveFormValues:
    """設定からフォームの初期値を作る。手編集された不正値でも例外にしない"""
    language = str(config.live_language or "").strip()
    return LiveFormValues(
        engine=str(config.live_engine),
        model=str(config.live_whisper_model),
        vad=str(config.live_vad),
        language=language or LANGUAGE_AUTO,
        llm_enabled=bool(config.live_llm_enabled),
        llm_model=str(config.live_llm_model),
        base_url=str(config.live_llm_base_url),
        summary_interval=str(config.live_summary_interval_sec),
        refine_enabled=bool(config.live_refine_enabled),
        refine_model=str(config.live_refine_model),
    )


def _parse_language(raw: str) -> str | FormError:
    """「自動」/空欄/auto は ""（自動判定）、それ以外は小文字の言語コードにする"""
    text = raw.strip()
    if text in ("", LANGUAGE_AUTO) or text.lower() == "auto":
        return ""
    code = text.lower()
    if not _LANGUAGE_RE.fullmatch(code):
        return FormError("language", "「自動」または言語コード（例: ja, en）を入力してください")
    return code


def _parse_interval(raw: str) -> int | FormError:
    """要約間隔（秒）を正の整数に変換する"""
    text = raw.strip()
    if not _POSITIVE_INT_RE.fullmatch(text) or int(text) <= 0:
        return FormError("summary_interval", "1以上の整数（秒）を入力してください")
    return int(text)


def _field_overrides(values: LiveFormValues) -> dict | FormError:
    """各フィールドを個別に検証し、live_* の上書き dict を作る"""
    model = values.model.strip()
    if not model:
        return FormError("model", "Whisperモデルを入力してください")
    language = _parse_language(values.language)
    if isinstance(language, FormError):
        return language
    llm_model = values.llm_model.strip()
    if values.llm_enabled and not llm_model:
        return FormError("llm_model", "要約に使うLLMモデルを入力してください")
    base_url = values.base_url.strip()
    # 要約を無効にしていても、保存される接続先は常にローカルに限定する
    if not is_loopback_url(base_url):
        return FormError("base_url", LOOPBACK_ONLY_MESSAGE)
    interval = _parse_interval(values.summary_interval)
    if isinstance(interval, FormError):
        return interval
    refine_model = values.refine_model.strip()
    if not refine_model:
        return FormError("refine_model", "高精度モデルを入力してください")
    return {
        "live_engine": values.engine.strip(),
        "live_whisper_model": model,
        "live_vad": values.vad.strip(),
        "live_language": language,
        "live_llm_enabled": bool(values.llm_enabled),
        "live_llm_model": llm_model,
        "live_llm_base_url": base_url,
        "live_summary_interval_sec": interval,
        "live_refine_enabled": bool(values.refine_enabled),
        "live_refine_model": refine_model,
    }


def validate_live_form(values: LiveFormValues, base_config: VoiceNoteConfig) -> dict | FormError:
    """入力値を検証し、live_* フィールドの上書き dict か FormError を返す（例外は送出しない）

    最後に `LiveSettings.from_config` で全体を検証し、エンジン・VAD の不正などを検出する。
    返す dict は live_* キーのみなので、基本タブの項目は上書きしない。
    """
    overrides = _field_overrides(values)
    if isinstance(overrides, FormError):
        return overrides
    try:
        LiveSettings.from_config(replace(base_config, **overrides))
    except ValueError as e:
        return FormError("settings", str(e))
    return overrides


def model_after_engine_change(old_engine: str, new_engine: str, current_model: str) -> str:
    """エンジン切替後のモデル名を返す

    現在のモデルが旧エンジンの既定のままなら新エンジンの既定に差し替える。
    利用者が編集済み（既定と異なる）なら、その値を保持する。
    """
    if old_engine == new_engine:
        return current_model
    if current_model.strip() == default_live_model_for(old_engine):
        return default_live_model_for(new_engine)
    return current_model


def cpu_model_warning(engine: str, model: str) -> str | None:
    """faster-whisper(CPU) で実時間に追いつかないモデルを選んだときの警告文を返す"""
    if engine != ENGINE_FASTER_WHISPER or model.strip() not in CPU_UNSUITABLE_MODELS:
        return None
    return (
        f"{model.strip()} は CPU（faster-whisper）では実時間に追いつきません。"
        "small などの軽いモデルを推奨します"
    )


def _mlx_installed(find_spec: Callable[[str], object]) -> bool:
    """mlx_whisper が import 可能か（読み込みはしない）"""
    try:
        return find_spec(MLX_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def engine_choices(
    *,
    find_spec: Callable[[str], object] = importlib.util.find_spec,
    machine: Callable[[], str] = platform.machine,
    system: str = sys.platform,
) -> EngineChoices:
    """エンジンの選択肢を返す。mlx が使えない環境では mlx を無効にして理由を注記する

    Args:
        find_spec: パッケージ探索関数（テスト用に差し替え可能）
        machine: CPU アーキテクチャを返す関数
        system: sys.platform 相当の文字列
    """
    if system != "darwin" or machine() != "arm64":
        note = "mlx は Apple Silicon の Mac でのみ利用できます"
    elif not _mlx_installed(find_spec):
        note = f"mlx を使うには `{MLX_INSTALL_HINT}` が必要です"
    else:
        return EngineChoices(values=ENGINE_ORDER, disabled=frozenset(), note=None)
    return EngineChoices(values=ENGINE_ORDER, disabled=frozenset({ENGINE_MLX}), note=note)


def model_candidates(engine: str) -> tuple[str, ...]:
    """エンジン別の Whisper モデル候補（mlx は短縮名マップのキー）"""
    if engine == ENGINE_MLX:
        return tuple(MLX_MODEL_REPOS)
    return FASTER_WHISPER_MODELS


def refine_model_candidates() -> tuple[str, ...]:
    """高精度再文字起こしのモデル候補(mlx の短縮名マップのキー)"""
    return tuple(MLX_MODEL_REPOS)
