"""
ライブ文字起こしセッションの設定・コールバック・結果の型定義
すべてイミュータブル(frozen dataclass / タプル)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from config import (
    DEFAULT_LIVE_ENGINE,
    DEFAULT_LIVE_LLM_BASE_URL,
    DEFAULT_LIVE_LLM_MODEL,
    DEFAULT_LIVE_SUMMARY_INTERVAL_SEC,
    DEFAULT_LIVE_VAD,
    DEFAULT_LIVE_WHISPER_MODEL,
    LIVE_VAD_CHOICES,
)
from live.segmenter import DEFAULT_SAMPLE_RATE

SUPPORTED_ENGINES = frozenset({"faster-whisper", "mlx"})
SUPPORTED_VADS = frozenset(LIVE_VAD_CHOICES)


@dataclass(frozen=True)
class LiveSettings:
    """ライブ文字起こしの設定。生成時に値を検証し、不正なら ValueError を送出する"""

    engine: str = DEFAULT_LIVE_ENGINE
    model: str = DEFAULT_LIVE_WHISPER_MODEL
    language: str | None = None  # None は自動判定
    vad: str = DEFAULT_LIVE_VAD
    llm_enabled: bool = True
    llm_base_url: str = DEFAULT_LIVE_LLM_BASE_URL
    llm_model: str = DEFAULT_LIVE_LLM_MODEL
    summary_interval_sec: float = DEFAULT_LIVE_SUMMARY_INTERVAL_SEC
    max_backlog_sec: float = 10.0  # 文字起こしの遅れがこれを超えたら要約を保留
    sample_rate: int = DEFAULT_SAMPLE_RATE
    chunk_queue_size: int = 2000  # 録音コールバックからのチャンク(16kHzで数十秒分)
    segment_queue_size: int = 32  # 文字起こし待ちの区間
    summary_poll_sec: float = 0.5  # 要約スレッドが実行条件を確認する間隔
    poll_sec: float = 0.05  # 各ワーカーがキューを待つ間隔(停止要求への応答性)

    def __post_init__(self):
        _validate_types(self)
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError(f"未対応のライブエンジンです: {self.engine!r}")
        if self.vad not in SUPPORTED_VADS:
            raise ValueError(f"未対応の VAD です: {self.vad!r}")
        positives = {
            "summary_interval_sec": self.summary_interval_sec,
            "summary_poll_sec": self.summary_poll_sec,
            "poll_sec": self.poll_sec,
            "chunk_queue_size": self.chunk_queue_size,
            "segment_queue_size": self.segment_queue_size,
            "sample_rate": self.sample_rate,
        }
        for name, value in positives.items():
            if value <= 0:
                raise ValueError(f"{name} は正の値が必要です: {value}")

    @classmethod
    def from_config(cls, config) -> LiveSettings:
        """VoiceNoteConfig の live_* フィールドから生成する(空の言語は自動判定)"""
        return cls(
            engine=config.live_engine,
            model=config.live_whisper_model,
            language=_normalize_language(config.live_language),
            vad=config.live_vad,
            llm_enabled=config.live_llm_enabled,
            llm_base_url=config.live_llm_base_url,
            llm_model=config.live_llm_model,
            summary_interval_sec=config.live_summary_interval_sec,
        )


def _is_number(value) -> bool:
    """bool を除く int/float か(JSON の true が数値として通らないようにする)"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_types(settings: LiveSettings) -> None:
    """手編集された config.json 由来の型の誤りを ValueError として検出する"""
    for name in ("engine", "vad", "llm_base_url", "llm_model"):
        if not isinstance(getattr(settings, name), str):
            raise ValueError(f"{name} は文字列が必要です: {getattr(settings, name)!r}")
    if not isinstance(settings.model, str) or not settings.model.strip():
        raise ValueError(f"model は空でない文字列が必要です: {settings.model!r}")
    if settings.language is not None and not isinstance(settings.language, str):
        raise ValueError(f"language は文字列が必要です: {settings.language!r}")
    if not isinstance(settings.llm_enabled, bool):
        raise ValueError(f"llm_enabled は真偽値が必要です: {settings.llm_enabled!r}")
    for name in ("summary_interval_sec", "max_backlog_sec", "summary_poll_sec", "poll_sec"):
        if not _is_number(getattr(settings, name)):
            raise ValueError(f"{name} は数値が必要です: {getattr(settings, name)!r}")


def _normalize_language(language) -> str | None:
    """config の live_language を検証し、空文字は None(自動判定)にする"""
    if not isinstance(language, str):
        raise ValueError(f"live_language は文字列が必要です: {language!r}")
    return language.strip() or None


def _noop(*_args) -> None:
    """既定のコールバック(何もしない)"""
    return None


@dataclass(frozen=True)
class LiveCallbacks:
    """セッションからの通知先。ワーカースレッドから呼ばれるため、重い処理は避けること"""

    on_segment: Callable[[int, float, float, str], None] = _noop
    on_summary: Callable[[str], None] = _noop
    on_status: Callable[[str], None] = _noop
    on_warning: Callable[[str], None] = _noop


@dataclass(frozen=True)
class LiveSegment:
    """文字起こし済みの1区間。時刻は録音開始からの秒数"""

    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class LiveResult:
    """セッション終了時の結果

    complete が False の場合、stop の timeout 内に処理しきれなかった区間がある。
    """

    segments: tuple[LiveSegment, ...]
    final_summary: str
    dropped_chunks: int
    dropped_segments: int = 0
    complete: bool = True
