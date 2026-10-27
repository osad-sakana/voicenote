"""
音声区間検出(VAD)モジュール
フレームごとの発話確率を返す検出器を提供する
"""

import logging
import math
from collections import deque
from typing import Protocol

import numpy as np

_logger = logging.getLogger("voicenote")

SILERO_FRAME_SIZE = 512
SILERO_WINDOW_FRAMES = 16  # 16 * 512 = 8192 サンプル(16kHz で約0.5秒)
DEFAULT_ENERGY_THRESHOLD_DB = -40.0
ENERGY_SLOPE_DB = 3.0  # シグモイドの傾き(dB)
_EPSILON = 1e-10


class VoiceActivityDetector(Protocol):
    """1フレームの発話確率(0..1)を返す検出器"""

    def speech_prob(self, frame: np.ndarray) -> float: ...


class EnergyVAD:
    """RMS エネルギー(dBFS)を閾値中心のシグモイドで 0..1 に写像する簡易 VAD"""

    def __init__(self, threshold_db: float = DEFAULT_ENERGY_THRESHOLD_DB):
        self._threshold_db = threshold_db

    def speech_prob(self, frame: np.ndarray) -> float:
        samples = np.asarray(frame, dtype=np.float32).ravel()
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        dbfs = 20.0 * math.log10(rms + _EPSILON)
        return _sigmoid((dbfs - self._threshold_db) / ENERGY_SLOPE_DB)


def _sigmoid(x: float) -> float:
    """オーバーフローしない標準シグモイド"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _load_silero_model():
    """faster_whisper 同梱の Silero VAD モデルを遅延 import で読み込む"""
    from faster_whisper.vad import get_vad_model

    return get_vad_model()


class SileroVAD:
    """
    faster_whisper 同梱の silero_vad_v6.onnx を使う VAD。
    SileroVADModel は呼び出しごとに内部状態をリセットするバッチ API のため、
    直近 window_frames フレームのローリング窓をまとめて渡し、最後のフレームの確率を使う。
    """

    def __init__(self, model=None, window_frames: int = SILERO_WINDOW_FRAMES):
        if window_frames <= 0:
            raise ValueError(f"window_frames は正の整数が必要です: {window_frames}")
        self._model = model if model is not None else _load_silero_model()
        self._window: deque[np.ndarray] = deque(
            (np.zeros(SILERO_FRAME_SIZE, dtype=np.float32) for _ in range(window_frames)),
            maxlen=window_frames,
        )
        self._padding = np.zeros(SILERO_FRAME_SIZE, dtype=np.float32)

    def speech_prob(self, frame: np.ndarray) -> float:
        samples = np.array(frame, dtype=np.float32, copy=True).ravel()
        if samples.size != SILERO_FRAME_SIZE:
            raise ValueError(
                f"SileroVAD は {SILERO_FRAME_SIZE} サンプルのフレームが必要です: {samples.size}"
            )
        self._window.append(samples)
        # SileroVADModel は最後のフレームの末尾64サンプルを in-place でゼロにするため、
        # 末尾にゼロ埋めフレームを1つ足した新しい配列を渡し、その直前(最新フレーム)の確率を読む
        audio = np.concatenate([*self._window, self._padding])
        out = np.asarray(self._model(audio)).ravel()
        return float(np.clip(out[-2], 0.0, 1.0))


def vad_kind(vad: VoiceActivityDetector) -> str | None:
    """検出器の実際の種類("energy" / "silero")を返す。それ以外は None

    create_vad("silero") がフォールバックしたかを呼び出し側で判定するために使う
    """
    if isinstance(vad, EnergyVAD):
        return "energy"
    if isinstance(vad, SileroVAD):
        return "silero"
    return None


def create_vad(name: str) -> VoiceActivityDetector:
    """名前から VAD を生成する。silero の初期化に失敗したら EnergyVAD にフォールバック"""
    if name == "energy":
        return EnergyVAD()
    if name == "silero":
        try:
            return SileroVAD()
        except Exception as e:
            _logger.warning("Silero VAD の初期化に失敗したため EnergyVAD を使用します: %s", e)
            return EnergyVAD()
    raise ValueError(f"未知の VAD 名です: {name}")
