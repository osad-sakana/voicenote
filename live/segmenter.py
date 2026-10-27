"""
発話区間切り出しモジュール
フレームごとの発話確率から、文字起こし単位となる発話区間を切り出す純粋な状態機械
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_SIZE = 512


@dataclass(frozen=True)
class Segment:
    """切り出された発話区間。時刻は録音開始からの秒数"""

    start_sec: float
    end_sec: float
    samples: np.ndarray = field(compare=False, repr=False)

    @property
    def duration_sec(self) -> float:
        """区間の長さ(秒)"""
        return self.end_sec - self.start_sec


def _ms_to_frames(ms: float, sample_rate: int, frame_size: int) -> int:
    """ミリ秒を(切り上げで)フレーム数に変換する"""
    return math.ceil(ms / 1000.0 * sample_rate / frame_size)


def _validate(onset_frames: int, durations_ms: tuple[float, ...], positives: tuple[float, ...]):
    """パラメータを検証し、不正なら ValueError を送出する"""
    if onset_frames < 1:
        raise ValueError(f"onset_frames は1以上が必要です: {onset_frames}")
    if min(durations_ms) < 0:
        raise ValueError("hangover_ms / pre_roll_ms / min_segment_ms は0以上が必要です")
    if min(positives) <= 0:
        raise ValueError("max_segment_sec / sample_rate / frame_size は正の値が必要です")


class Segmenter:
    """
    発話区間を切り出す状態機械。

    - 発話確率が speech_threshold 以上のフレームが onset_frames 連続したら区間開始
    - 開始時には直前 pre_roll_ms 分の音声を区間の先頭に含める
    - 無音が hangover_ms 続いたら区間終了(無音部分も区間に含める)
    - 発話部分が min_segment_ms 未満の区間は破棄する(強制分割の続きは除く)
    - max_segment_sec に達した連続発話は強制的に分割し、続きは切れ目なく次区間となる
    - 出力区間の時刻は単調増加し、互いに重ならない
    """

    def __init__(
        self,
        onset_frames: int = 3,
        hangover_ms: float = 600,
        pre_roll_ms: float = 300,
        min_segment_ms: float = 400,
        max_segment_sec: float = 15,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_size: int = DEFAULT_FRAME_SIZE,
        speech_threshold: float = 0.5,
    ):
        _validate(
            onset_frames,
            (hangover_ms, pre_roll_ms, min_segment_ms),
            (max_segment_sec, sample_rate, frame_size),
        )
        self._sample_rate = sample_rate
        self._onset_frames = onset_frames
        self._hangover_frames = max(1, _ms_to_frames(hangover_ms, sample_rate, frame_size))
        self._min_speech_samples = min_segment_ms / 1000.0 * sample_rate
        self._max_samples = int(max_segment_sec * sample_rate)
        self._speech_threshold = speech_threshold
        pre_roll_frames = _ms_to_frames(pre_roll_ms, sample_rate, frame_size)
        self._lookback: deque[np.ndarray] = deque(maxlen=pre_roll_frames + onset_frames)
        self._pos = 0
        self._reset_idle()

    def _reset_idle(self) -> None:
        """待機状態(区間外)に戻す"""
        self._lookback.clear()
        self._onset_run = 0
        self._active = False
        self._frames: list[np.ndarray] = []
        self._seg_start = self._pos
        self._seg_samples = 0
        self._speech_start = self._pos
        self._last_speech_end = self._pos
        self._silence_run = 0
        self._is_continuation = False

    def push(self, frame: np.ndarray, prob: float) -> list[Segment]:
        """1フレームと発話確率を入力し、確定した区間のリストを返す"""
        samples = np.array(frame, dtype=np.float32, copy=True).ravel()
        self._pos += len(samples)
        is_speech = prob >= self._speech_threshold
        if self._active:
            return self._push_active(samples, is_speech)
        return self._push_idle(samples, is_speech)

    def flush(self) -> list[Segment]:
        """開いている区間を確定して返し、待機状態に戻す"""
        if not self._active:
            self._reset_idle()
            return []
        return self._close()

    def _push_idle(self, samples: np.ndarray, is_speech: bool) -> list[Segment]:
        self._lookback.append(samples)
        self._onset_run = self._onset_run + 1 if is_speech else 0
        if self._onset_run < self._onset_frames:
            return []
        self._open_from_lookback()
        return self._cut() if self._seg_samples >= self._max_samples else []

    def _open_from_lookback(self) -> None:
        """先読みバッファ(pre_roll + onset)から区間を開始する"""
        frames = list(self._lookback)
        onset_samples = sum(len(f) for f in frames[-self._onset_frames :])
        self._frames = frames
        self._seg_samples = sum(len(f) for f in frames)
        self._seg_start = self._pos - self._seg_samples
        self._speech_start = self._pos - onset_samples
        self._last_speech_end = self._pos
        self._silence_run = 0
        self._active = True
        self._lookback.clear()
        self._onset_run = 0

    def _push_active(self, samples: np.ndarray, is_speech: bool) -> list[Segment]:
        self._frames.append(samples)
        self._seg_samples += len(samples)
        if is_speech:
            self._silence_run = 0
            self._last_speech_end = self._pos
        else:
            self._silence_run += 1
        if self._silence_run >= self._hangover_frames:
            return self._close()
        if self._seg_samples >= self._max_samples:
            return self._cut()
        return []

    def _close(self) -> list[Segment]:
        """区間を終了して待機状態に戻る"""
        segment = self._build()
        self._reset_idle()
        return [segment] if segment is not None else []

    def _cut(self) -> list[Segment]:
        """最大長に達した区間を分割し、発話状態のまま次区間を現在位置から始める"""
        segment = self._build()
        self._frames = []
        self._seg_start = self._pos
        self._seg_samples = 0
        self._speech_start = self._pos
        self._last_speech_end = self._pos
        self._is_continuation = True
        return [segment] if segment is not None else []

    def _has_enough_speech(self) -> bool:
        """
        区間を出力するだけの発話を含むか。
        強制分割の続きは発話の一部なので min_segment を適用せず、発話を少しでも含めば残す。
        """
        speech_samples = self._last_speech_end - self._speech_start
        if speech_samples <= 0:
            return False
        return self._is_continuation or speech_samples >= self._min_speech_samples

    def _build(self) -> Segment | None:
        """現在の区間から Segment を作る。発話が不足していれば None"""
        if not self._frames or not self._has_enough_speech():
            return None
        samples = np.concatenate(self._frames)
        samples.flags.writeable = False
        return Segment(
            start_sec=self._seg_start / self._sample_rate,
            end_sec=(self._seg_start + len(samples)) / self._sample_rate,
            samples=samples,
        )
