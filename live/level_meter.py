"""
入力レベルメーター(Tk非依存)
録音コールバック経路から chunk を受け取り、最新の RMS(dBFS) と直近の履歴を保持する。
開始から数秒の履歴を評価して「入力が小さい」ことを利用者に知らせる判定も提供する。

しきい値の根拠:
- 入力が小さい内蔵マイクの実例: RMS -47dB・ピーク0.12 程度
- 通常の声の大きさ: RMS -35〜-25dB 前後
- live.vad.EnergyVAD の既定しきい値が -40dB で、これを下回る入力は発話として検出されにくい
→ 両者の間で VAD 既定と一致する -40dB を「入力が小さい」のしきい値にする。
  無音区間を含む履歴でも発話部分を拾えるよう、平均ではなく 90 パーセンタイルで評価する。
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

_logger = logging.getLogger("voicenote")

SILENCE_FLOOR_DB = -100.0  # 無音時の下限(-inf を履歴に入れないため)
LOW_INPUT_THRESHOLD_DB = -40.0  # これ未満の 90 パーセンタイルを「入力が小さい」とみなす
ASSESS_PERCENTILE = 90.0
MIN_ASSESS_SAMPLES = 20  # 判定に必要な最小チャンク数(足りなければ判定保留)
DEFAULT_HISTORY_SIZE = 500  # 16kHz・512サンプル/チャンクで約16秒分
_EPSILON = 1e-10  # live.vad と同じ値(dBFS 計算を揃える)

ChunkSink = Callable[[np.ndarray], None]
AssessmentStatus = Literal["pending", "low", "ok"]


def rms_dbfs(chunk: np.ndarray) -> float | None:
    """chunk の RMS を dBFS で返す。空なら None。下限は SILENCE_FLOOR_DB"""
    if chunk.size == 0:
        return None
    rms = math.sqrt(float(np.mean(np.square(chunk, dtype=np.float64))))
    return max(20.0 * math.log10(rms + _EPSILON), SILENCE_FLOOR_DB)


class LevelMeter:
    """最新の入力レベル(dBFS)と固定長の履歴を保持するメーター

    feed は録音コールバックスレッドから呼ばれる前提で、計算と代入/追記だけを行う。
    属性代入と deque.append は GIL 下で原子的なのでロックは使わない。
    """

    def __init__(self, history_size: int = DEFAULT_HISTORY_SIZE):
        if history_size <= 0:
            raise ValueError(f"history_size は1以上が必要です: {history_size}")
        self._level_db = SILENCE_FLOOR_DB
        self._history: deque[float] = deque(maxlen=history_size)

    def feed(self, chunk: np.ndarray) -> None:
        """chunk((n,1) float32)の RMS を計算して最新値を更新する。chunk は変更しない"""
        db = rms_dbfs(chunk)
        if db is None:
            return
        self._level_db = db
        self._history.append(db)

    @property
    def level_db(self) -> float:
        """最新の入力レベル(dBFS)。未入力時は SILENCE_FLOOR_DB"""
        return self._level_db

    @property
    def history_db(self) -> tuple[float, ...]:
        """直近の入力レベル履歴(古い順)のスナップショット"""
        # deque の反復中に別スレッドが append すると RuntimeError になりうるため copy() を経由する
        return tuple(self._history.copy())


@dataclass(frozen=True)
class InputLevelAssessment:
    """入力レベル判定の結果。percentile_db は判定保留時 None"""

    status: AssessmentStatus
    percentile_db: float | None
    message: str

    @property
    def is_low(self) -> bool:
        return self.status == "low"


def assess_input_level(
    history_db: Iterable[float],
    *,
    threshold_db: float = LOW_INPUT_THRESHOLD_DB,
    min_samples: int = MIN_ASSESS_SAMPLES,
) -> InputLevelAssessment:
    """開始から数秒の履歴の 90 パーセンタイルがしきい値未満なら「入力が小さい」と判定する"""
    values = np.asarray(tuple(history_db), dtype=np.float64)
    if values.size < min_samples:
        return InputLevelAssessment("pending", None, "")
    level = float(np.percentile(values, ASSESS_PERCENTILE))
    if level < threshold_db:
        message = (
            f"マイクの入力が小さいようです(約 {level:.0f} dB)。"
            "マイクに近づくか、システム設定で入力音量を上げてください。"
        )
        return InputLevelAssessment("low", level, message)
    return InputLevelAssessment("ok", level, "")


def tee(*sinks: ChunkSink) -> ChunkSink:
    """複数の chunk_sink を順に呼ぶ sink を返す。1つの例外は記録して他の sink を続行する"""
    for sink in sinks:
        if not callable(sink):
            raise TypeError(f"chunk_sink は呼び出し可能である必要があります: {sink!r}")
    targets = tuple(sinks)

    def _fan_out(chunk: np.ndarray) -> None:
        for sink in targets:
            try:
                sink(chunk)
            except Exception:
                _logger.exception("chunk_sink の呼び出しに失敗しました: %r", sink)

    return _fan_out
