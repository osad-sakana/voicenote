"""
フレーム分割モジュール
可変長の音声チャンクを固定長フレームに切り揃える
"""

import numpy as np

DEFAULT_FRAME_SIZE = 512


def _to_mono_1d(chunk: np.ndarray) -> np.ndarray:
    """(n,) または (n, 1) の入力を1次元 float32 の新しい配列に変換する"""
    arr = np.asarray(chunk)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"モノラル音声 (n,) または (n, 1) が必要です: shape={arr.shape}")
    return np.array(arr, dtype=np.float32, copy=True)


class Framer:
    """
    可変長チャンクを固定長フレームへ分割する。
    フレーム長に満たない端数は内部に保持し、次回の push で先頭に連結する。
    入力配列は変更せず、返すフレームは常に新しい配列。
    """

    def __init__(self, frame_size: int = DEFAULT_FRAME_SIZE):
        if frame_size <= 0:
            raise ValueError(f"frame_size は正の整数が必要です: {frame_size}")
        self._frame_size = frame_size
        self._remainder = np.zeros(0, dtype=np.float32)

    @property
    def frame_size(self) -> int:
        """1フレームのサンプル数"""
        return self._frame_size

    @property
    def pending_samples(self) -> int:
        """まだフレームになっていない端数サンプル数"""
        return len(self._remainder)

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """チャンクを追加し、確定した固定長フレームのリストを返す"""
        buffer = np.concatenate([self._remainder, _to_mono_1d(chunk)])
        n_frames = len(buffer) // self._frame_size
        cut = n_frames * self._frame_size
        frames = [buffer[i : i + self._frame_size].copy() for i in range(0, cut, self._frame_size)]
        self._remainder = buffer[cut:].copy()
        return frames
