"""live.framer のテスト"""

import numpy as np
import pytest

from live.framer import Framer


def test_returns_no_frame_until_enough_samples():
    """フレーム長未満の入力ではフレームを返さず、端数として保持する"""
    framer = Framer(frame_size=4)

    frames = framer.push(np.arange(3, dtype=np.float32))

    assert frames == []
    assert framer.pending_samples == 3


def test_keeps_remainder_after_emitting_frames():
    """フレーム長を超える分は端数として次回に持ち越す"""
    framer = Framer(frame_size=4)

    frames = framer.push(np.arange(10, dtype=np.float32))

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(frames[1], [4, 5, 6, 7])
    assert framer.pending_samples == 2


def test_frame_spans_multiple_chunks():
    """複数チャンクにまたがるサンプルが1フレームに連結される"""
    framer = Framer(frame_size=4)

    first = framer.push(np.array([0, 1, 2], dtype=np.float32))
    second = framer.push(np.array([3, 4], dtype=np.float32))
    third = framer.push(np.array([5, 6, 7, 8], dtype=np.float32))

    assert first == []
    assert len(second) == 1
    np.testing.assert_array_equal(second[0], [0, 1, 2, 3])
    assert len(third) == 1
    np.testing.assert_array_equal(third[0], [4, 5, 6, 7])
    assert framer.pending_samples == 1


def test_accepts_column_vector_input():
    """sounddevice 形式の (n, 1) 入力を1次元フレームに変換する"""
    framer = Framer(frame_size=4)
    chunk = np.arange(8, dtype=np.float32).reshape(-1, 1)

    frames = framer.push(chunk)

    assert len(frames) == 2
    assert all(f.ndim == 1 and f.shape == (4,) for f in frames)
    assert all(f.dtype == np.float32 for f in frames)
    np.testing.assert_array_equal(frames[1], [4, 5, 6, 7])


def test_does_not_mutate_input_and_returns_new_arrays():
    """入力配列を変更せず、入力とメモリを共有しない新しい配列を返す"""
    framer = Framer(frame_size=4)
    chunk = np.arange(8, dtype=np.float32).reshape(-1, 1)
    original = chunk.copy()

    frames = framer.push(chunk)
    frames[0][:] = 99.0

    np.testing.assert_array_equal(chunk, original)
    assert not any(np.shares_memory(f, chunk) for f in frames)


def test_converts_dtype_to_float32():
    """float64 入力も float32 フレームとして返す"""
    framer = Framer(frame_size=2)

    frames = framer.push(np.array([0.1, 0.2], dtype=np.float64))

    assert frames[0].dtype == np.float32


def test_default_frame_size_is_512():
    """デフォルトのフレーム長は512サンプル"""
    framer = Framer()

    frames = framer.push(np.zeros(1024, dtype=np.float32))

    assert [f.shape for f in frames] == [(512,), (512,)]


def test_empty_chunk_returns_no_frames():
    """空チャンクは何も返さない"""
    framer = Framer(frame_size=4)

    assert framer.push(np.zeros(0, dtype=np.float32)) == []
    assert framer.pending_samples == 0


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_frame_size_raises(size):
    """フレーム長が正でなければ ValueError"""
    with pytest.raises(ValueError):
        Framer(frame_size=size)


def test_multichannel_input_raises():
    """2チャンネル以上の入力は ValueError"""
    framer = Framer(frame_size=4)

    with pytest.raises(ValueError):
        framer.push(np.zeros((8, 2), dtype=np.float32))
