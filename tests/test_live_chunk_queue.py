"""live.chunk_queue のテスト"""

import threading
import time

import numpy as np
import pytest

from live.chunk_queue import BoundedChunkQueue


def _chunk(value: float = 0.0) -> np.ndarray:
    return np.full(4, value, dtype=np.float32)


def test_put_and_get_preserves_order():
    """投入順に取り出せる"""
    q = BoundedChunkQueue(maxsize=3)
    q.put_nowait(_chunk(1))
    q.put_nowait(_chunk(2))

    assert q.qsize() == 2
    assert q.get(timeout=0.1)[0] == 1
    assert q.get(timeout=0.1)[0] == 2
    assert q.qsize() == 0


def test_put_when_full_drops_without_blocking_or_raising():
    """満杯時はブロックも例外もせず破棄し、dropped を加算する"""
    q = BoundedChunkQueue(maxsize=2)
    q.put_nowait(_chunk(1))
    q.put_nowait(_chunk(2))

    started = time.monotonic()
    accepted = q.put_nowait(_chunk(3))
    elapsed = time.monotonic() - started

    assert accepted is False
    assert elapsed < 0.05
    assert q.dropped == 1
    assert q.qsize() == 2
    assert q.get(timeout=0.1)[0] == 1


def test_put_returns_true_when_accepted():
    """受け入れたら True を返し、dropped は増えない"""
    q = BoundedChunkQueue(maxsize=1)

    assert q.put_nowait(_chunk()) is True
    assert q.dropped == 0


def test_get_returns_none_on_timeout():
    """空のままタイムアウトしたら None を返す"""
    q = BoundedChunkQueue(maxsize=1)

    assert q.get(timeout=0.01) is None


@pytest.mark.parametrize("size", [0, -5])
def test_invalid_maxsize_raises(size):
    """maxsize が正でなければ ValueError(Queue(0) は無制限になるため)"""
    with pytest.raises(ValueError):
        BoundedChunkQueue(maxsize=size)


def test_concurrent_producers_account_for_every_chunk():
    """複数スレッドから投入しても accepted + dropped == 総数"""
    q = BoundedChunkQueue(maxsize=50)
    per_thread = 200
    threads_n = 8
    accepted_counts = [0] * threads_n

    def produce(idx: int) -> None:
        accepted_counts[idx] = sum(q.put_nowait(_chunk()) for _ in range(per_thread))

    threads = [threading.Thread(target=produce, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_accepted = sum(accepted_counts)
    assert total_accepted == q.qsize() == 50
    assert total_accepted + q.dropped == per_thread * threads_n
