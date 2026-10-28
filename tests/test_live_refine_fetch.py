"""live.refine_fetch(高精度モデルのバックグラウンド取得)のテスト。通信せず Fake で検証する"""

from __future__ import annotations

import threading

from live.model_fetch import RepoSpec
from live.refine_fetch import BackgroundModelFetch


class FakeFetcher:
    def __init__(self, cached=False, error: Exception | None = None, gate=None):
        self.cached = cached
        self.error = error
        self.gate = gate
        self.fetched: list[str] = []

    def resolve_repo(self, engine, model):
        if model == "bad":
            raise ValueError("不明なモデル")
        return RepoSpec(f"repo/{model}")

    def is_cached(self, repo):
        return self.cached

    def fetch(self, repo, on_progress):
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        if self.error is not None:
            raise self.error
        self.fetched.append(repo.repo_id)
        self.cached = True
        return "/cache"


class DeferredThread:
    def __init__(self, target, args=(), daemon=None, name=None, **kwargs):
        self.target, self.args, self.daemon, self.name = target, args, daemon, name
        self.started = self.finished = False

    def start(self):
        self.started = True

    def run(self):
        try:
            self.target(*self.args)
        finally:
            self.finished = True

    def is_alive(self):
        return self.started and not self.finished


def _deferred_factory(created):
    def factory(**kwargs):
        thread = DeferredThread(**kwargs)
        created.append(thread)
        return thread

    return factory


def test_cached_model_starts_nothing():
    created: list = []
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    assert bg.start(FakeFetcher(cached=True), "mlx", "large-v3") == "cached"
    assert created == []
    assert bg.is_fetching(FakeFetcher(cached=True), "mlx", "large-v3") is False


def test_starts_daemon_thread_and_reports_fetching():
    created: list = []
    fetcher = FakeFetcher()
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    assert bg.start(fetcher, "mlx", "large-v3") == "started"
    assert created[0].daemon is True
    assert bg.is_fetching(fetcher, "mlx", "large-v3") is True
    created[0].run()
    assert fetcher.fetched == ["repo/large-v3"]
    assert bg.is_fetching(fetcher, "mlx", "large-v3") is False


def test_second_start_while_running_does_not_fetch_twice():
    created: list = []
    fetcher = FakeFetcher()
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    bg.start(fetcher, "mlx", "large-v3")
    assert bg.start(fetcher, "mlx", "large-v3") == "running"
    assert len(created) == 1


def test_failure_calls_on_failed_with_message_only():
    created: list = []
    failures: list[str] = []
    fetcher = FakeFetcher(error=OSError("secret path /Users/x"))
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    bg.start(fetcher, "mlx", "large-v3", on_failed=failures.append)
    created[0].run()
    assert len(failures) == 1
    assert "OSError" in failures[0]
    assert "secret" not in failures[0]
    # 失敗後は再び開始できる
    assert bg.start(fetcher, "mlx", "large-v3") == "started"


def test_on_done_called_after_success():
    created: list = []
    done: list[str] = []
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    bg.start(FakeFetcher(), "mlx", "large-v3", on_done=lambda: done.append("ok"))
    created[0].run()
    assert done == ["ok"]


def test_invalid_model_reports_failure_without_thread():
    created: list = []
    failures: list[str] = []
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    assert bg.start(FakeFetcher(), "mlx", "bad", on_failed=failures.append) == "invalid"
    assert created == [] and failures
    assert bg.is_fetching(FakeFetcher(), "mlx", "bad") is False


def test_callback_errors_do_not_escape():
    created: list = []
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))

    def broken():
        raise RuntimeError("UI")

    bg.start(FakeFetcher(), "mlx", "large-v3", on_done=broken)
    created[0].run()  # 例外にしない


def test_default_factory_uses_real_daemon_thread():
    gate = threading.Event()
    fetcher = FakeFetcher(gate=gate)
    bg = BackgroundModelFetch()
    assert bg.start(fetcher, "mlx", "large-v3") == "started"
    assert bg.is_fetching(fetcher, "mlx", "large-v3") is True
    gate.set()
    for _ in range(200):
        if not bg.is_fetching(fetcher, "mlx", "large-v3"):
            break
        threading.Event().wait(0.01)
    assert fetcher.fetched == ["repo/large-v3"]
