"""live.refine_fetch(高精度モデルのバックグラウンド取得)のテスト。通信せず Fake で検証する"""

from __future__ import annotations

import threading

import pytest

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
    created: list[threading.Thread] = []

    def factory(**kwargs):
        thread = threading.Thread(**kwargs)
        created.append(thread)
        return thread

    bg = BackgroundModelFetch(thread_factory=factory)
    assert bg.start(fetcher, "mlx", "large-v3") == "started"
    assert created[0].daemon is True
    assert bg.is_fetching(fetcher, "mlx", "large-v3") is True
    gate.set()
    created[0].join(timeout=5.0)
    assert not created[0].is_alive()
    assert fetcher.fetched == ["repo/large-v3"]
    assert bg.is_fetching(fetcher, "mlx", "large-v3") is False


class _RaisingResolve(FakeFetcher):
    def __init__(self, error):
        super().__init__()
        self.resolve_error = error

    def resolve_repo(self, engine, model):
        raise self.resolve_error


@pytest.mark.parametrize(
    "error",
    [RuntimeError("Could not determine home directory."), PermissionError("secret /Users/x")],
    ids=["runtime", "permission"],
)
def test_start_resolve_error_is_invalid_with_type_name_only(error):
    created: list = []
    failures: list[str] = []
    bg = BackgroundModelFetch(thread_factory=_deferred_factory(created))
    status = bg.start(_RaisingResolve(error), "mlx", "large-v3", on_failed=failures.append)
    assert status == "invalid"
    assert created == []
    assert failures == [f"高精度モデル large-v3 を確認できません（{type(error).__name__}）"]


@pytest.mark.parametrize("error", [RuntimeError("x"), PermissionError("x")])
def test_is_fetching_resolve_error_is_false(error):
    bg = BackgroundModelFetch(thread_factory=_deferred_factory([]))
    assert bg.is_fetching(_RaisingResolve(error), "mlx", "large-v3") is False


def test_is_cached_error_is_invalid():
    class _RaisingCached(FakeFetcher):
        def is_cached(self, repo):
            raise OSError("disk")

    failures: list[str] = []
    bg = BackgroundModelFetch(thread_factory=_deferred_factory([]))
    assert bg.start(_RaisingCached(), "mlx", "large-v3", on_failed=failures.append) == "invalid"
    assert "OSError" in failures[0]


def test_thread_start_failure_is_invalid_and_can_retry():
    failures: list[str] = []
    attempts = {"n": 0}

    class _FailingStart(DeferredThread):
        def start(self):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("can't start new thread")
            super().start()

    bg = BackgroundModelFetch(thread_factory=lambda **kw: _FailingStart(**kw))
    fetcher = FakeFetcher()
    assert bg.start(fetcher, "mlx", "large-v3", on_failed=failures.append) == "invalid"
    assert "RuntimeError" in failures[0]
    assert bg.is_fetching(fetcher, "mlx", "large-v3") is False
    assert bg.start(fetcher, "mlx", "large-v3") == "started"


def test_thread_factory_failure_is_invalid():
    def factory(**_kwargs):
        raise RuntimeError("no threads")

    bg = BackgroundModelFetch(thread_factory=factory)
    assert bg.start(FakeFetcher(), "mlx", "large-v3") == "invalid"
