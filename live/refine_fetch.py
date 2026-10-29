"""高精度モデル(停止後の再文字起こし用)のバックグラウンド取得(CLI と GUI で共有する)

約3GB の large-v3 の取得で録音の開始を待たせないよう、録音開始後に daemon スレッドで取得する。
停止時にまだ取得中なら、再文字起こしはスキップする(decide_refine(fetching=True))。
同じリポジトリの取得が動いている間は、新たに取得を始めない(二重取得の防止)。

失敗の通知は例外の型名だけにする(パスなどの詳細はログにも出さない)。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Literal

_logger = logging.getLogger("voicenote")

FetchStart = Literal["cached", "started", "running", "invalid"]


def _noop(*_args) -> None:
    return None


def _safe(fn: Callable, *args) -> None:
    try:
        fn(*args)
    except Exception:
        _logger.exception("高精度モデル取得の通知でエラーが発生しました")


class BackgroundModelFetch:
    """リポジトリ単位で1本だけ取得スレッドを動かす

    fetcher は resolve_repo(engine, model) / is_cached(repo) / fetch(repo, on_progress) を持つ
    (live.model_fetch モジュールや gui の ModelFetcher)。
    """

    def __init__(self, *, thread_factory: Callable[..., threading.Thread] = threading.Thread):
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def start(
        self,
        fetcher,
        engine: str,
        model: str,
        *,
        on_failed: Callable[[str], None] = _noop,
        on_done: Callable[[], None] = _noop,
    ) -> FetchStart:
        """未取得なら取得を始める。取得済みは cached、取得中は running、確認できなければ invalid

        どんな例外も外へ出さない(呼び出し側は録音中なので、失敗で録音を止めない)。
        通知・ログは例外の型名だけにする(ホームディレクトリのパスなどを出さない)。
        """
        try:
            return self._start(fetcher, engine, model, on_failed, on_done)
        except Exception as e:
            _logger.warning("高精度モデルを確認できません: %s", type(e).__name__)
            _safe(on_failed, f"高精度モデル {model} を確認できません（{type(e).__name__}）")
            return "invalid"

    def _start(self, fetcher, engine, model, on_failed, on_done) -> FetchStart:
        repo = fetcher.resolve_repo(engine, model)
        if repo is None or fetcher.is_cached(repo):
            return "cached"
        with self._lock:
            current = self._threads.get(repo.repo_id)
            if current is not None and current.is_alive():
                return "running"
            thread = self._thread_factory(
                target=self._run,
                args=(fetcher, repo, model, on_failed, on_done),
                daemon=True,
                name="refine-model-fetch",
            )
            # 登録と開始を同じロック下で行う(開始前の is_alive は False のため)。
            # 開始に失敗したら登録しない(次回の開始を妨げない)
            thread.start()
            self._threads[repo.repo_id] = thread
        return "started"

    def is_fetching(self, fetcher, engine: str, model: str) -> bool:
        """そのモデルの取得スレッドが動いているか(確認できなければ False)"""
        try:
            repo = fetcher.resolve_repo(engine, model)
        except Exception as e:
            _logger.warning("高精度モデルを確認できません: %s", type(e).__name__)
            return False
        if repo is None:
            return False
        with self._lock:
            thread = self._threads.get(repo.repo_id)
            return thread is not None and thread.is_alive()

    @staticmethod
    def _run(fetcher, repo, model: str, on_failed, on_done) -> None:
        try:
            fetcher.fetch(repo, None)
        except Exception as e:
            _logger.warning("高精度モデルの取得に失敗しました: %s", type(e).__name__)
            _safe(on_failed, f"高精度モデル {model} を取得できませんでした（{type(e).__name__}）")
            return
        _logger.info("高精度モデル %s を取得しました", model)
        _safe(on_done)
