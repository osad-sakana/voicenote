"""
Whisper モデル取得モジュール(Tk非依存)
GUI が録音開始前にモデルを HuggingFace Hub から取得し、進捗を表示するための部品。
huggingface_hub / faster_whisper / tqdm は重いので関数内で遅延 import する。

根拠(huggingface_hub 1.3.2 の _snapshot_download.py で確認, 2026-10-27):
- snapshot_download(tqdm_class=X) は X(unit="B", total=0, name=..., disable=...) で
  バイト数の集約バーを1本作り、各ファイルのメタデータ取得時に `bar.total += size; bar.refresh()`、
  受信ごとに `bar.update(n)` する。ファイル数のバー(unit 既定 "it")も同じ X で作られる
- name= は huggingface_hub 独自の tqdm だけが受け付ける引数で、素の tqdm に渡すと例外になる
- オフラインかつ未キャッシュの場合は LocalEntryNotFoundError(OSError 派生)を送出する

faster-whisper 1.2.1 の utils.download_model は `_MODELS` の repo と下記 allow_patterns を使う。
"""

from __future__ import annotations

import errno
import logging
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from live.engines import ENGINE_FASTER_WHISPER, ENGINE_MLX, MLX_MODEL_REPOS

_logger = logging.getLogger("voicenote")

FASTER_WHISPER_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)
# is_cached で存在を確かめるファイル(preprocessor_config.json / vocabulary.* は無いリポジトリがある)
FASTER_WHISPER_REQUIRED_FILES: tuple[str, ...] = ("model.bin", "config.json", "tokenizer.json")
# mlx は重み(weights.npz または *.safetensors)と config.json が必要
MLX_REQUIRED_FILE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("config.json",),
    ("weights.npz", "*.safetensors"),
)

# faster_whisper.utils._MODELS(private)の import に失敗した場合の控え(1.2.1 と同内容)
_FALLBACK_FASTER_WHISPER_MODELS: dict[str, str] = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

ProgressCallback = Callable[[int, int], None]
FetchErrorKind = Literal["network", "disk_full", "other"]

_NETWORK_ERRNOS = frozenset(
    {
        errno.ENETUNREACH,
        errno.ENETDOWN,
        errno.EHOSTUNREACH,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ETIMEDOUT,
    }
)
# import せずにクラス名(MRO)で判定する通信系例外(huggingface_hub / httpx / requests)
_NETWORK_CLASS_NAMES = frozenset(
    {
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "TransportError",  # httpx.ConnectError / ReadTimeout などの基底
        "ConnectionError",  # requests.ConnectionError も含む
        "Timeout",
    }
)
_RECOVERY_HINT = "再試行するか、小さいモデル(small)に変更するか、録音だけモードで録音してください。"


@dataclass(frozen=True)
class RepoSpec:
    """取得対象の HuggingFace リポジトリ。allow_patterns=None はリポジトリ全体"""

    repo_id: str
    allow_patterns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FetchErrorInfo:
    """取得失敗の分類と利用者向け案内文"""

    kind: FetchErrorKind
    message: str


def _faster_whisper_models() -> dict[str, str]:
    try:
        from faster_whisper.utils import _MODELS
    except Exception:
        return _FALLBACK_FASTER_WHISPER_MODELS
    return dict(_MODELS)


def resolve_repo(engine: str, model: str) -> RepoSpec | None:
    """エンジンとモデル名から取得対象を決める。存在するローカルディレクトリなら None"""
    if engine not in (ENGINE_FASTER_WHISPER, ENGINE_MLX):
        raise ValueError(f"未対応の engine です: {engine}")
    if Path(model).expanduser().is_dir():
        return None
    if engine == ENGINE_MLX:
        return RepoSpec(MLX_MODEL_REPOS.get(model, model), None)
    if "/" in model:
        return RepoSpec(model, FASTER_WHISPER_ALLOW_PATTERNS)
    repo_id = _faster_whisper_models().get(model)
    if repo_id is None:
        raise ValueError(f"faster-whisper の不明なモデル名です: {model}")
    return RepoSpec(repo_id, FASTER_WHISPER_ALLOW_PATTERNS)


def _download_kwargs(repo: RepoSpec) -> dict:
    if repo.allow_patterns is None:
        return {}
    return {"allow_patterns": list(repo.allow_patterns)}


def _required_file_groups(repo: RepoSpec) -> tuple[tuple[str, ...], ...]:
    """スナップショットに揃っているべきファイル。各グループのどれか1つ(glob)があればよい

    allow_patterns があれば faster-whisper(CTranslate2)、無ければ mlx のリポジトリとみなす。
    """
    if repo.allow_patterns is not None:
        return tuple((name,) for name in FASTER_WHISPER_REQUIRED_FILES)
    return MLX_REQUIRED_FILE_GROUPS


def _has_files(snapshot: Path, groups: tuple[tuple[str, ...], ...]) -> bool:
    """各グループのパターンに一致する実ファイルがあるか(壊れたシンボリックリンクは欠損扱い)"""
    return all(
        any(path.is_file() for pattern in group for path in snapshot.glob(pattern))
        for group in groups
    )


def is_cached(repo: RepoSpec) -> bool:
    """ローカルキャッシュだけで必要ファイルが揃った snapshot を解決できれば True(通信しない)

    snapshot_download(local_files_only=True) は refs とスナップショットのフォルダがあれば
    ファイルが欠けていても成功するため、中断した部分ダウンロードを取得済みと誤判定しないよう
    必要ファイルの存在も確認する(欠けていれば False を返し、fetch で再開させる)。
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(repo.repo_id, local_files_only=True, **_download_kwargs(repo))
    except Exception:
        return False
    return _has_files(Path(snapshot), _required_file_groups(repo))


def _safe_notify(on_progress: ProgressCallback | None, done: int, total: int) -> None:
    if on_progress is None:
        return
    try:
        on_progress(done, total)
    except Exception:
        _logger.exception("モデル取得の進捗通知に失敗しました")


def _make_progress_tqdm(on_progress: ProgressCallback | None, sink):
    """unit=="B" のバーだけを on_progress に通知する tqdm 派生クラスを作る"""
    from tqdm.std import tqdm as base_tqdm

    class _ProgressTqdm(base_tqdm):
        monitor_interval = 0  # tqdm の監視スレッドを起動しない

        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)  # huggingface_hub 独自引数
            kwargs["disable"] = False  # disable 指定は無視して常に進捗を受け取る
            kwargs["file"] = sink
            self._last_reported: tuple[int, int] | None = None
            super().__init__(*args, **kwargs)

        def _report(self) -> None:
            if getattr(self, "unit", None) != "B":
                return
            current = (int(self.n), int(self.total or 0))
            if current == getattr(self, "_last_reported", None):
                return
            self._last_reported = current
            _safe_notify(on_progress, *current)

        def update(self, n=1):
            displayed = super().update(n)
            self._report()
            return displayed

        def refresh(self, *args, **kwargs):
            result = super().refresh(*args, **kwargs)
            self._report()
            return result

    return _ProgressTqdm


def fetch(repo: RepoSpec, on_progress: ProgressCallback | None) -> str:
    """モデルを取得してローカルパスを返す。進捗を on_progress(done_bytes, total_bytes) で通知する

    - 開始時に必ず (0, 0) を通知する。total_bytes=0 は「総量不明」を表す
    - on_progress は huggingface_hub のワーカースレッドから呼ばれる。GUI 側で UI スレッドへ渡すこと
    - 取得失敗の例外はそのまま送出する(describe_fetch_error で案内文に変換できる)
    """
    from huggingface_hub import snapshot_download

    _safe_notify(on_progress, 0, 0)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        tqdm_class = _make_progress_tqdm(on_progress, sink)
        return snapshot_download(repo.repo_id, tqdm_class=tqdm_class, **_download_kwargs(repo))


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    """__cause__ / __context__ をたどった例外列(循環は打ち切る)"""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_disk_full(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno == errno.ENOSPC


def _is_network(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror)):
        return True
    if isinstance(exc, OSError) and exc.errno in _NETWORK_ERRNOS:
        return True
    return any(cls.__name__ in _NETWORK_CLASS_NAMES for cls in type(exc).__mro__)


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.1f} GB"
    return f"{num_bytes / 1_000_000:.0f} MB"


def describe_fetch_error(
    exc: BaseException, *, required_bytes: int | None = None
) -> FetchErrorInfo:
    """取得失敗の例外を分類し、利用者向けの案内文に変換する(純関数)

    required_bytes: 必要容量がわかる場合に渡す(fetch の進捗で得た total_bytes など)
    """
    chain = _exception_chain(exc)
    if any(_is_disk_full(e) for e in chain):
        need = f"約 {_format_size(required_bytes)} 必要です。" if required_bytes else ""
        message = (
            f"ディスクの空き容量が不足していてモデルを保存できません。{need}"
            "空き容量を確保して再試行するか、小さいモデル(small)に変更してください。"
        )
        return FetchErrorInfo("disk_full", message)
    if any(_is_network(e) for e in chain):
        message = f"ネットワークに接続できずモデルを取得できませんでした。{_RECOVERY_HINT}"
        return FetchErrorInfo("network", message)
    message = f"モデルの取得に失敗しました({type(exc).__name__}: {exc})。{_RECOVERY_HINT}"
    return FetchErrorInfo("other", message)
