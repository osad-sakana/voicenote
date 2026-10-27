"""live.model_fetch のテスト(実ネットワークは使わない)"""

import errno
import socket
import sys

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

from live.engines import ENGINE_FASTER_WHISPER, ENGINE_MLX, MLX_MODEL_REPOS
from live.model_fetch import (
    FASTER_WHISPER_ALLOW_PATTERNS,
    FetchErrorInfo,
    RepoSpec,
    describe_fetch_error,
    fetch,
    is_cached,
    resolve_repo,
)


class TestResolveRepo:
    def test_mlx_short_name_uses_engine_map(self):
        spec = resolve_repo(ENGINE_MLX, "small")
        assert spec == RepoSpec(repo_id=MLX_MODEL_REPOS["small"], allow_patterns=None)

    def test_mlx_turbo_has_no_mlx_suffix(self):
        assert resolve_repo(ENGINE_MLX, "large-v3-turbo").repo_id == (
            "mlx-community/whisper-large-v3-turbo"
        )

    def test_mlx_full_repo_passes_through(self):
        assert resolve_repo(ENGINE_MLX, "someone/custom-mlx").repo_id == "someone/custom-mlx"

    def test_faster_whisper_short_name(self):
        spec = resolve_repo(ENGINE_FASTER_WHISPER, "small")
        assert spec.repo_id == "Systran/faster-whisper-small"
        assert spec.allow_patterns == FASTER_WHISPER_ALLOW_PATTERNS
        assert "model.bin" in spec.allow_patterns

    def test_faster_whisper_turbo(self):
        assert resolve_repo(ENGINE_FASTER_WHISPER, "turbo").repo_id == (
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        )

    def test_faster_whisper_matches_library_table(self):
        from faster_whisper.utils import _MODELS

        for name, repo in _MODELS.items():
            assert resolve_repo(ENGINE_FASTER_WHISPER, name).repo_id == repo

    def test_faster_whisper_fallback_table_when_private_import_fails(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "faster_whisper.utils", None)
        spec = resolve_repo(ENGINE_FASTER_WHISPER, "large-v3")
        assert spec.repo_id == "Systran/faster-whisper-large-v3"

    def test_faster_whisper_full_repo_id(self):
        spec = resolve_repo(ENGINE_FASTER_WHISPER, "org/my-ct2-model")
        assert spec.repo_id == "org/my-ct2-model"
        assert spec.allow_patterns == FASTER_WHISPER_ALLOW_PATTERNS

    def test_faster_whisper_unknown_short_name(self):
        with pytest.raises(ValueError, match="no-such-model"):
            resolve_repo(ENGINE_FASTER_WHISPER, "no-such-model")

    @pytest.mark.parametrize("engine", [ENGINE_FASTER_WHISPER, ENGINE_MLX])
    def test_local_directory_returns_none(self, tmp_path, engine):
        assert resolve_repo(engine, str(tmp_path)) is None

    def test_unknown_engine(self):
        with pytest.raises(ValueError, match="engine"):
            resolve_repo("whisper.cpp", "small")


_FW_REQUIRED = ("model.bin", "config.json", "tokenizer.json")


def _make_cache(cache_root, repo_id: str, files: tuple[str, ...], *, snapshot=True):
    """tmp_path に HF Hub キャッシュの構造(refs/main とスナップショット)を作る"""
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123")
    snap = repo_dir / "snapshots" / "abc123"
    if snapshot:
        snap.mkdir(parents=True)
        for name in files:
            (snap / name).write_text("x")
    return snap


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """snapshot_download が呼び出し時に読むキャッシュの場所を tmp_path に向ける"""
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)  # 万一でも通信しない
    return cache


class TestIsCachedRequiredFiles:
    """部分ダウンロード(ref と snapshot はあるがファイルが欠けている)を取得済みとみなさない"""

    def test_missing_model_bin_is_not_cached(self, hf_cache):
        _make_cache(hf_cache, "Systran/faster-whisper-small", ("config.json",))
        assert is_cached(resolve_repo(ENGINE_FASTER_WHISPER, "small")) is False

    @pytest.mark.parametrize("missing", _FW_REQUIRED)
    def test_each_faster_whisper_file_is_required(self, hf_cache, missing):
        files = tuple(f for f in _FW_REQUIRED if f != missing)
        _make_cache(hf_cache, "Systran/faster-whisper-small", files)
        assert is_cached(resolve_repo(ENGINE_FASTER_WHISPER, "small")) is False

    def test_complete_faster_whisper_snapshot_is_cached(self, hf_cache):
        _make_cache(hf_cache, "Systran/faster-whisper-small", _FW_REQUIRED)
        assert is_cached(resolve_repo(ENGINE_FASTER_WHISPER, "small")) is True

    def test_dangling_symlink_counts_as_missing(self, hf_cache):
        snap = _make_cache(hf_cache, "Systran/faster-whisper-small", ("config.json",))
        (snap / "tokenizer.json").write_text("x")
        (snap / "model.bin").symlink_to(snap / "no-such-blob")
        assert is_cached(resolve_repo(ENGINE_FASTER_WHISPER, "small")) is False

    @pytest.mark.parametrize("weights", ["weights.npz", "weights.safetensors", "model.safetensors"])
    def test_mlx_weights_npz_or_safetensors(self, hf_cache, weights):
        repo = resolve_repo(ENGINE_MLX, "small")
        _make_cache(hf_cache, repo.repo_id, ("config.json", weights))
        assert is_cached(repo) is True

    def test_mlx_without_weights_is_not_cached(self, hf_cache):
        repo = resolve_repo(ENGINE_MLX, "small")
        _make_cache(hf_cache, repo.repo_id, ("config.json",))
        assert is_cached(repo) is False

    def test_mlx_without_config_is_not_cached(self, hf_cache):
        repo = resolve_repo(ENGINE_MLX, "small")
        _make_cache(hf_cache, repo.repo_id, ("weights.npz",))
        assert is_cached(repo) is False

    def test_no_snapshot_is_not_cached(self, hf_cache):
        _make_cache(hf_cache, "Systran/faster-whisper-small", (), snapshot=False)
        assert is_cached(resolve_repo(ENGINE_FASTER_WHISPER, "small")) is False


class TestIsCached:
    def test_true_when_local_lookup_succeeds(self, monkeypatch, tmp_path):
        received = {}
        for name in _FW_REQUIRED:
            (tmp_path / name).write_text("x")

        def fake(repo_id, **kwargs):
            received.update(repo_id=repo_id, **kwargs)
            return str(tmp_path)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        spec = RepoSpec("Systran/faster-whisper-small", FASTER_WHISPER_ALLOW_PATTERNS)
        assert is_cached(spec) is True
        assert received["repo_id"] == "Systran/faster-whisper-small"
        assert received["local_files_only"] is True
        assert received["allow_patterns"] == list(FASTER_WHISPER_ALLOW_PATTERNS)

    def test_false_when_not_in_cache(self, monkeypatch):
        def fake(repo_id, **kwargs):
            raise LocalEntryNotFoundError("not cached")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        assert is_cached(RepoSpec("a/b", None)) is False

    def test_mlx_spec_omits_allow_patterns(self, monkeypatch):
        received = {}

        def fake(repo_id, **kwargs):
            received.update(kwargs)
            return "/x"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        is_cached(RepoSpec("mlx-community/whisper-small-mlx", None))
        assert "allow_patterns" not in received


def _hf_like_download(byte_steps, total, *, with_bytes_bar=True, path="/cache/snap"):
    """huggingface_hub 1.x の snapshot_download の tqdm 呼び出し方を再現する偽物"""

    def fake(repo_id, *, tqdm_class=None, **kwargs):
        if with_bytes_bar:
            bar = tqdm_class(
                desc="Downloading (incomplete total...)",
                disable=True,
                total=0,
                initial=0,
                unit="B",
                unit_scale=True,
                name="huggingface_hub.snapshot_download",
            )
        files = tqdm_class(desc="Fetching 2 files", total=2)
        if with_bytes_bar:
            bar.total += total
            bar.refresh()
            for step in byte_steps:
                bar.update(step)
        files.update(1)
        files.update(1)
        files.close()
        if with_bytes_bar:
            bar.set_description("Download complete")
            bar.close()
        return path

    return fake


class TestFetch:
    def test_reports_byte_progress(self, monkeypatch):
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", _hf_like_download([100, 200, 700], 1000)
        )
        events = []
        result = fetch(RepoSpec("a/b", None), lambda d, t: events.append((d, t)))
        assert result == "/cache/snap"
        assert events[0] == (0, 0)
        assert (0, 1000) in events
        assert events[-1] == (1000, 1000)
        dones = [d for d, _ in events]
        assert dones == sorted(dones)

    def test_file_count_bar_is_not_reported(self, monkeypatch):
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _hf_like_download([5], 5))
        events = []
        fetch(RepoSpec("a/b", None), lambda d, t: events.append((d, t)))
        assert all(t in (0, 5) for _, t in events)
        assert (1, 2) not in events and (2, 2) not in events

    def test_without_bytes_bar_reports_unknown_total(self, monkeypatch):
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            _hf_like_download([], 0, with_bytes_bar=False),
        )
        events = []
        fetch(RepoSpec("a/b", None), lambda d, t: events.append((d, t)))
        assert events == [(0, 0)]

    def test_passes_repo_and_patterns(self, monkeypatch):
        received = {}

        def fake(repo_id, **kwargs):
            received.update(repo_id=repo_id, **kwargs)
            return "/p"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        fetch(RepoSpec("Systran/faster-whisper-small", FASTER_WHISPER_ALLOW_PATTERNS), None)
        assert received["repo_id"] == "Systran/faster-whisper-small"
        assert received["allow_patterns"] == list(FASTER_WHISPER_ALLOW_PATTERNS)
        assert received.get("local_files_only", False) is False
        assert received["tqdm_class"] is not None

    def test_progress_callback_errors_do_not_abort_download(self, monkeypatch):
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _hf_like_download([1], 1))

        def broken(_d, _t):
            raise RuntimeError("ui gone")

        assert fetch(RepoSpec("a/b", None), broken) == "/cache/snap"

    def test_download_errors_propagate(self, monkeypatch):
        def fake(repo_id, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        with pytest.raises(OSError):
            fetch(RepoSpec("a/b", None), lambda d, t: None)


class TestDescribeFetchError:
    @pytest.mark.parametrize(
        "exc",
        [
            LocalEntryNotFoundError("offline and not cached"),
            OfflineModeIsEnabled("HF_HUB_OFFLINE=1"),
            ConnectionError("refused"),
            TimeoutError("timed out"),
            socket.gaierror(8, "nodename nor servname provided"),
            OSError(errno.ENETUNREACH, "Network is unreachable"),
        ],
    )
    def test_network_errors(self, exc):
        info = describe_fetch_error(exc)
        assert isinstance(info, FetchErrorInfo)
        assert info.kind == "network"
        assert "再試行" in info.message
        assert "small" in info.message
        assert "録音だけ" in info.message

    def test_httpx_connect_error(self):
        import httpx

        info = describe_fetch_error(httpx.ConnectError("dns failure"))
        assert info.kind == "network"

    def test_network_error_wrapped_in_cause_chain(self):
        try:
            try:
                raise ConnectionError("reset")
            except ConnectionError as inner:
                raise RuntimeError("download failed") from inner
        except RuntimeError as outer:
            assert describe_fetch_error(outer).kind == "network"

    def test_disk_full_with_required_size(self):
        exc = OSError(errno.ENOSPC, "No space left on device")
        info = describe_fetch_error(exc, required_bytes=1_500_000_000)
        assert info.kind == "disk_full"
        assert "1.5 GB" in info.message

    def test_disk_full_without_required_size(self):
        info = describe_fetch_error(OSError(errno.ENOSPC, "No space left on device"))
        assert info.kind == "disk_full"
        assert "空き容量" in info.message
        assert "GB" not in info.message

    def test_disk_full_takes_priority_in_chain(self):
        try:
            try:
                raise OSError(errno.ENOSPC, "No space left on device")
            except OSError as inner:
                raise LocalEntryNotFoundError("wrapped") from inner
        except LocalEntryNotFoundError as outer:
            assert describe_fetch_error(outer).kind == "disk_full"

    def test_disk_full_megabytes(self):
        info = describe_fetch_error(OSError(errno.ENOSPC, "full"), required_bytes=144_000_000)
        assert "144 MB" in info.message

    def test_other_errors(self):
        info = describe_fetch_error(ValueError("bad repo"))
        assert info.kind == "other"
        assert "bad repo" in info.message

    def test_cyclic_chain_terminates(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert describe_fetch_error(a).kind == "other"

    def test_info_is_immutable(self):
        info = describe_fetch_error(ValueError("x"))
        with pytest.raises(AttributeError):
            info.kind = "network"  # type: ignore[misc]
