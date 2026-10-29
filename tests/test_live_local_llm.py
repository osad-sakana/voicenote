"""live.local_llm のユニットテスト（ローカルLLM接続のセキュリティ境界）。"""

import ipaddress
import os

import pytest

from live.local_llm import (
    DEFAULT_BASE_URL,
    DEFAULT_TEMPERATURE,
    LocalLLMClient,
    ProbeResult,
    is_loopback_url,
    probe,
)

ALLOWED_URLS = [
    "http://localhost:11434/v1",
    "http://LOCALHOST/",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.2/",
    "http://[::1]:11434",
    "http://localhost./",
    "https://localhost:8443/v1",
]

REJECTED_URLS = [
    "http://localhost.evil.com",
    "http://127.0.0.1.nip.io",
    "http://127.0.0.1@evil.com/",
    "http://evil.com#@127.0.0.1",
    "http://evil.com\\@localhost",
    "http://localhost\\@evil.com",
    "http://0.0.0.0",
    "http://192.168.1.5",
    "http://[::]:11434",
    "ftp://localhost",
    "file://localhost/etc/passwd",
    "",
    "localhost:11434",
    "//localhost:11434",
    "http://",
    "http://localhost:abc/",
    "http://localhost../",
    "http://local host/",
    "http://localhost\n.evil.com/",
]


class TestIsLoopbackUrl:
    @pytest.mark.parametrize("url", ALLOWED_URLS)
    def test_allows_loopback(self, url):
        assert is_loopback_url(url) is True

    @pytest.mark.parametrize("url", REJECTED_URLS)
    def test_rejects_non_loopback(self, url):
        assert is_loopback_url(url) is False

    @pytest.mark.parametrize("value", [None, 123, b"http://localhost/"])
    def test_rejects_non_string(self, value):
        assert is_loopback_url(value) is False

    def test_default_base_url_is_loopback(self):
        assert is_loopback_url(DEFAULT_BASE_URL) is True


class SpyOpenAI:
    """openai.OpenAI の代わりに生成引数を記録するスタブ。"""

    instances: list["SpyOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        SpyOpenAI.instances.append(self)


@pytest.fixture
def spy_openai(monkeypatch):
    import openai

    SpyOpenAI.instances = []
    monkeypatch.setattr(openai, "OpenAI", SpyOpenAI)
    return SpyOpenAI


@pytest.fixture
def hostile_env(monkeypatch):
    """OpenAI SDK / httpx が参照しうる環境変数を攻撃者寄りの値に設定する。"""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://evil.com/v1")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil-proxy.com:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://evil-proxy.com:8080")
    monkeypatch.setenv("ALL_PROXY", "http://evil-proxy.com:8080")


class TestLocalLLMClientConstruction:
    @pytest.mark.parametrize("url", REJECTED_URLS)
    def test_rejects_non_loopback_url(self, url, spy_openai):
        with pytest.raises(ValueError):
            LocalLLMClient(base_url=url, model="qwen3:8b")
        assert spy_openai.instances == []

    def test_passes_explicit_base_url_and_safe_http_client(self, spy_openai):
        LocalLLMClient(base_url="http://localhost:11434/v1", model="qwen3:8b", timeout=12.5)

        assert len(spy_openai.instances) == 1
        kwargs = spy_openai.instances[0].kwargs
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        assert kwargs["base_url"] is not None
        assert kwargs["api_key"] == "ollama"
        http_client = kwargs["http_client"]
        assert http_client.trust_env is False
        assert http_client.follow_redirects is False
        assert http_client.timeout.read == 12.5
        http_client.close()

    def test_real_sdk_targets_loopback_despite_hostile_env(self, hostile_env):
        client = LocalLLMClient(base_url="http://localhost:11434/v1", model="qwen3:8b")
        sdk = client._client
        assert sdk.base_url.host == "localhost"
        assert sdk.base_url.port == 11434
        http_client = sdk._client
        assert http_client.trust_env is False
        assert http_client.follow_redirects is False
        # trust_env=False なのでプロキシ環境変数由来のマウントは作られない
        assert all(
            transport is None or "evil" not in repr(transport)
            for transport in http_client._mounts.values()
        )
        sdk.close()

    @pytest.mark.parametrize("url", ALLOWED_URLS)
    def test_httpx_parsed_host_is_loopback(self, url):
        """urlsplit の判定と httpx が実際に接続するホストが一致することを確認する。"""
        client = LocalLLMClient(base_url=url, model="m")
        host = client._client.base_url.host.rstrip(".")
        if host != "localhost":
            assert ipaddress.ip_address(host).is_loopback
        client._client.close()


class FakeSDKModel:
    def __init__(self, model_id):
        self.id = model_id


class FakeSDK:
    def __init__(self, content="要約結果", models=("qwen3:8b",)):
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                message = type("M", (), {"content": content})()
                choice = type("C", (), {"message": message})()
                return type("R", (), {"choices": [choice]})()

        class _Models:
            def list(self):
                return [FakeSDKModel(m) for m in models]

        self.chat = type("Chat", (), {"completions": _Completions()})()
        self.models = _Models()


class TestLocalLLMClientCalls:
    def _client_with(self, monkeypatch, sdk, **kwargs):
        import openai

        monkeypatch.setattr(openai, "OpenAI", lambda **_: sdk)
        return LocalLLMClient(base_url="http://localhost:11434/v1", model="qwen3:8b", **kwargs)

    def test_complete_sends_default_low_temperature(self, monkeypatch):
        sdk = FakeSDK()
        client = self._client_with(monkeypatch, sdk)
        client.complete([{"role": "user", "content": "hi"}])
        assert DEFAULT_TEMPERATURE == 0.2
        assert sdk.calls[0]["temperature"] == 0.2

    def test_complete_sends_custom_temperature(self, monkeypatch):
        sdk = FakeSDK()
        client = self._client_with(monkeypatch, sdk, temperature=0.7)
        client.complete([{"role": "user", "content": "hi"}])
        assert sdk.calls[0]["temperature"] == 0.7

    def test_custom_temperature_keeps_loopback_check(self, spy_openai):
        with pytest.raises(ValueError):
            LocalLLMClient(base_url="http://evil.com/v1", model="m", temperature=0.5)
        assert spy_openai.instances == []

    def test_complete_sends_model_and_messages(self, monkeypatch):
        sdk = FakeSDK(content="こんにちは")
        client = self._client_with(monkeypatch, sdk)
        messages = [{"role": "user", "content": "hi"}]

        assert client.complete(messages) == "こんにちは"
        assert sdk.calls[0]["model"] == "qwen3:8b"
        assert sdk.calls[0]["messages"] == messages

    def test_complete_returns_empty_string_for_none_content(self, monkeypatch):
        client = self._client_with(monkeypatch, FakeSDK(content=None))
        assert client.complete([{"role": "user", "content": "hi"}]) == ""

    def test_list_models_returns_ids(self, monkeypatch):
        client = self._client_with(monkeypatch, FakeSDK(models=("a:1", "b:2")))
        assert client.list_models() == ["a:1", "b:2"]


class FakeBackend:
    def __init__(self, models=None, error: Exception | None = None):
        self._models = models or []
        self._error = error

    def complete(self, messages):
        return ""

    def list_models(self):
        if self._error is not None:
            raise self._error
        return list(self._models)


class TestProbe:
    def test_ok_when_model_present(self):
        result = probe(FakeBackend(models=["qwen3:8b", "llama3:8b"]), "qwen3:8b")
        assert result == ProbeResult(status="ok", message=result.message)
        assert result.status == "ok"

    def test_ok_when_model_matches_latest_tag(self):
        assert probe(FakeBackend(models=["qwen3:latest"]), "qwen3").status == "ok"

    def test_server_down_when_list_models_fails(self):
        result = probe(FakeBackend(error=ConnectionError("refused")), "qwen3:8b")
        assert result.status == "server_down"
        assert "ollama serve" in result.message

    def test_model_missing(self):
        result = probe(FakeBackend(models=["llama3:8b"]), "qwen3:8b")
        assert result.status == "model_missing"
        assert "ollama pull qwen3:8b" in result.message

    def test_probe_result_is_frozen(self):
        result = probe(FakeBackend(models=["m"]), "m")
        with pytest.raises(AttributeError):
            result.status = "server_down"


@pytest.mark.ollama
@pytest.mark.skipif(
    os.environ.get("VOICENOTE_RUN_OLLAMA") != "1",
    reason="実機 Ollama テストは VOICENOTE_RUN_OLLAMA=1 のときのみ実行",
)
class TestRealOllama:
    def test_probe_against_local_server(self):
        model = os.environ.get("VOICENOTE_OLLAMA_MODEL", "qwen3:8b")
        client = LocalLLMClient(base_url=DEFAULT_BASE_URL, model=model)
        result = probe(client, model)
        assert result.status == "ok", result.message
        reply = client.complete([{"role": "user", "content": "「はい」とだけ答えてください。"}])
        assert isinstance(reply, str)
