"""ローカルLLM（Ollama 等の OpenAI 互換サーバー）クライアント。

リアルタイム文字起こしの要約は「ローカル完結」が要件のため、このモジュールは
**ループバックアドレス以外への接続を一切行わない** ことを保証するセキュリティ境界となる。

- 接続先 URL は `is_loopback_url()` で検証し、`localhost` またはループバック IP 以外は拒否する
- DNS 解決は行わない（`localhost.evil.com` のような名前はそもそも許可しない）
- OpenAI SDK には検証済み URL を必ず明示的に渡す（`base_url=None` を渡すと
  `OPENAI_BASE_URL` 環境変数が使われてしまうため）
- httpx クライアントは `trust_env=False` で生成し、`HTTPS_PROXY` 等のプロキシ環境変数で
  通信が外部へ迂回されないようにする。リダイレクトも追従しない
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT_SEC = 60.0
# 要約は創作より忠実さを優先するため低めの温度にする（入力に無い内容の混入を減らす）
DEFAULT_TEMPERATURE = 0.2
_ALLOWED_SCHEMES = frozenset({"http", "https"})
# Ollama はローカルサーバーでも api_key 引数を要求するSDK仕様のためのダミー値
_DUMMY_API_KEY = "ollama"

Message = Mapping[str, str]
ProbeStatus = Literal["ok", "server_down", "model_missing"]


def _has_unsafe_chars(url: str) -> bool:
    """パーサー間で解釈が分かれうる文字（バックスラッシュ・空白・制御文字）を含むか。"""
    return any(ch == "\\" or ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url)


def _normalize_host(hostname: str) -> str:
    """末尾のドットを1つだけ除去する（`localhost.` → `localhost`）。"""
    return hostname[:-1] if hostname.endswith(".") else hostname


def is_loopback_url(url: object) -> bool:
    """URL の接続先がループバック（自マシン内）であれば True を返す。

    許可するのは scheme が http/https で、ホストが `localhost` もしくは
    ループバック IP アドレス（127.0.0.0/8, ::1）の場合のみ。DNS 解決は行わない。
    userinfo（`user@`）は urlsplit の hostname で除去された実ホストで判定する。
    """
    if not isinstance(url, str) or not url or _has_unsafe_chars(url):
        return False
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        _ = parts.port  # 不正なポート（`:abc` 等）はここで ValueError になる
    except ValueError:
        return False
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not hostname:
        return False
    host = _normalize_host(hostname)
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ChatBackend(Protocol):
    """要約処理が依存するチャットLLMの最小インターフェース。"""

    def complete(self, messages: Sequence[Message]) -> str:
        """メッセージ列を送り、アシスタントの応答テキストを返す。"""
        ...

    def list_models(self) -> list[str]:
        """サーバーで利用可能なモデル名一覧を返す。"""
        ...


class LocalLLMClient:
    """ループバック上の OpenAI 互換サーバー（Ollama 等）へ接続するクライアント。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Args:
            base_url: OpenAI 互換 API のベース URL（ループバックのみ許可）
            model: 使用するモデル名（例: `qwen3:8b`）
            timeout: HTTP タイムアウト秒数
            temperature: 生成時の温度（chat completions の temperature）

        Raises:
            ValueError: base_url がループバック以外を指している場合
        """
        if not is_loopback_url(base_url):
            raise ValueError(
                f"ローカルLLMの接続先はループバックアドレスのみ許可されています: {base_url!r}"
            )
        # 依存の重い SDK は実際に使うときだけ読み込む
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(trust_env=False, timeout=timeout, follow_redirects=False)
        self.base_url = base_url
        self.model = model
        self._temperature = temperature
        self._client = OpenAI(
            base_url=base_url,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
            timeout=timeout,
            max_retries=0,
        )

    def complete(self, messages: Sequence[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[dict(m) for m in messages],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        return [m.id for m in self._client.models.list()]


@dataclass(frozen=True)
class ProbeResult:
    """ローカルLLMサーバーの疎通確認結果。"""

    status: ProbeStatus
    message: str


def _model_available(model: str, available: Sequence[str]) -> bool:
    """タグ省略時は Ollama の慣例どおり `:latest` とみなして照合する。"""
    candidates = {model} if ":" in model else {model, f"{model}:latest"}
    return any(name in candidates for name in available)


def probe(backend: ChatBackend, model: str) -> ProbeResult:
    """サーバーの起動状態とモデルの有無を確認する。例外は送出しない。"""
    try:
        available = backend.list_models()
    except Exception:
        return ProbeResult(
            status="server_down",
            message="ローカルLLMサーバーに接続できません。`ollama serve` を起動してください。",
        )
    if not _model_available(model, available):
        return ProbeResult(
            status="model_missing",
            message=f"モデル {model} が見つかりません。`ollama pull {model}` を実行してください。",
        )
    return ProbeResult(status="ok", message=f"ローカルLLM（{model}）に接続しました。")
