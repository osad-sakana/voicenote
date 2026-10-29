"""
音声認識(ASR)推論エンジン抽象モジュール
リアルタイム文字起こしで、発話区間(16kHz mono float32)を文字列に変換するエンジンを提供する。

対応エンジン:
- faster-whisper: CPU + int8。全プラットフォームで利用可(本体依存)
- mlx: Apple Silicon の MLX 上で動く mlx-whisper。`uv sync --extra live-mlx` で導入

根拠(mlx-whisper 0.4.3 のソースと HuggingFace API で確認, 2026-10-27):
- `mlx_whisper.transcribe(audio, *, path_or_hf_repo, ...)` の audio 型は
  `Union[str, np.ndarray, mx.array]`。ndarray は `audio.log_mel_spectrogram` 内で
  `mx.array(audio)` に変換されるため、16kHz float32 の1次元配列をそのまま渡せる
- 戻り値は `dict(text=..., segments=[...], language=...)` で、各 segment dict は
  "no_speech_prob" と "avg_logprob" を持つ。無音ウィンドウはスキップされ segments が空になりうる
- `beam_size` を指定すると decoding.py が
  `NotImplementedError("Beam search decoder is not yet implemented")` を送出するため、
  mlx は貪欲デコード固定(beam_size は渡さない)
- モデルは `transcribe.ModelHolder` が path_or_hf_repo 単位でプロセス内キャッシュする
- 実在を確認したリポジトリ(HF API): mlx-community/whisper-base-mlx(144MB),
  whisper-small-mlx(481MB), whisper-medium-mlx(1.5GB), whisper-large-v3-mlx(3.1GB),
  whisper-large-v3-turbo(1.6GB, こちらは "-mlx" 接尾辞なし)

- `initial_prompt: Optional[str] = None` を受け付ける(mlx-whisper 0.4.3 の .venv で確認)。
  condition_on_previous_text=False でも、区間(30秒以内)の最初のウィンドウには効く

faster-whisper 1.2.1: `WhisperModel.transcribe(audio: Union[str, BinaryIO, np.ndarray], ...)` は
`(segments ジェネレータ, info)` を返し、各 Segment は no_speech_prob / avg_logprob を持つ。
"""

from __future__ import annotations

import gc
import importlib
import logging
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
FASTER_WHISPER_BEAM_SIZE = 1
MLX_INSTALL_HINT = "uv sync --extra live-mlx"
_WARMUP_SECONDS = 1.0
_WARMUP_LANGUAGE = "en"

ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_MLX = "mlx"
ENGINE_NAMES = (ENGINE_FASTER_WHISPER, ENGINE_MLX)

# 実在を確認した mlx-community リポジトリへの短縮名マップ("/" を含む名前はそのまま使う)
MLX_MODEL_REPOS: dict[str, str] = {
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

Notify = Callable[[str], None]

_logger = logging.getLogger("voicenote")


@dataclass(frozen=True)
class TranscribedText:
    """1区間の文字起こし結果。確率系の値はエンジンが返さない場合 None"""

    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    language: str | None = None


@dataclass(frozen=True)
class TimedText:
    """時刻付きの1セグメント(時刻は渡した音声の先頭からの秒数)。確率系はエンジンが返さなければ None"""

    start: float
    end: float
    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


class SpeechEngine(Protocol):
    """発話区間を文字列に変換する推論エンジン"""

    def load(self) -> None:
        """モデルを読み込む(複数回呼んでも読み込みは1回)"""
        ...

    def transcribe(
        self, samples: np.ndarray, language: str | None, prompt: str | None = None
    ) -> TranscribedText:
        """16kHz mono float32 の1次元配列を文字起こしする。language=None は自動検出

        prompt は Whisper の initial_prompt(用語ヒント)。None・空文字はヒントなし
        """
        ...


class EngineUnavailableError(RuntimeError):
    """エンジンが利用できない(未インストール・非対応環境・モデル読み込み失敗)"""


def _noop(_message: str) -> None:
    """notify 未指定時の何もしない通知先"""


def _is_apple_silicon() -> bool:
    """macOS かつ arm64 なら True"""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _prepare_samples(samples: np.ndarray) -> np.ndarray:
    """入力を変更せず、1次元 float32 の配列にして返す。多次元は ValueError"""
    arr = np.asarray(samples)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"1次元のモノラル音声が必要です: shape={arr.shape}")
    return arr if arr.dtype == np.float32 else arr.astype(np.float32)


def _summarize(
    texts: Sequence[str],
    no_speech_probs: Sequence[float],
    avg_logprobs: Sequence[float],
    language: str | None,
) -> TranscribedText:
    """セグメント列を1つの結果にまとめる。no_speech_prob は最大値、avg_logprob は平均"""
    if not texts:
        return TranscribedText(text="", language=language)
    return TranscribedText(
        text="".join(texts).strip(),
        no_speech_prob=float(max(no_speech_probs)) if no_speech_probs else None,
        avg_logprob=float(np.mean(avg_logprobs)) if avg_logprobs else None,
        language=language,
    )


class FasterWhisperEngine:
    """
    faster-whisper による CPU(int8)推論エンジン。
    モデルはインスタンスにつき1回だけ読み込む。VAD は呼び出し側で済んでいるため vad_filter=False、
    区間をまたぐ幻覚の連鎖を防ぐため condition_on_previous_text=False、遅延優先で beam_size=1。
    """

    def __init__(self, model_name: str, notify: Notify | None = None):
        self._model_name = model_name
        self._notify = notify or _noop
        self._model = None

    @property
    def model_name(self) -> str:
        """モデル名(faster-whisper のサイズ名またはパス)"""
        return self._model_name

    def load(self) -> None:
        if self._model is not None:
            return
        # monkeypatch で差し替えられるよう、モジュール属性として遅延参照する
        import faster_whisper

        self._notify(f"faster-whisper モデル '{self._model_name}' を読み込んでいます...")
        try:
            self._model = faster_whisper.WhisperModel(
                self._model_name, device="cpu", compute_type="int8"
            )
        except Exception as e:
            raise EngineUnavailableError(
                f"faster-whisper モデル '{self._model_name}' を読み込めませんでした: {e}"
            ) from e
        self._notify(f"faster-whisper モデル '{self._model_name}' の読み込みが完了しました")

    def transcribe(
        self, samples: np.ndarray, language: str | None, prompt: str | None = None
    ) -> TranscribedText:
        audio = _prepare_samples(samples)
        if audio.size == 0:
            return TranscribedText(text="", language=language)
        self.load()
        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=FASTER_WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            vad_filter=False,
        )
        segments = list(segments_iter)
        return _summarize(
            [s.text for s in segments],
            [s.no_speech_prob for s in segments],
            [s.avg_logprob for s in segments],
            getattr(info, "language", None) or language,
        )

    def transcribe_timed(
        self, samples: np.ndarray, language: str | None, prompt: str | None = None
    ) -> tuple[TimedText, ...]:
        """長い音声を文字起こしし、セグメントごとの時刻付きテキストを返す"""
        audio = _prepare_samples(samples)
        if audio.size == 0:
            return ()
        self.load()
        segments_iter, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=FASTER_WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            vad_filter=False,
        )
        return tuple(
            TimedText(
                float(s.start),
                float(s.end),
                s.text.strip(),
                _optional_float(s.no_speech_prob),
                _optional_float(s.avg_logprob),
            )
            for s in segments_iter
        )

    def unload(self) -> None:
        """モデルを手放してメモリを解放する(再度 load() すれば読み直す)"""
        self._model = None
        gc.collect()


def _resolve_mlx_repo(model: str) -> str:
    """短縮名を実在確認済みの mlx-community リポジトリ名に変換する。それ以外はそのまま"""
    return MLX_MODEL_REPOS.get(model, model)


def _import_mlx_whisper():
    """mlx_whisper を遅延 import する。使えなければ EngineUnavailableError"""
    if not _is_apple_silicon():
        raise EngineUnavailableError(
            "mlx エンジンは Apple Silicon (macOS arm64) でのみ利用できます。"
            f"Apple Silicon の Mac で `{MLX_INSTALL_HINT}` を実行してください"
        )
    try:
        import mlx_whisper
    except ImportError as e:
        raise EngineUnavailableError(
            f"mlx-whisper がインストールされていません。`{MLX_INSTALL_HINT}` を実行してください"
        ) from e
    return mlx_whisper


class MlxWhisperEngine:
    """
    mlx-whisper による Apple Silicon (MLX/Metal) 推論エンジン。

    スレッドに関する設計要件: load() と transcribe() は同じスレッドから呼ぶこと
    (呼び出し側のワーカースレッドが保証する)。mlx 0.32.4 / M1 の実測では別スレッドでも動作したが、
    MLX の既定ストリーム・Metal 資源のスレッド安全性は保証されていないため安全側に倒している。

    制約: mlx_whisper の ModelHolder はプロセス全体で1スロットのキャッシュなので、
    異なるリポジトリの MlxWhisperEngine を同一プロセスで交互に使うと毎回モデルを読み直す。

    load() は無音の短い配列で公開 API の transcribe を1回呼んでウォームアップする。
    mlx_whisper はモデルを ModelHolder(path_or_hf_repo 単位)にキャッシュするため、
    これでモデルのダウンロード・読み込みと Metal カーネルの初回コンパイルを先に済ませる。
    """

    def __init__(self, model_repo: str, notify: Notify | None = None):
        self._model_repo = _resolve_mlx_repo(model_repo)
        self._notify = notify or _noop
        self._mlx_whisper = None

    @property
    def model_name(self) -> str:
        """HuggingFace リポジトリ名またはローカルパス"""
        return self._model_repo

    def load(self) -> None:
        if self._mlx_whisper is not None:
            return
        module = _import_mlx_whisper()
        self._notify(f"mlx モデル '{self._model_repo}' を読み込んでいます...")
        try:
            module.transcribe(
                np.zeros(int(SAMPLE_RATE * _WARMUP_SECONDS), dtype=np.float32),
                path_or_hf_repo=self._model_repo,
                language=_WARMUP_LANGUAGE,
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=None,
            )
        except Exception as e:
            raise EngineUnavailableError(
                f"mlx モデル '{self._model_repo}' を読み込めませんでした: {e}"
            ) from e
        self._mlx_whisper = module
        self._notify(f"mlx モデル '{self._model_repo}' の読み込みが完了しました")

    def transcribe(
        self, samples: np.ndarray, language: str | None, prompt: str | None = None
    ) -> TranscribedText:
        audio = _prepare_samples(samples)
        if audio.size == 0:
            return TranscribedText(text="", language=language)
        self.load()
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_repo,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            verbose=None,
        )
        segments = result.get("segments") or []
        return _summarize(
            [s.get("text", "") for s in segments],
            [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None],
            [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None],
            result.get("language") or language,
        )

    def transcribe_timed(
        self, samples: np.ndarray, language: str | None, prompt: str | None = None
    ) -> tuple[TimedText, ...]:
        """長い音声を文字起こしし、セグメントごとの時刻付きテキストを返す"""
        audio = _prepare_samples(samples)
        if audio.size == 0:
            return ()
        self.load()
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_repo,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            verbose=None,
        )
        return tuple(
            TimedText(
                float(s.get("start", 0.0)),
                float(s.get("end", 0.0)),
                (s.get("text") or "").strip(),
                _optional_float(s.get("no_speech_prob")),
                _optional_float(s.get("avg_logprob")),
            )
            for s in result.get("segments") or []
        )

    def unload(self) -> None:
        """ModelHolder(プロセス全体で1枠)のモデルを手放し、MLX のキャッシュを解放する

        ModelHolder は新しいモデルを読み込んでから古いものを置き換えるため、別のモデルを
        読む前にこれを呼ばないと2モデルが同時にメモリへ載る。gc で配列を解放してから
        clear_cache を呼ぶ(解放前のバッファはキャッシュへ戻らない)。
        """
        self._mlx_whisper = None
        holder = getattr(_optional_module("mlx_whisper.transcribe"), "ModelHolder", None)
        if holder is not None:
            holder.model = None
            holder.model_path = None
        gc.collect()
        _clear_mlx_cache()


def _optional_module(name: str):
    """モジュールを import する。無ければ None(解放処理を失敗させない)"""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _clear_mlx_cache() -> None:
    """mx.clear_cache(古い版は mx.metal.clear_cache)を呼ぶ。失敗は警告ログのみ"""
    core = _optional_module("mlx.core")
    if core is None:
        return
    clear = getattr(core, "clear_cache", None) or getattr(
        getattr(core, "metal", None), "clear_cache", None
    )
    if clear is None:
        return
    try:
        clear()
    except Exception as e:
        _logger.warning("MLX のキャッシュを解放できませんでした: %s", type(e).__name__)


def create_engine(name: str, model: str, notify: Notify | None = None) -> SpeechEngine:
    """
    名前からエンジンを生成する。
    mlx が使えない環境では EngineUnavailableError を送出する(フォールバック判断は呼び出し側)。
    モデルの読み込みは行わない(load() を呼ぶこと)。
    """
    if name == ENGINE_FASTER_WHISPER:
        return FasterWhisperEngine(model, notify=notify)
    if name == ENGINE_MLX:
        _import_mlx_whisper()
        return MlxWhisperEngine(model, notify=notify)
    raise ValueError(f"未知のエンジン名です: {name!r}(指定可能: {', '.join(ENGINE_NAMES)})")
