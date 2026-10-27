"""
リアルタイム文字起こし用 ASR エンジンのベンチマーク(開発用スクリプト。パッケージには含めない)

WAV を live.framer / live.vad("energy") / live.segmenter で発話区間に分割し、
指定したエンジン×モデルの組合せごとに以下を計測して rich の表で出力する。

- モデル読み込み時間(初回ダウンロードを含みうる)
- 区間単位の推論遅延 p50 / p95
- RTF(総処理時間 / 総音声長)
- ピーク RSS メモリ(resource.getrusage。組合せごとに spawn した子プロセスで計測)
- 区間ごとの認識テキスト

各組合せは別プロセスで実行するため、ru_maxrss が前の組合せの影響を受けない。
また子プロセスのメインスレッドで load と推論を行うので、mlx の同一スレッド要件も満たす。

使い方:
    uv run scripts/bench_live_asr.py --wav sample.wav \\
        --engine faster-whisper --model small \\
        --engine mlx --model small --engine mlx --model large-v3-turbo --language ja
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from live.engines import SAMPLE_RATE, EngineUnavailableError, create_engine  # noqa: E402
from live.framer import Framer  # noqa: E402
from live.segmenter import Segmenter  # noqa: E402
from live.vad import create_vad  # noqa: E402

console = Console()
_WARMUP_SECONDS = 1.0


@dataclass(frozen=True)
class SegmentAudio:
    """ベンチ対象の1区間(子プロセスへ渡すため素の値で持つ)"""

    start_sec: float
    end_sec: float
    samples: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class BenchResult:
    """1組合せの計測結果。skipped_reason があれば計測していない"""

    engine: str
    model: str
    load_sec: float = 0.0
    latencies_sec: tuple[float, ...] = ()
    audio_sec: float = 0.0
    peak_rss_mb: float = 0.0
    texts: tuple[str, ...] = ()
    skipped_reason: str | None = None


def load_wav_16k_mono(path: Path) -> np.ndarray:
    """WAV を読み込み、16kHz mono float32 に変換した新しい配列を返す"""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(str(path))
    audio = np.asarray(data)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        gcd = np.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, rate // gcd).astype(np.float32)
    return audio


def split_segments(audio: np.ndarray) -> tuple[SegmentAudio, ...]:
    """Framer + EnergyVAD + Segmenter で発話区間に分割する"""
    framer = Framer()
    vad = create_vad("energy")
    segmenter = Segmenter()
    segments = []
    for frame in framer.push(audio):
        segments.extend(segmenter.push(frame, vad.speech_prob(frame)))
    segments.extend(segmenter.flush())
    return tuple(SegmentAudio(s.start_sec, s.end_sec, np.array(s.samples)) for s in segments)


def peak_rss_mb() -> float:
    """このプロセスのピーク RSS(MB)。macOS はバイト、Linux は KB 単位で返る"""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return maxrss / divisor


def run_combo(
    engine_name: str, model: str, segments: tuple[SegmentAudio, ...], language: str | None
) -> BenchResult:
    """1組合せを計測する(子プロセスのメインスレッドで実行される)"""
    try:
        engine = create_engine(engine_name, model)
        t0 = time.perf_counter()
        engine.load()
        load_sec = time.perf_counter() - t0
    except EngineUnavailableError as e:
        return BenchResult(engine_name, model, skipped_reason=str(e))
    # 条件をそろえるため、両エンジンとも1回ウォームアップしてから計測する
    engine.transcribe(np.zeros(int(SAMPLE_RATE * _WARMUP_SECONDS), dtype=np.float32), language)
    latencies = []
    texts = []
    for seg in segments:
        t0 = time.perf_counter()
        result = engine.transcribe(seg.samples, language)
        latencies.append(time.perf_counter() - t0)
        texts.append(result.text)
    return BenchResult(
        engine=engine_name,
        model=model,
        load_sec=load_sec,
        latencies_sec=tuple(latencies),
        audio_sec=sum(s.end_sec - s.start_sec for s in segments),
        peak_rss_mb=peak_rss_mb(),
        texts=tuple(texts),
    )


def run_isolated(
    engine_name: str, model: str, segments: tuple[SegmentAudio, ...], language: str | None
) -> BenchResult:
    """組合せを spawn した子プロセスで実行し、結果を返す"""
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        try:
            return pool.apply(run_combo, (engine_name, model, segments, language))
        except Exception as e:
            return BenchResult(engine_name, model, skipped_reason=f"実行エラー: {e}")


def percentile(values: tuple[float, ...], q: float) -> float:
    """値の q パーセンタイル。空なら 0.0"""
    return float(np.percentile(values, q)) if values else 0.0


def render_summary(results: list[BenchResult], n_segments: int, audio_sec: float) -> Table:
    """組合せごとの集計表を作る"""
    table = Table(title=f"ASR ベンチ: {n_segments} 区間 / 音声 {audio_sec:.1f} 秒")
    for col in ("engine", "model", "load(s)", "p50(s)", "p95(s)", "RTF", "peak RSS(MB)", "備考"):
        table.add_column(col)
    for r in results:
        if r.skipped_reason:
            reason = r.skipped_reason.strip().splitlines()[0]
            table.add_row(r.engine, r.model, "-", "-", "-", "-", "-", f"skip: {reason}")
            continue
        rtf = sum(r.latencies_sec) / r.audio_sec if r.audio_sec > 0 else 0.0
        table.add_row(
            r.engine,
            r.model,
            f"{r.load_sec:.2f}",
            f"{percentile(r.latencies_sec, 50):.3f}",
            f"{percentile(r.latencies_sec, 95):.3f}",
            f"{rtf:.3f}",
            f"{r.peak_rss_mb:.0f}",
            "",
        )
    return table


def render_texts(result: BenchResult, segments: tuple[SegmentAudio, ...]) -> Table:
    """1組合せの区間ごとのテキストと遅延の表を作る"""
    table = Table(title=f"{result.engine} / {result.model}")
    for col in ("#", "区間(s)", "遅延(s)", "テキスト"):
        table.add_column(col)
    for i, (seg, lat, text) in enumerate(
        zip(segments, result.latencies_sec, result.texts, strict=True), start=1
    ):
        table.add_row(str(i), f"{seg.start_sec:.1f}-{seg.end_sec:.1f}", f"{lat:.3f}", text)
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="リアルタイム文字起こし ASR エンジンのベンチ")
    parser.add_argument("--wav", type=Path, required=True, help="入力 WAV ファイル")
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        help="faster-whisper | mlx(--model と同数・同順で複数指定可)",
    )
    parser.add_argument("--model", action="append", required=True, help="モデル名(複数指定可)")
    parser.add_argument("--language", default=None, help="言語コード(省略時は自動検出)")
    args = parser.parse_args(argv)
    if len(args.engine) != len(args.model):
        parser.error("--engine と --model は同じ回数だけ指定してください")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.wav.is_file():
        console.print(f"[red]WAV ファイルが見つかりません: {args.wav}[/red]")
        return 1
    audio = load_wav_16k_mono(args.wav)
    segments = split_segments(audio)
    if not segments:
        console.print("[red]発話区間が検出されませんでした[/red]")
        return 1
    audio_sec = sum(s.end_sec - s.start_sec for s in segments)
    console.print(f"区間数: {len(segments)} / 区間合計 {audio_sec:.1f} 秒")
    results = []
    for engine_name, model in zip(args.engine, args.model, strict=True):
        console.print(f"[cyan]計測中: {engine_name} / {model}[/cyan]")
        results.append(run_isolated(engine_name, model, segments, args.language))
    console.print(render_summary(results, len(segments), audio_sec))
    for r in results:
        if not r.skipped_reason:
            console.print(render_texts(r, segments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
