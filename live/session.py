"""
ライブ文字起こしセッション

録音コールバックから受け取ったチャンクを、3つの daemon スレッドで処理する。

- segmenter: Framer → VAD → Segmenter で発話区間を切り出し、区間キューへ積む
- asr: engine.load() を実行した後、区間を順に文字起こしし、幻覚を除いて on_segment で通知
  (参考情報から作った用語ヒントを initial_prompt として渡し、ヒントの読み上げも除く)
- summary: RollingSummarizer で要約を更新する(文字起こしが遅れている間は保留)

ライブのテキストは best-effort であり、正となるのは WAV ファイルである。
どのスレッドで例外が起きてもセッションは止めず、on_warning で通知する。
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from live.asr_hint import HINT_MIN_SEGMENT_SEC, build_asr_hint, is_hint_echo
from live.chunk_queue import BoundedChunkQueue
from live.context import LiveContext
from live.framer import Framer
from live.full_summary import summarize_full
from live.hallucination import is_hallucination
from live.segment_queue import DropOldestQueue
from live.segmenter import Segment, Segmenter
from live.session_models import LiveCallbacks, LiveResult, LiveSegment, LiveSettings
from live.summarizer import RollingSummarizer

if TYPE_CHECKING:
    from live.engines import SpeechEngine
    from live.local_llm import ChatBackend
    from live.vad import VoiceActivityDetector

__all__ = ["LiveCallbacks", "LiveResult", "LiveSegment", "LiveSession", "LiveSettings"]

_logger = logging.getLogger("voicenote")

# 停止時の残り時間のうち、全文要約に使う割合(残りはローリング要約のフォールバック用)
_FULL_SUMMARY_TIME_SHARE = 0.7


def _safe_call(fn: Callable, *args) -> None:
    """コールバックを呼ぶ。UI 側の例外でワーカースレッドを殺さない"""
    try:
        fn(*args)
    except Exception:
        _logger.exception("ライブ文字起こしのコールバックでエラーが発生しました")


class LiveSession:
    """ライブ文字起こしセッション。start() → feed()… → stop() の順に使う"""

    def __init__(
        self,
        settings: LiveSettings,
        engine: SpeechEngine,
        vad: VoiceActivityDetector,
        backend: ChatBackend | None,
        callbacks: LiveCallbacks = LiveCallbacks(),
        clock: Callable[[], float] = time.monotonic,
        context: LiveContext | None = None,
    ):
        """
        Args:
            settings: セッション設定
            engine: 音声認識エンジン(load() は asr スレッドで呼ぶ)
            vad: フレームごとの発話確率を返す検出器
            backend: 要約用のチャットLLM。None なら要約しない
            callbacks: 通知先
            clock: 要約の実行間隔判定に使う時刻関数(stop の timeout には使わない)
            context: 要約と用語ヒントに使う初期の参考情報(参加者・メモ)。None なら参考情報なし
        """
        _validate_context(context)
        self._settings = settings
        self._engine = engine
        self._vad = vad
        self._callbacks = callbacks
        self._chunks = BoundedChunkQueue(settings.chunk_queue_size)
        self._segments: DropOldestQueue[Segment] = DropOldestQueue(settings.segment_queue_size)
        self._backend = backend
        self._summarizer = self._create_summarizer(backend, clock, context)
        self._lock = threading.Lock()
        # 参考情報と、そこから作った用語ヒント(ヒント文はログに出さない)。_lock 下で差し替える
        self._context = context
        self._asr_hint = build_asr_hint(context)
        self._backlog_samples = 0  # 整数で数え、浮動小数の誤差を溜めない
        self._dropped_segments = 0
        self._results: tuple[LiveSegment, ...] = ()
        # 破棄したが時刻にまだ反映していないサンプル数。feed(録音コールバック)だけが読み書きする
        self._pending_gap_samples = 0
        self._closing = threading.Event()  # feed 停止 + segmenter に残りの処理を促す
        self._segmenter_done = threading.Event()
        self._abort_asr = threading.Event()  # stop の timeout 超過時に asr を打ち切る
        self._summary_stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._result: LiveResult | None = None

    def _create_summarizer(self, backend, clock, context) -> RollingSummarizer | None:
        if backend is None:
            return None
        return RollingSummarizer(
            backend,
            self._settings.llm_model,
            interval_sec=self._settings.summary_interval_sec,
            max_backlog_sec=self._settings.max_backlog_sec,
            clock=clock,
            on_warning=lambda msg: _safe_call(self._callbacks.on_warning, msg),
            context=context,
        )

    # ------------------------------------------------------------ 公開API

    @property
    def backlog_sec(self) -> float:
        """文字起こし待ち(処理中を含む)の音声の長さ(秒)"""
        with self._lock:
            return self._backlog_samples / self._settings.sample_rate

    @property
    def dropped_chunks(self) -> int:
        """チャンクキューが満杯のため破棄された録音チャンク数"""
        return self._chunks.dropped

    def set_context(self, context: LiveContext | None) -> None:
        """参考情報(参加者・メモ)を差し替える。スレッドセーフ

        用語ヒントは要約の有無に関係なく作り直し、次に文字起こしする区間から使う。
        要約には次回の要約から反映する(参考情報の変更だけでは要約を起動しない)。
        """
        _validate_context(context)
        hint = build_asr_hint(context)
        with self._lock:
            self._context = context
            self._asr_hint = hint
        # 要約器のロックと入れ子にしない
        if self._summarizer is not None:
            self._summarizer.set_context(context)

    def start(self) -> None:
        """ワーカースレッドを起動する。2回目以降は RuntimeError"""
        if self._threads:
            raise RuntimeError("LiveSession は既に開始されています")
        targets = {"live-segmenter": self._segmenter_loop, "live-asr": self._asr_loop}
        if self._summarizer is not None:
            targets["live-summary"] = self._summary_loop
        self._threads = {
            name: threading.Thread(target=fn, name=name, daemon=True)
            for name, fn in targets.items()
        }
        for thread in self._threads.values():
            thread.start()

    def feed(self, chunk: np.ndarray) -> None:
        """
        録音チャンクを受け取る(chunk_sink)。キューに積むだけで即座に戻る。

        満杯で破棄したチャンクのサンプル数は累計し、次に積めたチャンクに
        (直前の欠損サンプル数, チャンク) として添える。キューが満杯のときに別のマーカーを
        積むことはできないため、欠損はチャンクと同じ要素で運ぶ。
        """
        if self._closing.is_set():
            return
        gap = self._pending_gap_samples
        if self._chunks.put_nowait((gap, chunk)):
            self._pending_gap_samples = 0
        else:
            self._pending_gap_samples = gap + len(chunk)

    def stop(self, timeout: float = 30.0, rebuild_summary: bool = True) -> LiveResult:
        """
        セッションを終了して結果を返す。全体で最大 timeout 秒(実時間)で戻る。

        1. 新規チャンクの受付を止める
        2. 残りのチャンクを区間化し、開いている区間を flush する
        3. 区間キューを文字起こしし切る(ドレイン)
        4. 全文から最終要約を作り直す。失敗・時間切れならローリング要約を1回試み、
           それも駄目なら直前の要約を使う

        rebuild_summary=False なら 4 の全文要約を省き、ローリング要約だけで締める
        (停止後に高精度の再文字起こしを予定し、LLM とメモリを奪い合わないようにする場合)。
        """
        if self._result is not None:
            return self._result
        deadline = time.monotonic() + timeout
        self._closing.set()
        if not self._threads:
            self._segmenter_done.set()
        complete = self._drain(deadline)
        final_summary = self._finish_summary(deadline, rebuild_summary)
        with self._lock:
            self._result = LiveResult(
                segments=self._results,
                final_summary=final_summary,
                dropped_chunks=self._chunks.dropped,
                dropped_segments=self._dropped_segments,
                complete=complete,
            )
        return self._result

    def wait_asr_idle(self, timeout: float) -> bool:
        """asr スレッドの終了を最大 timeout 秒待つ。終わっていれば(未開始も)True

        stop が時間切れでも asr スレッドは処理中の区間を終えるまで動くため、エンジンを
        解放する前にこれで待つ。
        """
        return self._join("live-asr", time.monotonic() + timeout)

    # ------------------------------------------------------------ 停止処理

    def _join(self, name: str, deadline: float) -> bool:
        """スレッドを deadline まで待つ。終了していれば True"""
        thread = self._threads.get(name)
        if thread is None:
            return True
        thread.join(max(0.0, deadline - time.monotonic()))
        return not thread.is_alive()

    def _drain(self, deadline: float) -> bool:
        """segmenter → asr の順に終了を待つ。時間内に処理しきれなければ False"""
        done = self._join("live-segmenter", deadline) and self._join("live-asr", deadline)
        if not done:
            self._abort_asr.set()
            self._warn("時間内に文字起こしできなかった区間があります（WAVファイルは保存済みです）")
        return done

    def _finish_summary(self, deadline: float, rebuild_summary: bool = True) -> str:
        """周期要約を止め、最終要約を timeout 付きで作る(全文要約 → ローリング要約の順に試す)"""
        summarizer = self._summarizer
        if summarizer is None:
            return ""
        self._summary_stop.set()
        if not self._join("live-summary", deadline):
            return summarizer.summary
        before = summarizer.summary
        full = self._summarize_full(deadline) if rebuild_summary else None
        if full:
            if full != before:
                _safe_call(self._callbacks.on_summary, full)
            return full
        return self._rolling_final_summary(summarizer, before, deadline)

    def _summarize_full(self, deadline: float) -> str | None:
        """全セグメントの文字起こしから要約を作り直す。残り時間の 70% までで諦める"""
        with self._lock:
            texts = tuple(seg.text for seg in self._results)
            context = self._context
        if self._backend is None or not texts:
            return None
        now = time.monotonic()
        full_deadline = now + max(0.0, deadline - now) * _FULL_SUMMARY_TIME_SHARE
        outcome: list[str | None] = []
        worker = threading.Thread(
            target=lambda: outcome.append(
                summarize_full(
                    self._backend,
                    texts,
                    context,
                    deadline=full_deadline,
                    clock=time.monotonic,
                    cancel=self._abort_asr,
                )
            ),
            name="live-full-summary",
            daemon=True,
        )
        worker.start()
        worker.join(max(0.0, full_deadline - time.monotonic()))
        if worker.is_alive() or not outcome:
            self._warn("全文からの最終要約が時間内に終わらなかったため、直前の要約を使います")
            return None
        return outcome[0]

    def _rolling_final_summary(self, summarizer, before: str, deadline: float) -> str:
        """ローリング要約を強制実行して最終要約とする。時間切れなら直前の要約"""
        # interval 判定を無視して強制実行するため now=inf を渡す(送信中なら何もしない)
        worker = threading.Thread(
            target=lambda: summarizer.maybe_run(now=math.inf, backlog_sec=0.0),
            name="live-final-summary",
            daemon=True,
        )
        worker.start()
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            self._warn("最終要約が時間内に終わらなかったため、直前の要約を使います")
            return before
        summary = summarizer.summary
        if summary != before:
            _safe_call(self._callbacks.on_summary, summary)
        return summary

    # ------------------------------------------------------------ ワーカー

    def _warn(self, message: str) -> None:
        _logger.warning(message)
        _safe_call(self._callbacks.on_warning, message)

    def _segmenter_loop(self) -> None:
        """チャンクを固定長フレーム→発話区間に変換して区間キューへ積む"""
        framer = Framer()
        segmenter = Segmenter(sample_rate=self._settings.sample_rate, frame_size=framer.frame_size)
        warned = False
        try:
            while True:
                item = self._chunks.get(timeout=self._settings.poll_sec)
                if item is None:
                    if self._closing.is_set():
                        break
                    continue
                gap, chunk = item
                try:
                    # Framer の端数(1フレーム未満, 約32ms)は欠損の後ろに回るため、その分だけ
                    # 誤差が残るが、以降の新しいチャンクの時刻は WAV と一致する
                    if gap:
                        self._enqueue_all(segmenter.skip(gap))
                    for frame in framer.push(chunk):
                        prob = self._vad.speech_prob(frame)
                        self._enqueue_all(segmenter.push(frame, prob))
                except Exception as e:
                    _logger.exception("音声区間の検出に失敗しました")
                    if not warned:
                        warned = True
                        _safe_call(self._callbacks.on_warning, f"音声区間の検出に失敗しました: {e}")
            self._enqueue_all(segmenter.flush())
        finally:
            self._segmenter_done.set()

    def _enqueue_all(self, segments: list[Segment]) -> None:
        for segment in segments:
            self._enqueue(segment)

    def _enqueue(self, segment: Segment) -> None:
        """区間を積む。満杯なら最古を破棄して警告する"""
        # asr スレッドが取り出して減算するより先に加算しておく(順序が逆だと値がずれる)
        with self._lock:
            self._backlog_samples += len(segment.samples)
        evicted = self._segments.put(segment)
        if evicted is not None:
            with self._lock:
                self._backlog_samples -= len(evicted.samples)
                self._dropped_segments += 1
        if evicted is not None:
            self._warn(
                f"文字起こしが追いつかないため区間 {evicted.start_sec:.1f}s〜 を破棄しました"
            )

    def _asr_loop(self) -> None:
        """エンジンを読み込み、区間を順に文字起こしする"""
        loaded = self._load_engine()
        while not self._abort_asr.is_set():
            segment = self._segments.get(timeout=self._settings.poll_sec)
            if segment is None:
                if self._segmenter_done.is_set() and len(self._segments) == 0:
                    break
                continue
            try:
                if loaded:
                    self._transcribe(segment)
            finally:
                with self._lock:
                    self._backlog_samples -= len(segment.samples)

    def _load_engine(self) -> bool:
        _safe_call(self._callbacks.on_status, "音声認識モデルを読み込んでいます...")
        try:
            self._engine.load()
        except Exception as e:
            _logger.exception("音声認識エンジンの読み込みに失敗しました")
            self._warn(f"音声認識エンジンの読み込みに失敗しました（録音は継続します）: {e}")
            return False
        _safe_call(self._callbacks.on_status, "ライブ文字起こし中")
        return True

    def _hint_for(self, segment: Segment) -> str | None:
        """区間に付ける用語ヒント。短い区間(反響しやすい)には付けない"""
        if len(segment.samples) < HINT_MIN_SEGMENT_SEC * self._settings.sample_rate:
            return None
        with self._lock:
            return self._asr_hint

    def _transcribe(self, segment: Segment) -> None:
        """1区間を文字起こしし、幻覚・ヒントの反響でなければ結果に追加して通知する"""
        # ヒントは区間ごとに1回だけ読み、エンジンへの指定と反響判定に同じ値を使う
        hint = self._hint_for(segment)
        try:
            result = self._engine.transcribe(segment.samples, self._settings.language, prompt=hint)
        except Exception as e:
            _logger.exception("区間の文字起こしに失敗しました")
            self._warn(f"区間 {segment.start_sec:.1f}s の文字起こしに失敗しました: {e}")
            return
        text = (getattr(result, "text", "") or "").strip()
        if is_hallucination(
            text,
            no_speech_prob=getattr(result, "no_speech_prob", None),
            avg_logprob=getattr(result, "avg_logprob", None),
        ) or is_hint_echo(text, hint):
            return
        with self._lock:
            live_segment = LiveSegment(len(self._results), segment.start_sec, segment.end_sec, text)
            self._results = (*self._results, live_segment)
        _safe_call(self._callbacks.on_segment, *_segment_args(live_segment))
        if self._summarizer is not None:
            self._summarizer.add_text(text)

    def _summary_loop(self) -> None:
        """一定間隔で要約の実行条件を確認する(条件判定は RollingSummarizer に委ねる)"""
        summarizer = self._summarizer
        while not self._summary_stop.wait(self._settings.summary_poll_sec):
            if summarizer.maybe_run(backlog_sec=self.backlog_sec):
                _safe_call(self._callbacks.on_summary, summarizer.summary)


def _validate_context(context) -> None:
    """context が None か LiveContext でなければ TypeError"""
    if context is not None and not isinstance(context, LiveContext):
        raise TypeError(f"context は LiveContext が必要です: {type(context).__name__}")


def _segment_args(segment: LiveSegment) -> tuple[int, float, float, str]:
    return (segment.index, segment.start_sec, segment.end_sec, segment.text)
