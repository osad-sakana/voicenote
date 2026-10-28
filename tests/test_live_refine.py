"""live.refine(停止後の高精度再文字起こし)のテスト。実モデルは使わず Fake エンジンで検証する"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from live.engines import TimedText
from live.refine import (
    REFINE_LOAD_GRACE_SEC,
    REFINE_MAX_RTF,
    RefineDecision,
    decide_refine,
    format_refine_progress,
    load_wav_mmap,
    plan_windows,
    refine_wav,
    rms_reader,
)
from live.session_models import LiveSettings

SR = 16000


def _write_wav(path: Path, samples: np.ndarray, rate=SR, channels=1, width=2) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def _tone(seconds: float, amplitude=8000) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n)
    return (amplitude * np.sin(2 * np.pi * 440 * t / SR)).astype(np.int16)


class FakeRefineEngine:
    """transcribe_timed / load / unload を記録する偽エンジン"""

    def __init__(self, log, outputs=None, fail_at=None, gate=None):
        self.log = log
        self.outputs = list(outputs or [])
        self.fail_at = fail_at
        self.gate = gate
        self.calls: list[tuple[int, str | None, str | None]] = []

    def load(self):
        self.log.append("refine.load")

    def transcribe_timed(self, samples, language, prompt=None):
        index = len(self.calls)
        self.calls.append((len(samples), language, prompt))
        self.log.append(f"refine.window{index}")
        assert samples.dtype == np.float32
        if self.gate is not None:
            self.gate(index)
        if self.fail_at == index:
            raise RuntimeError("推論失敗")
        if self.outputs:
            return self.outputs.pop(0)
        return (TimedText(0.0, 1.0, f"窓{index}の発話です"),)

    def unload(self):
        self.log.append("refine.unload")


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def _run(wav, engine, log, **kwargs):
    defaults = dict(
        language="ja",
        hint=None,
        cancel=threading.Event(),
        on_progress=lambda *_: None,
        clock=Clock(),
    )
    return refine_wav(
        wav,
        make_engine=lambda: log.append("make_engine") or engine,
        release_live=lambda: log.append("release_live"),
        **{**defaults, **kwargs},
    )


# ──────────────── decide_refine ────────────────


class TestDecideRefine:
    def test_disabled_does_not_run_and_has_no_reason(self):
        decision = decide_refine(LiveSettings(engine="mlx"), "mlx", True)
        assert decision == RefineDecision(False, "")

    def test_runs_with_mlx_and_cached_model(self):
        settings = LiveSettings(engine="mlx", refine_enabled=True)
        assert decide_refine(settings, "mlx", True).run is True

    def test_skips_when_effective_engine_is_faster_whisper(self):
        settings = LiveSettings(engine="mlx", refine_enabled=True)
        decision = decide_refine(settings, "faster-whisper", True)
        assert decision.run is False
        assert "faster-whisper" in decision.reason

    def test_skips_when_model_not_cached(self):
        settings = LiveSettings(engine="mlx", refine_enabled=True, refine_model="large-v3")
        decision = decide_refine(settings, "mlx", False)
        assert decision.run is False
        assert "large-v3" in decision.reason


# ──────────────── plan_windows ────────────────


class TestPlanWindows:
    def test_empty_audio_has_no_window(self):
        assert plan_windows(0, lambda s, e: 1.0, SR) == ()

    def test_short_audio_is_single_window(self):
        assert plan_windows(SR * 30, lambda s, e: 1.0, SR) == ((0, SR * 30),)

    def test_up_to_target_plus_search_is_single_window(self):
        n = SR * 65
        assert plan_windows(n, lambda s, e: 1.0, SR) == ((0, n),)

    def test_cuts_at_quietest_position_near_target(self):
        audio = _tone(130.0)
        quiet = int(57.3 * SR)
        audio[quiet : quiet + SR // 10] = 0  # 57.3〜57.4秒が無音
        windows = plan_windows(len(audio), rms_reader(audio), SR)
        cut = windows[0][1]
        assert abs(cut / SR - 57.35) <= 0.1
        assert windows[0][0] == 0 and windows[-1][1] == len(audio)
        for (_, end), (start, _) in zip(windows, windows[1:], strict=False):
            assert end == start  # 隙間も重なりもない

    def test_uniform_audio_cuts_near_target(self):
        n = SR * 200
        windows = plan_windows(n, lambda s, e: 1.0, SR)
        assert abs(windows[0][1] / SR - 60.0) <= 0.1
        assert all((e - s) <= 65 * SR for s, e in windows)
        assert windows[-1][1] == n

    def test_searches_only_within_range(self):
        def read_rms(start, end):
            return 0.0 if start < 50 * SR else 1.0  # 範囲外は静かでも選ばない

        windows = plan_windows(SR * 130, read_rms, SR)
        assert windows[0][1] >= 55 * SR


# ──────────────── load_wav_mmap ────────────────


class TestLoadWavMmap:
    def test_reads_16k_mono_int16(self, tmp_path):
        audio = _tone(1.0)
        samples = load_wav_mmap(_write_wav(tmp_path / "a.wav", audio))
        assert samples.dtype == np.int16
        np.testing.assert_array_equal(np.asarray(samples), audio)

    @pytest.mark.parametrize(
        "kwargs", [{"rate": 44100}, {"channels": 2}, {"width": 1}], ids=["rate", "stereo", "8bit"]
    )
    def test_rejects_other_formats(self, tmp_path, kwargs):
        width = kwargs.get("width", 2)
        data = np.zeros(1600, dtype=np.int16 if width == 2 else np.uint8)
        path = _write_wav(tmp_path / "b.wav", data, **kwargs)
        with pytest.raises(ValueError):
            load_wav_mmap(path)

    def test_rejects_empty_data(self, tmp_path):
        path = _write_wav(tmp_path / "c.wav", np.zeros(0, dtype=np.int16))
        with pytest.raises(ValueError):
            load_wav_mmap(path)

    def test_rejects_non_wav(self, tmp_path):
        path = tmp_path / "d.wav"
        path.write_bytes(b"not a wav file at all")
        with pytest.raises(ValueError):
            load_wav_mmap(path)

    def test_clamps_to_actual_file_size(self, tmp_path):
        path = _write_wav(tmp_path / "e.wav", _tone(1.0))
        raw = bytearray(path.read_bytes())
        raw[40:44] = (10**8).to_bytes(4, "little")  # data サイズが実際より大きい(中断時)
        path.write_bytes(bytes(raw))
        assert len(load_wav_mmap(path)) == SR

    def test_skips_unknown_chunks(self, tmp_path):
        path = _write_wav(tmp_path / "f.wav", _tone(0.5))
        raw = path.read_bytes()
        extra = b"LIST" + (3).to_bytes(4, "little") + b"abc\x00"  # 奇数長は詰め物付き
        path.write_bytes(raw[:36] + extra + raw[36:])
        assert len(load_wav_mmap(path)) == SR // 2


# ──────────────── refine_wav ────────────────


class TestRefineWav:
    def test_releases_live_engine_before_loading_and_unloads_at_end(self, tmp_path):
        log: list[str] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(110.0))
        result = _run(wav, FakeRefineEngine(log), log)
        assert log[:3] == ["release_live", "make_engine", "refine.load"]
        assert log[-1] == "refine.unload"
        assert log.count("refine.unload") == 1
        assert result.complete is True
        assert len(result.segments) == 2

    def test_offsets_segment_times_by_window_start(self, tmp_path):
        log: list[str] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(110.0))
        engine = FakeRefineEngine(
            log,
            outputs=[
                (TimedText(1.0, 2.0, "最初の窓です"),),
                (TimedText(3.0, 4.5, "二つ目の窓です"),),
            ],
        )
        result = _run(wav, engine, log)
        first, second = result.segments
        assert (first.index, first.start_sec, first.end_sec) == (0, 1.0, 2.0)
        assert second.index == 1
        window_start = engine.calls[0][0] / SR
        assert second.start_sec == pytest.approx(window_start + 3.0)
        assert second.end_sec == pytest.approx(window_start + 4.5)

    def test_drops_hallucinations_and_hint_echo(self, tmp_path):
        log: list[str] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(10.0))
        hint = "用語: 量子計算、超伝導。参加者: 山田、佐藤。"
        engine = FakeRefineEngine(
            log,
            outputs=[
                (
                    TimedText(0.0, 1.0, "ご視聴ありがとうございました"),
                    TimedText(1.0, 2.0, "量子計算、超伝導"),
                    TimedText(2.0, 3.0, "無音", no_speech_prob=0.9, avg_logprob=-1.5),
                    TimedText(3.0, 4.0, "本題に入ります"),
                ),
            ],
        )
        result = _run(wav, engine, log, hint=hint)
        assert [s.text for s in result.segments] == ["本題に入ります"]
        assert engine.calls[0][2] == hint
        assert engine.calls[0][1] == "ja"

    def test_short_audio_gets_no_hint(self, tmp_path):
        log: list[str] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(0.5))
        engine = FakeRefineEngine(log)
        _run(wav, engine, log, hint="参加者: 山田。")
        assert engine.calls[0][2] is None

    def test_cancel_between_windows_stops_and_unloads(self, tmp_path):
        log: list[str] = []
        cancel = threading.Event()
        wav = _write_wav(tmp_path / "r.wav", _tone(200.0))
        engine = FakeRefineEngine(log, gate=lambda i: cancel.set())
        result = _run(wav, engine, log, cancel=cancel)
        assert len(engine.calls) == 1
        assert result.complete is False
        assert log[-1] == "refine.unload"

    def test_cancel_before_start_skips_loading(self, tmp_path):
        log: list[str] = []
        cancel = threading.Event()
        cancel.set()
        wav = _write_wav(tmp_path / "r.wav", _tone(5.0))
        result = _run(wav, FakeRefineEngine(log), log, cancel=cancel)
        assert result.complete is False
        assert "make_engine" not in log

    def test_deadline_stops_between_windows(self, tmp_path):
        log: list[str] = []
        clock = Clock()
        wav = _write_wav(tmp_path / "r.wav", _tone(200.0))

        def advance(_index):
            clock.now += 1000.0

        engine = FakeRefineEngine(log, gate=advance)
        result = _run(wav, engine, log, clock=clock, deadline=500.0)
        assert len(engine.calls) == 1
        assert result.complete is False
        assert log[-1] == "refine.unload"

    def test_default_deadline_from_recording_length(self, tmp_path):
        log: list[str] = []
        clock = Clock()
        wav = _write_wav(tmp_path / "r.wav", _tone(110.0))
        limit = 110.0 * REFINE_MAX_RTF + REFINE_LOAD_GRACE_SEC

        def advance(_index):
            clock.now += limit / 2 + 1  # 2窓目の後で超過、3窓目は無い

        engine = FakeRefineEngine(log, gate=advance)
        result = _run(wav, engine, log, clock=clock)
        assert len(engine.calls) == 2
        assert result.complete is True

    def test_window_error_propagates_after_unload(self, tmp_path):
        log: list[str] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(5.0))
        with pytest.raises(RuntimeError):
            _run(wav, FakeRefineEngine(log, fail_at=0), log)
        assert log[-1] == "refine.unload"

    def test_invalid_wav_raises_before_release(self, tmp_path):
        log: list[str] = []
        path = _write_wav(tmp_path / "r.wav", np.zeros(100, dtype=np.int16), rate=44100)
        with pytest.raises(ValueError):
            _run(path, FakeRefineEngine(log), log)
        assert log == []  # リアルタイム用エンジンも解放しない

    def test_progress_reports_done_total_and_eta(self, tmp_path):
        log: list[str] = []
        clock = Clock()
        reports: list[tuple[float, float, float | None]] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(130.0))

        def advance(_index):
            clock.now += 10.0

        _run(
            wav,
            FakeRefineEngine(log, gate=advance),
            log,
            clock=clock,
            on_progress=lambda *args: reports.append(args),
        )
        assert reports[0] == (0.0, pytest.approx(130.0), None)
        assert reports[-1][0] == pytest.approx(130.0)
        assert reports[-1][2] == pytest.approx(0.0)
        assert reports[1][2] is not None and reports[1][2] > 0

    def test_progress_callback_errors_are_ignored(self, tmp_path):
        log: list[str] = []
        wav = _write_wav(tmp_path / "r.wav", _tone(5.0))

        def broken(*_args):
            raise RuntimeError("UI")

        result = _run(wav, FakeRefineEngine(log), log, on_progress=broken)
        assert result.complete is True

    def test_elapsed_uses_clock(self, tmp_path):
        log: list[str] = []
        clock = Clock(10.0)
        wav = _write_wav(tmp_path / "r.wav", _tone(5.0))

        def advance(_index):
            clock.now += 3.0

        result = _run(wav, FakeRefineEngine(log, gate=advance), log, clock=clock)
        assert result.elapsed == pytest.approx(3.0)


# ──────────────── format_refine_progress ────────────────


class TestFormatRefineProgress:
    def test_with_eta(self):
        text = format_refine_progress(120.0, 600.0, 480.0)
        assert "02:00 / 10:00" in text
        assert "20%" in text
        assert "残り約8分" in text

    def test_without_eta(self):
        text = format_refine_progress(0.0, 600.0, None)
        assert "0%" in text
        assert "残り" not in text

    def test_short_eta_in_seconds(self):
        assert "残り約30秒" in format_refine_progress(90.0, 120.0, 30.0)

    def test_zero_total_is_safe(self):
        assert "0%" in format_refine_progress(0.0, 0.0, None)
