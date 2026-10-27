"""live.level_meter のテスト"""

import logging
import math

import numpy as np
import pytest

from live.level_meter import (
    LOW_INPUT_THRESHOLD_DB,
    MIN_ASSESS_SAMPLES,
    SILENCE_FLOOR_DB,
    InputLevelAssessment,
    LevelMeter,
    assess_input_level,
    tee,
)


def _sine(amplitude: float, n: int = 1600) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / 16000.0
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32).reshape(-1, 1)


class TestLevelMeterFeed:
    def test_initial_level_is_floor(self):
        meter = LevelMeter()
        assert meter.level_db == SILENCE_FLOOR_DB
        assert meter.history_db == ()

    def test_full_scale_dc_is_zero_dbfs(self):
        meter = LevelMeter()
        meter.feed(np.ones((800, 1), dtype=np.float32))
        assert meter.level_db == pytest.approx(0.0, abs=1e-6)

    def test_half_amplitude_dc_is_minus_6db(self):
        meter = LevelMeter()
        meter.feed(np.full((800, 1), 0.5, dtype=np.float32))
        assert meter.level_db == pytest.approx(20 * math.log10(0.5), abs=1e-4)

    def test_sine_rms_is_amplitude_over_sqrt2(self):
        meter = LevelMeter()
        meter.feed(_sine(0.1, n=16000))
        expected = 20 * math.log10(0.1 / math.sqrt(2))
        assert meter.level_db == pytest.approx(expected, abs=0.05)

    def test_silence_is_clamped_to_floor_not_inf(self):
        meter = LevelMeter()
        meter.feed(np.zeros((800, 1), dtype=np.float32))
        assert meter.level_db == SILENCE_FLOOR_DB
        assert math.isfinite(meter.level_db)

    def test_empty_chunk_is_ignored(self):
        meter = LevelMeter()
        meter.feed(np.ones((800, 1), dtype=np.float32))
        meter.feed(np.zeros((0, 1), dtype=np.float32))
        assert meter.level_db == pytest.approx(0.0, abs=1e-6)
        assert len(meter.history_db) == 1

    def test_feed_does_not_modify_chunk(self):
        chunk = _sine(0.3)
        original = chunk.copy()
        LevelMeter().feed(chunk)
        np.testing.assert_array_equal(chunk, original)

    def test_latest_value_overwrites(self):
        meter = LevelMeter()
        meter.feed(np.ones((800, 1), dtype=np.float32))
        meter.feed(np.full((800, 1), 0.1, dtype=np.float32))
        assert meter.level_db == pytest.approx(-20.0, abs=1e-4)

    def test_history_is_bounded(self):
        meter = LevelMeter(history_size=3)
        for amp in (1.0, 0.5, 0.25, 0.1):
            meter.feed(np.full((10, 1), amp, dtype=np.float32))
        history = meter.history_db
        assert isinstance(history, tuple)
        assert len(history) == 3
        assert history[-1] == pytest.approx(-20.0, abs=1e-4)

    def test_history_snapshot_is_independent(self):
        meter = LevelMeter()
        meter.feed(np.ones((10, 1), dtype=np.float32))
        snapshot = meter.history_db
        meter.feed(np.ones((10, 1), dtype=np.float32))
        assert len(snapshot) == 1

    def test_invalid_history_size(self):
        with pytest.raises(ValueError):
            LevelMeter(history_size=0)


class TestAssessInputLevel:
    def test_insufficient_samples_is_pending(self):
        result = assess_input_level([-60.0] * (MIN_ASSESS_SAMPLES - 1))
        assert isinstance(result, InputLevelAssessment)
        assert result.status == "pending"
        assert result.is_low is False
        assert result.percentile_db is None

    def test_empty_is_pending(self):
        assert assess_input_level([]).status == "pending"

    def test_quiet_builtin_mic_is_low(self):
        # README実例相当: 発話でも -47dB 前後、無音区間は -70dB 程度
        history = [-47.0] * 20 + [-70.0] * 20
        result = assess_input_level(history)
        assert result.status == "low"
        assert result.is_low is True
        assert result.percentile_db == pytest.approx(-47.0)
        assert result.message

    def test_normal_voice_is_ok(self):
        history = [-30.0] * 20 + [-60.0] * 20
        result = assess_input_level(history)
        assert result.status == "ok"
        assert result.is_low is False
        assert result.message == ""

    def test_short_loud_burst_with_mostly_quiet_is_ok(self):
        # 発話は数秒のうち一部でも、上位10%が通常音量なら小さいとは判定しない
        history = [-70.0] * 30 + [-28.0] * 10
        assert assess_input_level(history).status == "ok"

    def test_complete_silence_is_low(self):
        history = [SILENCE_FLOOR_DB] * MIN_ASSESS_SAMPLES
        result = assess_input_level(history)
        assert result.status == "low"
        assert math.isfinite(result.percentile_db)

    def test_threshold_boundary_is_ok(self):
        history = [LOW_INPUT_THRESHOLD_DB] * MIN_ASSESS_SAMPLES
        assert assess_input_level(history).status == "ok"

    def test_custom_threshold_and_min_samples(self):
        history = [-30.0] * 5
        result = assess_input_level(history, threshold_db=-20.0, min_samples=5)
        assert result.status == "low"

    def test_accepts_tuple_from_meter(self):
        meter = LevelMeter()
        for _ in range(MIN_ASSESS_SAMPLES):
            meter.feed(_sine(0.1))
        assert assess_input_level(meter.history_db).status == "ok"


class TestTee:
    def test_calls_all_sinks_in_order(self):
        calls = []
        sink = tee(lambda c: calls.append(("a", c)), lambda c: calls.append(("b", c)))
        chunk = np.zeros((4, 1), dtype=np.float32)
        sink(chunk)
        assert [name for name, _ in calls] == ["a", "b"]
        assert all(c is chunk for _, c in calls)

    def test_exception_in_one_sink_does_not_stop_others(self, caplog):
        calls = []

        def broken(_chunk):
            raise RuntimeError("boom")

        sink = tee(broken, lambda c: calls.append(c))
        with caplog.at_level(logging.ERROR, logger="voicenote"):
            sink(np.zeros((4, 1), dtype=np.float32))
        assert len(calls) == 1
        assert "boom" in caplog.text

    def test_no_sinks_is_noop(self):
        tee()(np.zeros((4, 1), dtype=np.float32))

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError):
            tee(lambda c: None, "not callable")
