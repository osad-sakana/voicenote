"""live.vad のテスト"""

import numpy as np
import pytest

import live.vad as vad_module
from live.vad import EnergyVAD, create_vad, vad_kind

SAMPLE_RATE = 16000
FRAME = 512


def _sine(amplitude: float = 0.3, freq: float = 220.0, n: int = FRAME) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(n: int = FRAME) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


class TestEnergyVAD:
    def test_silence_lower_than_sine(self):
        """無音の確率はサイン波より低い"""
        vad = EnergyVAD()

        assert vad.speech_prob(_silence()) < vad.speech_prob(_sine())

    def test_probabilities_are_in_unit_range(self):
        """確率は 0..1 の float"""
        vad = EnergyVAD()

        for frame in (_silence(), _sine(), np.ones(FRAME, dtype=np.float32)):
            p = vad.speech_prob(frame)
            assert isinstance(p, float)
            assert 0.0 <= p <= 1.0

    def test_silence_below_half_and_loud_above_half(self):
        """無音は 0.5 未満、十分大きい音は 0.5 超"""
        vad = EnergyVAD(threshold_db=-40.0)

        assert vad.speech_prob(_silence()) < 0.5
        assert vad.speech_prob(_sine(amplitude=0.3)) > 0.5

    def test_threshold_shifts_decision(self):
        """閾値を上げると同じ音でも確率が下がる"""
        quiet = _sine(amplitude=0.01)

        assert EnergyVAD(threshold_db=-60.0).speech_prob(quiet) > EnergyVAD(
            threshold_db=-20.0
        ).speech_prob(quiet)

    def test_empty_frame_returns_zero(self):
        """空フレームは 0.0"""
        assert EnergyVAD().speech_prob(np.zeros(0, dtype=np.float32)) == 0.0


class TestCreateVad:
    def test_energy(self):
        """'energy' で EnergyVAD を返す"""
        assert isinstance(create_vad("energy"), EnergyVAD)

    def test_unknown_name_raises(self):
        """未知の名前は ValueError"""
        with pytest.raises(ValueError):
            create_vad("unknown")

    def test_silero_failure_falls_back_to_energy(self, monkeypatch):
        """Silero の初期化に失敗したら EnergyVAD にフォールバック"""

        def broken(*args, **kwargs):
            raise RuntimeError("onnxruntime missing")

        monkeypatch.setattr(vad_module, "SileroVAD", broken)

        assert isinstance(create_vad("silero"), EnergyVAD)


class TestVadKind:
    def test_energy(self):
        assert vad_kind(EnergyVAD()) == "energy"

    def test_silero(self):
        class FakeModel:
            def __call__(self, audio):
                return np.zeros(len(audio) // 512)

        assert vad_kind(vad_module.SileroVAD(model=FakeModel())) == "silero"

    def test_unknown_detector_is_none(self):
        class Custom:
            def speech_prob(self, frame):
                return 0.0

        assert vad_kind(Custom()) is None

    def test_fallback_is_reported_as_energy(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("onnxruntime missing")

        monkeypatch.setattr(vad_module, "SileroVAD", broken)
        assert vad_kind(create_vad("silero")) == "energy"


class TestSileroVAD:
    def test_real_model(self):
        """同梱の silero_vad_v6.onnx で推論できる(入力非破壊、無音は低確率)"""
        pytest.importorskip("onnxruntime")
        pytest.importorskip("faster_whisper")
        from live.vad import SileroVAD

        vad = SileroVAD()
        frame = _sine()
        original = frame.copy()

        p_sine = vad.speech_prob(frame)
        p_silence = None
        for _ in range(20):
            p_silence = vad.speech_prob(_silence())

        assert isinstance(p_sine, float)
        assert 0.0 <= p_sine <= 1.0
        assert p_silence < 0.5
        np.testing.assert_array_equal(frame, original)
        assert isinstance(create_vad("silero"), SileroVAD)

    def test_rejects_wrong_frame_length(self):
        """512 以外のフレーム長は ValueError"""
        pytest.importorskip("onnxruntime")
        pytest.importorskip("faster_whisper")
        from live.vad import SileroVAD

        with pytest.raises(ValueError):
            SileroVAD().speech_prob(np.zeros(100, dtype=np.float32))

    def test_uses_rolling_window_and_last_probability(self):
        """固定長のローリング窓+末尾ゼロ埋め1フレームを渡し、最新フレームの確率を返す"""
        from live.vad import SileroVAD

        calls = []

        class FakeModel:
            def __call__(self, audio):
                calls.append(audio.copy())
                audio[:] = 0.0  # 本家同様に入力を書き換えても影響しないこと
                n = len(audio) // FRAME
                return np.arange(n, dtype=np.float32) / 10.0

        vad = SileroVAD(model=FakeModel(), window_frames=4)
        frames = [np.full(FRAME, i + 1, dtype=np.float32) for i in range(6)]
        probs = [vad.speech_prob(f) for f in frames]

        assert all(len(c) == 5 * FRAME for c in calls)
        # 出力は [0.0, 0.1, 0.2, 0.3, 0.4]。最新フレーム(末尾から2番目)は 0.3
        assert probs[-1] == pytest.approx(0.3)
        # 末尾はモデルの in-place 書き換え対策のゼロ埋めフレーム
        np.testing.assert_array_equal(calls[-1][-FRAME:], 0.0)
        # 最新フレームがその直前、直近4フレームのみ保持
        np.testing.assert_array_equal(calls[-1][-2 * FRAME : -FRAME], frames[-1])
        np.testing.assert_array_equal(calls[-1][:FRAME], frames[2])
        # 窓が埋まるまでは先頭をゼロ埋め
        np.testing.assert_array_equal(calls[0][: 3 * FRAME], 0.0)
        assert frames[-1][0] == 6.0
