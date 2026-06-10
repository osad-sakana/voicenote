"""pipeline モジュールのユニットテスト (純粋ロジック部分のみ)。"""

import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pipeline import save_wav
from recorder import SAMPLE_RATE


class TestSaveWav:
    def test_creates_dest_dir_if_missing(self, tmp_path: Path):
        target = tmp_path / "nested" / "out"
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        save_wav(audio, target)
        assert target.is_dir()

    def test_filename_matches_recording_pattern(self, tmp_path: Path):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        saved = save_wav(audio, tmp_path)
        assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}_recording\.wav$", saved.name)

    def test_writes_at_target_sample_rate(self, tmp_path: Path):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        saved = save_wav(audio, tmp_path)
        rate, _ = wavfile.read(str(saved))
        assert rate == SAMPLE_RATE

    def test_converts_float32_to_int16(self, tmp_path: Path):
        # -1.0〜1.0 の float32 が int16 (-32768〜32767) に変換される
        audio = np.array([0.0, 0.5, 1.0, -1.0, -0.5], dtype=np.float32)
        saved = save_wav(audio, tmp_path)
        _, data = wavfile.read(str(saved))
        assert data.dtype == np.int16
        # 0.5 * 32767 ≈ 16383, -1.0 * 32767 = -32767
        assert data[0] == 0
        assert abs(int(data[1]) - 16383) <= 1
        assert int(data[2]) == 32767
        assert int(data[3]) == -32767

    def test_preserves_sample_count(self, tmp_path: Path):
        n_samples = SAMPLE_RATE * 2  # 2秒分
        audio = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
        saved = save_wav(audio, tmp_path)
        _, data = wavfile.read(str(saved))
        assert len(data) == n_samples

    def test_accepts_string_dest_dir(self, tmp_path: Path):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        saved = save_wav(audio, str(tmp_path))
        assert saved.exists()
