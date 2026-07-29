"""pipeline モジュールのユニットテスト (純粋ロジック部分のみ)。"""

from pathlib import Path

import pytest

import pipeline
from config import InvalidConfigError, VoiceNoteConfig
from pipeline import load_or_configure


class TestLoadOrConfigure:
    def test_propagates_invalid_config_error_without_exiting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        broken = tmp_path / "config.json"
        broken.write_text("{ not valid json", encoding="utf-8")
        monkeypatch.setattr(pipeline, "CONFIG_PATH", broken)

        with pytest.raises(InvalidConfigError):
            load_or_configure(force_config=False)

    def test_propagates_save_failure_without_exiting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # 保存先の親を既存ファイルにして save_config を失敗させる
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        broken_config_path = blocker / "child" / "config.json"
        monkeypatch.setattr(pipeline, "CONFIG_PATH", broken_config_path)
        monkeypatch.setattr(
            pipeline, "configure_interactive", lambda: VoiceNoteConfig(save_folder="/tmp/x")
        )

        with pytest.raises(RuntimeError):
            load_or_configure(force_config=True)
