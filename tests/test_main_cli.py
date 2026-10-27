"""main_cli のコマンドライン動作のテスト"""

import sys
from unittest.mock import MagicMock

import pytest

import main_cli
from config import VoiceNoteConfig


@pytest.fixture
def stubbed(monkeypatch):
    """設定読み込み・ログ・録音・文字起こしを差し替えて呼び出しを記録する"""
    config = VoiceNoteConfig(save_folder="/tmp/x")
    load = MagicMock(return_value=config)
    record = MagicMock()
    transcribe = MagicMock()
    monkeypatch.setattr(main_cli, "load_or_configure", load)
    monkeypatch.setattr(main_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(main_cli, "record_audio", record)
    monkeypatch.setattr(main_cli, "_run_transcription", transcribe)
    return load, record, transcribe


class TestConfigFlag:
    def test_config_saves_settings_without_recording(self, monkeypatch, stubbed):
        load, record, transcribe = stubbed
        monkeypatch.setattr(sys, "argv", ["main_cli.py", "--config"])

        main_cli.main()

        load.assert_called_once_with(force_config=True)
        record.assert_not_called()
        transcribe.assert_not_called()

    def test_config_with_live_does_not_start_live(self, monkeypatch, stubbed):
        _, record, _ = stubbed
        run_live = MagicMock()
        monkeypatch.setitem(sys.modules, "live_cli", MagicMock(run_live=run_live))
        monkeypatch.setattr(sys, "argv", ["main_cli.py", "--config", "--live"])

        main_cli.main()

        run_live.assert_not_called()
        record.assert_not_called()

    def test_without_config_flag_recording_still_starts(self, monkeypatch, stubbed):
        load, record, _ = stubbed
        record.side_effect = RuntimeError("stop here")
        monkeypatch.setattr(sys, "argv", ["main_cli.py"])

        with pytest.raises(SystemExit):
            main_cli.main()

        load.assert_called_once_with(force_config=False)
        record.assert_called_once()
