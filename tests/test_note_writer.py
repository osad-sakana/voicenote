"""note_writer モジュールのユニットテスト。"""

import re
from pathlib import Path

import pytest

from note_writer import save_live_note, save_transcript


class TestSaveTranscript:
    def test_creates_save_folder_if_missing(self, tmp_path: Path):
        target = tmp_path / "nested" / "notes"
        saved = save_transcript(target, "本文", format_mode="none")
        assert saved.parent == target
        assert target.is_dir()

    def test_filename_matches_timestamp_raw_pattern(self, tmp_path: Path):
        saved = save_transcript(tmp_path, "本文", format_mode="none")
        assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}_raw\.md$", saved.name)

    def test_writes_transcription_body(self, tmp_path: Path):
        saved = save_transcript(tmp_path, "これは文字起こし結果です", format_mode="rule")
        content = saved.read_text(encoding="utf-8")
        assert "これは文字起こし結果です" in content

    def test_frontmatter_contains_required_fields(self, tmp_path: Path):
        saved = save_transcript(tmp_path, "本文", format_mode="llm")
        content = saved.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "type: transcription" in content
        assert "format_mode: llm" in content
        assert "tags:" in content
        assert "- recording" in content
        assert "- raw" in content
        assert "created:" in content

    def test_returns_path_pointing_to_existing_file(self, tmp_path: Path):
        saved = save_transcript(tmp_path, "本文", format_mode="none")
        assert saved.exists()
        assert saved.is_file()

    def test_accepts_string_path(self, tmp_path: Path):
        saved = save_transcript(str(tmp_path), "本文", format_mode="none")
        assert saved.exists()

    def test_raises_runtime_error_when_write_blocked(self, tmp_path: Path):
        # ファイルをディレクトリとして指定した場合は mkdir が失敗する
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        with pytest.raises((RuntimeError, FileExistsError, NotADirectoryError)):
            save_transcript(blocker / "child", "本文")


class TestSaveLiveNote:
    def test_filename_matches_timestamp_live_pattern(self, tmp_path: Path):
        saved = save_live_note(tmp_path, "本文")
        assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}_live\.md$", saved.name)

    def test_creates_save_folder_if_missing(self, tmp_path: Path):
        target = tmp_path / "nested"
        saved = save_live_note(target, "本文")
        assert saved.parent == target

    def test_frontmatter_and_body(self, tmp_path: Path):
        content = save_live_note(tmp_path, "ライブ本文").read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "created:" in content
        assert "type: live-transcription" in content
        assert "- recording" in content
        assert "- live" in content
        assert "- raw" not in content
        assert content.rstrip().endswith("ライブ本文")

    def test_raises_runtime_error_on_failure(self, tmp_path: Path):
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        with pytest.raises(RuntimeError):
            save_live_note(blocker / "child", "本文")


class TestSaveLiveNoteNamedAfterWav:
    """ライブノート名は録音 WAV(開始時刻)に対応させる"""

    def test_name_derived_from_wav_stem(self, tmp_path: Path):
        wav = tmp_path / "2026-10-27_120000_recording.wav"
        saved = save_live_note(tmp_path / "notes", "本文", audio_file=wav)
        assert saved.name == "2026-10-27_120000_live.md"

    def test_numbered_wav_maps_to_same_timestamp(self, tmp_path: Path):
        wav = tmp_path / "2026-10-27_120000_recording_2.wav"
        saved = save_live_note(tmp_path, "本文", audio_file=wav)
        assert saved.name == "2026-10-27_120000_live.md"

    def test_collision_appends_suffix(self, tmp_path: Path):
        wav = tmp_path / "2026-10-27_120000_recording.wav"
        first = save_live_note(tmp_path, "一回目", audio_file=wav)
        second = save_live_note(tmp_path, "二回目", audio_file=wav)
        third = save_live_note(tmp_path, "三回目", audio_file=wav)
        assert first.name == "2026-10-27_120000_live.md"
        assert second.name == "2026-10-27_120000_live_2.md"
        assert third.name == "2026-10-27_120000_live_3.md"
        assert first.read_text(encoding="utf-8").rstrip().endswith("一回目")

    def test_arbitrary_wav_name_uses_stem(self, tmp_path: Path):
        saved = save_live_note(tmp_path, "本文", audio_file=Path("/x/meeting.wav"))
        assert saved.name == "meeting_live.md"
