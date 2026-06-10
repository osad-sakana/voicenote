"""note_writer モジュールのユニットテスト。"""

import re
from pathlib import Path

import pytest

from note_writer import save_transcript


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
