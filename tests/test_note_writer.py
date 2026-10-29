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


class TestSaveLiveNoteParticipants:
    """frontmatter の participants(参加者)。空なら従来と同一の frontmatter"""

    @staticmethod
    def _frontmatter(content: str) -> str:
        return content.split("---\n")[1]

    def test_without_participants_frontmatter_is_unchanged(self, tmp_path: Path):
        content = save_live_note(tmp_path, "本文").read_text(encoding="utf-8")
        fm = self._frontmatter(content)
        assert "participants" not in fm
        assert fm.endswith("type: live-transcription\ntags:\n  - recording\n  - live\n")

    def test_empty_participants_same_as_omitted(self, tmp_path: Path):
        a = save_live_note(tmp_path / "a", "本文", participants=()).read_text(encoding="utf-8")
        b = save_live_note(tmp_path / "b", "本文").read_text(encoding="utf-8")
        assert a.split("\n", 2)[2] == b.split("\n", 2)[2]  # created 行以外が一致

    def test_participants_are_listed(self, tmp_path: Path):
        saved = save_live_note(tmp_path, "本文", participants=("田中", "佐藤"))
        fm = self._frontmatter(saved.read_text(encoding="utf-8"))
        assert 'participants:\n  - "田中"\n  - "佐藤"\n' in fm
        assert fm.index("participants:") > fm.index("type: live-transcription")

    def test_participants_are_yaml_safe_quoted(self, tmp_path: Path):
        names = ("田中: 部長", "#1", "- x", "yes", "null", 'a"b\\c')
        saved = save_live_note(tmp_path, "本文", participants=names)
        fm = self._frontmatter(saved.read_text(encoding="utf-8"))
        assert '  - "田中: 部長"\n' in fm
        assert '  - "#1"\n' in fm
        assert '  - "- x"\n' in fm
        assert '  - "yes"\n' in fm
        assert '  - "null"\n' in fm
        assert '  - "a\\"b\\\\c"\n' in fm

    def test_participants_cannot_break_frontmatter(self, tmp_path: Path):
        saved = save_live_note(tmp_path, "本文", participants=("x\n---\ntype: evil",))
        content = saved.read_text(encoding="utf-8")
        assert content.count("\n---\n") == 1
        assert '  - "x\\n---\\ntype: evil"\n' in content


class TestOverwriteLiveNote:
    def _saved(self, tmp_path: Path) -> Path:
        return save_live_note(tmp_path, "速報版の本文", participants=("山田",))

    def test_replaces_body_and_adds_refined_tag(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        path = self._saved(tmp_path)
        result = overwrite_live_note(path, "高精度版の本文", participants=("山田",), refined=True)
        assert result == path
        content = path.read_text(encoding="utf-8")
        assert "高精度版の本文" in content
        assert "速報版の本文" not in content
        assert "  - refined\n" in content
        assert "type: live-transcription" in content
        assert 'participants:\n  - "山田"\n' in content

    def test_without_refined_flag_has_no_tag(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        path = self._saved(tmp_path)
        overwrite_live_note(path, "本文")
        assert "refined" not in path.read_text(encoding="utf-8")

    def test_leaves_no_temporary_files(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        path = self._saved(tmp_path)
        overwrite_live_note(path, "本文", refined=True)
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]

    def test_keeps_file_mode(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        path = self._saved(tmp_path)
        path.chmod(0o644)
        overwrite_live_note(path, "本文", refined=True)
        assert path.stat().st_mode & 0o777 == 0o644

    def test_failure_keeps_original_and_cleans_up(self, tmp_path: Path, monkeypatch):
        import note_writer

        path = self._saved(tmp_path)

        def broken_replace(src, dst):
            raise OSError("disk error")

        monkeypatch.setattr(note_writer.os, "replace", broken_replace)
        with pytest.raises(RuntimeError, match="保存エラー"):
            note_writer.overwrite_live_note(path, "高精度版", refined=True)
        assert "速報版の本文" in path.read_text(encoding="utf-8")
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]

    def test_missing_target_raises(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        with pytest.raises(RuntimeError):
            overwrite_live_note(tmp_path / "missing_live.md", "本文")


class TestOverwriteKeepsCreated:
    def test_keeps_created_of_existing_note(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        path = save_live_note(tmp_path, "速報版の本文")
        created_line = next(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("created: ")
        )
        overwrite_live_note(path, "高精度版", refined=True)
        assert created_line in path.read_text(encoding="utf-8").splitlines()

    def test_uses_now_when_created_unreadable(self, tmp_path: Path):
        from note_writer import overwrite_live_note

        path = tmp_path / "x_live.md"
        path.write_text("frontmatter なしの本文\ncreated: 偽物\n", encoding="utf-8")
        overwrite_live_note(path, "高精度版")
        content = path.read_text(encoding="utf-8")
        assert "created: 偽物" not in content
        assert re.search(r"^created: \d{4}-\d{2}-\d{2}T", content, re.MULTILINE)
