"""live.note.render_live_note のユニットテスト。"""

from pathlib import Path

import pytest

from live.note import external_send_warning, format_timestamp, render_live_note
from live.session_models import LiveResult, LiveSegment


def _result(**overrides) -> LiveResult:
    base = {
        "segments": (
            LiveSegment(0, 1.2, 3.4, "こんにちは"),
            LiveSegment(1, 65.0, 70.0, "本題です"),
        ),
        "final_summary": "現在の話題: 挨拶",
        "dropped_chunks": 0,
    }
    return LiveResult(**{**base, **overrides})


class TestFormatTimestamp:
    def test_minutes_and_seconds(self):
        assert format_timestamp(65.9) == "01:05"

    def test_hours(self):
        assert format_timestamp(3725.0) == "1:02:05"

    def test_negative_is_clamped(self):
        assert format_timestamp(-1.0) == "00:00"


class TestRenderLiveNote:
    def test_contains_summary_and_timestamped_segments(self):
        body = render_live_note(_result())
        assert "## 要約" in body
        assert "現在の話題: 挨拶" in body
        assert "## 文字起こし" in body
        assert "[00:01] こんにちは" in body
        assert "[01:05] 本題です" in body
        assert body.index("## 要約") < body.index("## 文字起こし")

    def test_segments_are_in_order(self):
        body = render_live_note(_result())
        assert body.index("こんにちは") < body.index("本題です")

    def test_empty_summary_shows_placeholder(self):
        body = render_live_note(_result(final_summary=""))
        assert "（要約なし）" in body

    def test_no_segments_shows_placeholder(self):
        body = render_live_note(_result(segments=()))
        assert "（文字起こし結果なし）" in body

    def test_wav_path_is_referenced(self):
        body = render_live_note(_result(), wav_path=Path("/tmp/a_recording.wav"))
        assert "/tmp/a_recording.wav" in body

    def test_notes_dropped_audio_and_incomplete(self):
        body = render_live_note(_result(dropped_chunks=3, dropped_segments=2, complete=False))
        assert "欠落" in body

    def test_no_warning_when_clean(self):
        assert "欠落" not in render_live_note(_result())

    def test_is_pure(self):
        result = _result()
        assert render_live_note(result) == render_live_note(result)


class TestExternalSendWarning:
    @pytest.mark.parametrize(
        ("transcription_mode", "format_mode"),
        [("openai", "rule"), ("local", "llm"), ("openai", "llm")],
    )
    def test_warns_when_openai_is_used(self, transcription_mode, format_mode):
        warning = external_send_warning(transcription_mode, format_mode)
        assert warning is not None
        assert "OpenAI に送信します" in warning
        assert "--config" in warning
        assert "local / rule" in warning

    @pytest.mark.parametrize("format_mode", ["rule", "none"])
    def test_no_warning_for_local_settings(self, format_mode):
        assert external_send_warning("local", format_mode) is None


class TestRenderLiveNotePrivacy:
    def test_quality_note_includes_external_send_warning(self):
        warning = external_send_warning("openai", "rule")
        body = render_live_note(_result(dropped_chunks=1), send_warning=warning)
        assert "--file" in body
        assert warning in body

    def test_local_quality_note_has_no_external_warning(self):
        body = render_live_note(_result(dropped_chunks=1))
        assert "OpenAI" not in body

    def test_clean_result_has_no_warning_even_with_openai(self):
        body = render_live_note(_result(), send_warning=external_send_warning("openai", "llm"))
        assert "OpenAI" not in body
