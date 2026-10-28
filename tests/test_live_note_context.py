"""live.note.render_live_note の参考情報（参加者・メモ）対応のテスト。"""

from pathlib import Path

from live.context import MAX_MEMO_CHARS, LiveContext
from live.note import render_live_note
from live.session_models import LiveResult, LiveSegment

WAV = Path("/tmp/2026-10-28_100000_recording.wav")


def _result(**overrides) -> LiveResult:
    base = {
        "segments": (LiveSegment(0, 1.2, 3.4, "こんにちは"),),
        "final_summary": "現在の話題: 挨拶",
        "dropped_chunks": 0,
    }
    return LiveResult(**{**base, **overrides})


LEGACY = (
    "## 要約\n\n現在の話題: 挨拶\n\n"
    "## 文字起こし\n\n- [00:01] こんにちは\n\n"
    "## 音声ファイル\n\n/tmp/2026-10-28_100000_recording.wav\n"
)


class TestRenderWithoutContext:
    def test_legacy_output_is_pinned(self):
        assert render_live_note(_result(), WAV) == LEGACY

    def test_none_and_empty_context_match_legacy_exactly(self):
        assert render_live_note(_result(), WAV, context=None) == LEGACY
        assert render_live_note(_result(), WAV, context=LiveContext()) == LEGACY

    def test_quality_note_case_is_unchanged(self):
        result = _result(dropped_chunks=2, complete=False)
        legacy = render_live_note(result, WAV, "警告")
        assert render_live_note(result, WAV, "警告", context=LiveContext()) == legacy


class TestRenderWithContext:
    def test_sections_in_natural_order(self):
        ctx = LiveContext(participants=("田中", "佐藤"), memo="議題: 予算\n- 来期")
        body = render_live_note(_result(), WAV, context=ctx)
        assert body == (
            "## 要約\n\n現在の話題: 挨拶\n\n"
            "## 参加者\n\n- 田中\n- 佐藤\n\n"
            "## メモ\n\n議題: 予算\n- 来期\n\n"
            "## 文字起こし\n\n- [00:01] こんにちは\n\n"
            "## 音声ファイル\n\n/tmp/2026-10-28_100000_recording.wav\n"
        )

    def test_participants_only(self):
        body = render_live_note(_result(), WAV, context=LiveContext(participants=("田中",)))
        assert "## 参加者\n\n- 田中" in body
        assert "## メモ" not in body

    def test_memo_only(self):
        body = render_live_note(_result(), WAV, context=LiveContext(memo="メモ本文"))
        assert "## メモ\n\nメモ本文\n\n## 文字起こし" in body
        assert "## 参加者" not in body

    def test_memo_is_plain_body_not_quote(self):
        body = render_live_note(_result(), WAV, context=LiveContext(memo="一行目\n二行目"))
        assert "> 一行目" not in body
        assert "一行目\n二行目" in body

    def test_truncated_memo_has_notice(self):
        ctx = LiveContext.from_inputs("", "あ" * (MAX_MEMO_CHARS + 1))
        body = render_live_note(_result(), WAV, context=ctx)
        section = body[body.index("## メモ") : body.index("## 文字起こし")]
        assert f"先頭 {MAX_MEMO_CHARS} 文字" in section
        assert "あ" * MAX_MEMO_CHARS in section

    def test_untruncated_memo_has_no_notice(self):
        body = render_live_note(_result(), WAV, context=LiveContext(memo="短い"))
        assert "文字のみ" not in body

    def test_quality_note_stays_first(self):
        result = _result(dropped_chunks=1)
        body = render_live_note(result, WAV, context=LiveContext(participants=("田中",)))
        assert body.startswith("> [!warning]")
        assert body.index("## 要約") < body.index("## 参加者")

    def test_without_wav(self):
        body = render_live_note(_result(), context=LiveContext(memo="m"))
        assert body.endswith("## 文字起こし\n\n- [00:01] こんにちは\n")
