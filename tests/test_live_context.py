"""live.context のユニットテスト。"""

import dataclasses

import pytest

from live.context import (
    MAX_MEMO_CHARS,
    MAX_PARTICIPANT_CHARS,
    MAX_PARTICIPANTS,
    LiveContext,
    parse_participants,
)

RLO = chr(0x202E)  # 双方向制御文字(RIGHT-TO-LEFT OVERRIDE)


class TestParseParticipants:
    @pytest.mark.parametrize(
        "text",
        [
            "田中、佐藤、鈴木",
            "田中，佐藤，鈴木",
            "田中,佐藤,鈴木",
            "田中\n佐藤\n鈴木",
            "田中\r\n佐藤\r鈴木",
            "田中、 佐藤 ,\n 鈴木",
        ],
    )
    def test_splits_on_supported_separators(self, text):
        assert parse_participants(text) == ("田中", "佐藤", "鈴木")

    def test_strips_whitespace_and_drops_empty(self):
        assert parse_participants("  田中 、、 ,\n\n 　佐藤　 ") == ("田中", "佐藤")

    def test_internal_spaces_are_kept(self):
        assert parse_participants("田中 太郎、John Smith") == ("田中 太郎", "John Smith")

    def test_deduplicates_preserving_order(self):
        assert parse_participants("佐藤、田中、佐藤、鈴木、田中") == ("佐藤", "田中", "鈴木")

    @pytest.mark.parametrize("text", ["", "   ", "、,，\n", None])
    def test_empty_input(self, text):
        assert parse_participants(text) == ()

    def test_name_is_truncated_to_max_chars(self):
        name = "あ" * (MAX_PARTICIPANT_CHARS + 5)
        assert parse_participants(name) == ("あ" * MAX_PARTICIPANT_CHARS,)

    def test_dedup_after_truncation(self):
        base = "い" * MAX_PARTICIPANT_CHARS
        assert parse_participants(f"{base}A、{base}B") == (base,)

    def test_count_is_capped(self):
        names = [f"人{i}" for i in range(MAX_PARTICIPANTS + 10)]
        result = parse_participants("、".join(names))
        assert result == tuple(names[:MAX_PARTICIPANTS])

    def test_control_characters_are_removed(self):
        assert parse_participants(f"田\x00中\x1b、佐{RLO}藤\t") == ("田中", "佐藤")

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            parse_participants(123)


class TestLiveContext:
    def test_defaults_are_empty(self):
        ctx = LiveContext()
        assert ctx.participants == ()
        assert ctx.memo == ""
        assert ctx.memo_truncated is False
        assert ctx.is_empty is True

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            LiveContext().memo = "x"

    def test_not_empty_with_participants_or_memo(self):
        assert LiveContext(participants=("田中",)).is_empty is False
        assert LiveContext(memo="議題").is_empty is False

    def test_direct_construction_is_normalized(self):
        ctx = LiveContext(
            participants=[" 田中 ", "田中", "", "a" * (MAX_PARTICIPANT_CHARS + 1)],
            memo="x" * (MAX_MEMO_CHARS + 1),
        )
        assert ctx.participants == ("田中", "a" * MAX_PARTICIPANT_CHARS)
        assert len(ctx.memo) == MAX_MEMO_CHARS
        assert ctx.memo_truncated is True

    def test_direct_participant_with_separator_is_kept_as_one(self):
        assert LiveContext(participants=("田中、佐藤",)).participants == ("田中、佐藤",)

    def test_direct_participants_count_is_capped(self):
        ctx = LiveContext(participants=tuple(f"p{i}" for i in range(MAX_PARTICIPANTS + 3)))
        assert len(ctx.participants) == MAX_PARTICIPANTS

    def test_normalization_is_idempotent(self):
        ctx = LiveContext.from_inputs("田中、佐藤", "m" * (MAX_MEMO_CHARS + 10))
        again = LiveContext(ctx.participants, ctx.memo, ctx.memo_truncated)
        assert again == ctx

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"participants": "田中"},
            {"participants": ("田中", 1)},
            {"memo": None},
            {"memo_truncated": "yes"},
        ],
    )
    def test_invalid_types_raise(self, kwargs):
        with pytest.raises(TypeError):
            LiveContext(**kwargs)

    def test_repr_does_not_leak_content(self):
        ctx = LiveContext.from_inputs("秘密の人", "秘密のメモ")
        text = repr(ctx)
        assert "秘密" not in text
        assert "participants=1" in text


class TestFromInputs:
    def test_parses_participants_and_memo(self):
        ctx = LiveContext.from_inputs("田中、佐藤", "  議題: 予算\n担当: 田中  ")
        assert ctx.participants == ("田中", "佐藤")
        assert ctx.memo == "議題: 予算\n担当: 田中"
        assert ctx.memo_truncated is False

    def test_none_inputs_become_empty(self):
        assert LiveContext.from_inputs(None, None) == LiveContext()

    def test_memo_keeps_head_when_too_long(self):
        memo = "頭" + "中" * MAX_MEMO_CHARS + "尾"
        ctx = LiveContext.from_inputs("", memo)
        assert len(ctx.memo) == MAX_MEMO_CHARS
        assert ctx.memo.startswith("頭")
        assert "尾" not in ctx.memo
        assert ctx.memo_truncated is True

    def test_memo_at_limit_is_not_truncated(self):
        ctx = LiveContext.from_inputs("", "a" * MAX_MEMO_CHARS)
        assert len(ctx.memo) == MAX_MEMO_CHARS
        assert ctx.memo_truncated is False

    def test_memo_limit_is_measured_after_strip(self):
        ctx = LiveContext.from_inputs("", "   " + "a" * MAX_MEMO_CHARS + "\n\n")
        assert ctx.memo_truncated is False

    def test_memo_control_characters_removed_but_newline_and_tab_kept(self):
        ctx = LiveContext.from_inputs("", f"a\x00b\x07c\r\nd\re\tf{RLO}g\x7f")
        assert ctx.memo == "abc\nd\ne\tfg"

    def test_memo_truncation_does_not_leave_trailing_whitespace(self):
        memo = "a" * (MAX_MEMO_CHARS - 1) + " " + "b" * 10
        ctx = LiveContext.from_inputs("", memo)
        assert ctx.memo == "a" * (MAX_MEMO_CHARS - 1)
        assert ctx.memo_truncated is True

    def test_rejects_non_string_memo(self):
        with pytest.raises(TypeError):
            LiveContext.from_inputs("", 1)
