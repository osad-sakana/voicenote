"""gui/live_state.py のテスト(Tk 非依存のライブ表示状態管理)"""

import copy
import dataclasses

import pytest

from gui.live_state import (
    MAX_DISPLAY_LINES,
    WARNING_THROTTLE_SEC,
    LiveBanner,
    LivePhase,
    LiveViewState,
    add_banner,
    apply_segment,
    apply_status,
    apply_summary,
    apply_warning,
    clear_banner,
    format_segment_line,
    new_display_lines,
    set_phase,
    to_live_result,
    warning_kind,
)
from live.session_models import LiveResult, LiveSegment


def _state_with_segments(count: int) -> LiveViewState:
    state = LiveViewState()
    for i in range(count):
        state = apply_segment(state, i, float(i), float(i) + 0.5, f"t{i}")
    return state


class TestInitialState:
    def test_defaults(self):
        state = LiveViewState()
        assert state.segments == ()
        assert state.summary == ""
        assert state.status == ""
        assert state.warnings == ()
        assert state.banners == ()
        assert state.phase is LivePhase.IDLE

    def test_is_frozen(self):
        state = LiveViewState()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.summary = "x"  # type: ignore[misc]

    def test_phase_members(self):
        names = {p.name for p in LivePhase}
        assert names == {
            "IDLE",
            "PREPARING",
            "RECORDING",
            "FINALIZING",
            "DRAINING",
            "REFINING",
            "SAVING",
        }


class TestApplySegment:
    def test_appends_live_segment(self):
        state = apply_segment(LiveViewState(), 0, 1.0, 2.5, "こんにちは")
        assert state.segments == (LiveSegment(0, 1.0, 2.5, "こんにちは"),)

    def test_keeps_order(self):
        state = _state_with_segments(3)
        assert [s.index for s in state.segments] == [0, 1, 2]

    def test_does_not_mutate_original(self):
        original = _state_with_segments(1)
        snapshot = copy.deepcopy(original)
        new = apply_segment(original, 1, 2.0, 3.0, "next")
        assert original == snapshot
        assert new is not original
        assert len(new.segments) == 2


class TestApplySummaryAndStatus:
    def test_summary_replaced(self):
        state = apply_summary(LiveViewState(), "要約1")
        state = apply_summary(state, "要約2")
        assert state.summary == "要約2"

    def test_status_replaced(self):
        state = apply_status(LiveViewState(), "録音中")
        assert state.status == "録音中"

    def test_do_not_mutate_original(self):
        original = LiveViewState()
        snapshot = copy.deepcopy(original)
        apply_summary(original, "s")
        apply_status(original, "m")
        assert original == snapshot

    def test_same_value_returns_same_object(self):
        state = apply_summary(LiveViewState(), "s")
        assert apply_summary(state, "s") is state
        state = apply_status(state, "m")
        assert apply_status(state, "m") is state


class TestWarningKind:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (
                "区間 12.3s の文字起こしに失敗しました: boom",
                "区間 4.0s の文字起こしに失敗しました: other error",
            ),
            (
                "文字起こしが追いつかないため区間 4.0s〜 を破棄しました",
                "文字起こしが追いつかないため区間 123.5s〜 を破棄しました",
            ),
            (
                "音声認識エンジンの読み込みに失敗しました（録音は継続します）: e1",
                "音声認識エンジンの読み込みに失敗しました（録音は継続します）: e2",
            ),
            ("音声区間の検出に失敗しました：A", "音声区間の検出に失敗しました：B"),
        ],
    )
    def test_variable_parts_share_kind(self, a, b):
        assert warning_kind(a) == warning_kind(b)

    def test_different_messages_differ(self):
        assert warning_kind("最終要約が時間内に終わりませんでした") != warning_kind(
            "時間内に文字起こしできなかった区間があります"
        )

    def test_strips_whitespace(self):
        assert warning_kind("  警告 ") == warning_kind("警告")


class TestApplyWarning:
    def test_adds_warning(self):
        state = apply_warning(LiveViewState(), "警告A", now=0.0)
        assert state.warnings == ("警告A",)

    def test_does_not_mutate_original(self):
        original = apply_warning(LiveViewState(), "警告A", now=0.0)
        snapshot = copy.deepcopy(original)
        new = apply_warning(original, "警告B", now=1.0)
        assert original == snapshot
        assert new.warnings == ("警告A", "警告B")

    def test_exact_duplicate_ignored_even_after_window(self):
        state = apply_warning(LiveViewState(), "警告A", now=0.0)
        assert apply_warning(state, "警告A", now=1.0) is state
        assert apply_warning(state, "警告A", now=100.0) is state

    def test_same_kind_throttled_within_window(self):
        state = apply_warning(LiveViewState(), "区間 1.0s の文字起こしに失敗しました: x", now=0.0)
        same_kind = "区間 5.0s の文字起こしに失敗しました: y"
        assert apply_warning(state, same_kind, now=9.99) is state

    def test_same_kind_allowed_at_window_boundary(self):
        state = apply_warning(LiveViewState(), "区間 1.0s の文字起こしに失敗しました: x", now=0.0)
        later = apply_warning(
            state, "区間 5.0s の文字起こしに失敗しました: y", now=WARNING_THROTTLE_SEC
        )
        assert later.warnings == (
            "区間 1.0s の文字起こしに失敗しました: x",
            "区間 5.0s の文字起こしに失敗しました: y",
        )

    def test_throttle_window_restarts_from_last_accepted(self):
        msg = "文字起こしが追いつかないため区間 {}s〜 を破棄しました"
        state = apply_warning(LiveViewState(), msg.format(1.0), now=0.0)
        state = apply_warning(state, msg.format(2.0), now=5.0)  # 間引き(記録時刻は 0.0 のまま)
        state = apply_warning(state, msg.format(3.0), now=10.0)  # 受理
        assert apply_warning(state, msg.format(4.0), now=15.0) is state
        assert len(state.warnings) == 2

    def test_different_kinds_not_throttled(self):
        state = apply_warning(LiveViewState(), "警告A", now=0.0)
        state = apply_warning(state, "区間 1.0s の文字起こしに失敗しました: x", now=0.1)
        assert len(state.warnings) == 2

    def test_empty_message_ignored(self):
        state = LiveViewState()
        assert apply_warning(state, "   ", now=0.0) is state


class TestBanners:
    def test_add_banner(self):
        state = add_banner(LiveViewState(), "ollama", "Ollama に接続できません")
        assert state.banners == (LiveBanner("ollama", "Ollama に接続できません"),)

    def test_add_same_kind_replaces_in_place(self):
        state = add_banner(LiveViewState(), "model_download", "DL 10%")
        state = add_banner(state, "ollama", "Ollama")
        state = add_banner(state, "model_download", "DL 50%")
        assert state.banners == (
            LiveBanner("model_download", "DL 50%"),
            LiveBanner("ollama", "Ollama"),
        )

    def test_add_identical_returns_same_object(self):
        state = add_banner(LiveViewState(), "low_input", "入力が小さい")
        assert add_banner(state, "low_input", "入力が小さい") is state

    def test_clear_banner(self):
        state = add_banner(LiveViewState(), "vad_fallback", "Energy VAD を使用")
        state = add_banner(state, "engine_fallback", "faster-whisper を使用")
        cleared = clear_banner(state, "vad_fallback")
        assert cleared.banners == (LiveBanner("engine_fallback", "faster-whisper を使用"),)

    def test_clear_missing_returns_same_object(self):
        state = add_banner(LiveViewState(), "external_send", "外部送信")
        assert clear_banner(state, "ollama") is state

    def test_do_not_mutate_original(self):
        original = add_banner(LiveViewState(), "ollama", "a")
        snapshot = copy.deepcopy(original)
        add_banner(original, "low_input", "b")
        clear_banner(original, "ollama")
        assert original == snapshot

    def test_empty_kind_rejected(self):
        with pytest.raises(ValueError):
            add_banner(LiveViewState(), "", "text")


class TestSetPhase:
    def test_set_phase(self):
        state = set_phase(LiveViewState(), LivePhase.RECORDING)
        assert state.phase is LivePhase.RECORDING

    def test_does_not_mutate_original(self):
        original = LiveViewState()
        set_phase(original, LivePhase.DRAINING)
        assert original.phase is LivePhase.IDLE

    def test_same_phase_returns_same_object(self):
        state = LiveViewState()
        assert set_phase(state, LivePhase.IDLE) is state

    def test_invalid_phase_rejected(self):
        with pytest.raises(TypeError):
            set_phase(LiveViewState(), "RECORDING")  # type: ignore[arg-type]


class TestFormatSegmentLine:
    def test_format(self):
        assert format_segment_line(LiveSegment(0, 65.4, 70.0, "テキスト")) == "[01:05] テキスト"

    def test_hour_format(self):
        assert format_segment_line(LiveSegment(0, 3661.0, 3662.0, "x")) == "[1:01:01] x"

    def test_collapses_newlines_and_whitespace(self):
        line = format_segment_line(LiveSegment(0, 0.0, 1.0, "  一行目\n二行目\r\n\t三行目  "))
        assert line == "[00:00] 一行目 二行目 三行目"
        assert "\n" not in line


class TestNewDisplayLines:
    def test_initial_render(self):
        state = _state_with_segments(3)
        delta = new_display_lines(state, 0)
        assert delta.lines == ("[00:00] t0", "[00:01] t1", "[00:02] t2")
        assert delta.new_count == 3
        assert delta.trim_count == 0

    def test_only_new_lines(self):
        state = _state_with_segments(5)
        delta = new_display_lines(state, 3)
        assert delta.lines == ("[00:03] t3", "[00:04] t4")
        assert delta.new_count == 5

    def test_nothing_new(self):
        state = _state_with_segments(2)
        delta = new_display_lines(state, 2)
        assert delta.lines == ()
        assert delta.new_count == 2
        assert delta.trim_count == 0

    def test_trim_when_exceeding_limit(self):
        state = _state_with_segments(MAX_DISPLAY_LINES + 3)
        delta = new_display_lines(state, MAX_DISPLAY_LINES)
        assert len(delta.lines) == 3
        assert delta.trim_count == 3
        assert delta.new_count == MAX_DISPLAY_LINES + 3

    def test_trim_after_already_trimmed(self):
        state = _state_with_segments(MAX_DISPLAY_LINES + 10)
        delta = new_display_lines(state, MAX_DISPLAY_LINES + 5)
        assert len(delta.lines) == 5
        assert delta.trim_count == 5

    def test_burst_larger_than_limit(self):
        state = _state_with_segments(MAX_DISPLAY_LINES + 20)
        delta = new_display_lines(state, 10)
        assert len(delta.lines) == MAX_DISPLAY_LINES
        assert delta.lines[-1] == format_segment_line(state.segments[-1])
        assert delta.trim_count == 10
        assert delta.new_count == MAX_DISPLAY_LINES + 20

    @pytest.mark.parametrize("bad", [-1, 4])
    def test_invalid_rendered_count(self, bad):
        with pytest.raises(ValueError):
            new_display_lines(_state_with_segments(3), bad)

    def test_does_not_mutate_state(self):
        state = _state_with_segments(3)
        snapshot = copy.deepcopy(state)
        new_display_lines(state, 1)
        assert state == snapshot


class TestToLiveResult:
    def test_builds_partial_result(self):
        state = apply_summary(_state_with_segments(2), "要約")
        result = to_live_result(state, dropped_chunks=3, complete=False)
        assert result == LiveResult(
            segments=state.segments,
            final_summary="要約",
            dropped_chunks=3,
            dropped_segments=0,
            complete=False,
        )

    def test_dropped_segments_keyword(self):
        result = to_live_result(LiveViewState(), 0, True, dropped_segments=2)
        assert result.dropped_segments == 2
        assert result.complete is True
        assert result.segments == ()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dropped_chunks": -1, "complete": True},
            {"dropped_chunks": 0, "complete": True, "dropped_segments": -1},
            {"dropped_chunks": True, "complete": True},
            {"dropped_chunks": 1.5, "complete": True},
        ],
    )
    def test_invalid_counts(self, kwargs):
        with pytest.raises(ValueError):
            to_live_result(LiveViewState(), **kwargs)

    def test_does_not_mutate_state(self):
        state = _state_with_segments(1)
        snapshot = copy.deepcopy(state)
        to_live_result(state, 0, True)
        assert state == snapshot
