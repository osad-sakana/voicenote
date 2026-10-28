"""gui/live_view_logic.py のテスト(Tk 非依存のライブ表示整形)"""

import dataclasses

import pytest

from gui.live_state import (
    MAX_DISPLAY_LINES,
    LiveBanner,
    LivePhase,
    LiveViewState,
    apply_segment,
    apply_warning,
)
from gui.live_view_logic import (
    BACKLOG_WARN_SEC,
    LEVEL_FLOOR_DB,
    ButtonAction,
    ButtonStyle,
    banner_view,
    button_spec,
    controls_locked,
    download_fraction,
    format_backlog,
    format_dropped,
    format_level,
    level_fraction,
    meter_view,
    new_warning,
    plan_segment_render,
    result_text,
    status_text,
)
from gui.live_workflow import (
    BANNER_MODEL_DOWNLOAD,
    BANNER_OLLAMA,
    MeterReading,
    format_download_progress,
)
from live.level_meter import InputLevelAssessment


def _segments(count: int, state: LiveViewState | None = None) -> LiveViewState:
    state = state or LiveViewState()
    start = len(state.segments)
    for i in range(start, start + count):
        state = apply_segment(state, i, float(i), float(i) + 0.5, f"t{i}")
    return state


def _reading(level=-30.0, backlog=0.0, dropped=0, status="ok") -> MeterReading:
    return MeterReading(
        level_db=level,
        backlog_sec=backlog,
        dropped_chunks=dropped,
        input_level=InputLevelAssessment(status, None, "入力が小さいです"),
    )


class TestButtonSpec:
    def test_idle_starts(self):
        spec = button_spec(LivePhase.IDLE)
        assert spec.text == "ライブ開始"
        assert spec.action is ButtonAction.START
        assert spec.style is ButtonStyle.PRIMARY
        assert spec.enabled

    @pytest.mark.parametrize("phase", [LivePhase.PREPARING, LivePhase.RECORDING])
    def test_preparing_and_recording_stop(self, phase):
        spec = button_spec(phase)
        assert spec.text == "停止"
        assert spec.action is ButtonAction.STOP
        assert spec.style is ButtonStyle.DANGER
        assert spec.enabled

    def test_draining_aborts(self):
        spec = button_spec(LivePhase.DRAINING)
        assert spec.text == "中断して保存"
        assert spec.action is ButtonAction.ABORT
        assert spec.enabled

    @pytest.mark.parametrize("phase", [LivePhase.FINALIZING, LivePhase.SAVING])
    def test_busy_phases_disabled(self, phase):
        spec = button_spec(phase)
        assert not spec.enabled
        assert spec.action is ButtonAction.NONE
        assert spec.text

    def test_every_phase_has_spec(self):
        for phase in LivePhase:
            assert button_spec(phase).text

    def test_spec_is_immutable(self):
        spec = button_spec(LivePhase.IDLE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.text = "x"

    def test_rejects_non_phase(self):
        with pytest.raises(TypeError):
            button_spec("idle")


class TestStatusText:
    def test_idle(self):
        assert status_text(LivePhase.IDLE, "") == "待機中"

    def test_recording_with_session_status(self):
        assert status_text(LivePhase.RECORDING, "ライブ文字起こし中") == "ライブ文字起こし中"

    def test_recording_without_status(self):
        assert status_text(LivePhase.RECORDING, "") == "ライブ録音中"

    @pytest.mark.parametrize(
        "phase", [LivePhase.PREPARING, LivePhase.FINALIZING, LivePhase.DRAINING, LivePhase.SAVING]
    )
    def test_phase_text_takes_priority(self, phase):
        text = status_text(phase, "ライブ文字起こし中")
        assert text
        assert text != "ライブ文字起こし中"

    def test_preparing_ignores_model_loading_status(self):
        # 準備中は段階の文言を出す(セッション状態は録音中にだけ使う)
        assert "準備" in status_text(LivePhase.PREPARING, "")


class TestControlsLocked:
    def test_idle_unlocked(self):
        assert controls_locked(LivePhase.IDLE) is False

    @pytest.mark.parametrize("phase", [p for p in LivePhase if p is not LivePhase.IDLE])
    def test_active_locked(self, phase):
        assert controls_locked(phase) is True


class TestPlanSegmentRender:
    def test_first_render(self):
        plan = plan_segment_render(_segments(3), 0)
        assert plan.reset is False
        assert plan.delta.lines == ("[00:00] t0", "[00:01] t1", "[00:02] t2")
        assert plan.delta.new_count == 3
        assert plan.delta.trim_count == 0

    def test_incremental(self):
        state = _segments(5)
        plan = plan_segment_render(state, 3)
        assert plan.reset is False
        assert plan.delta.lines == ("[00:03] t3", "[00:04] t4")
        assert plan.delta.new_count == 5

    def test_no_change(self):
        plan = plan_segment_render(_segments(2), 2)
        assert plan.reset is False
        assert plan.delta.lines == ()

    def test_new_run_resets(self):
        # start() で状態が初期化され区間数が描画済み数より少なくなった
        plan = plan_segment_render(_segments(1), 42)
        assert plan.reset is True
        assert plan.delta.lines == ("[00:00] t0",)
        assert plan.delta.new_count == 1
        assert plan.delta.trim_count == 0

    def test_new_run_empty(self):
        plan = plan_segment_render(LiveViewState(), 5)
        assert plan.reset is True
        assert plan.delta.lines == ()
        assert plan.delta.new_count == 0

    def test_trim_over_limit(self):
        state = _segments(MAX_DISPLAY_LINES + 2)
        plan = plan_segment_render(state, MAX_DISPLAY_LINES)
        assert plan.delta.trim_count == 2
        assert len(plan.delta.lines) == 2

    def test_negative_count_resets(self):
        plan = plan_segment_render(_segments(2), -1)
        assert plan.reset is True
        assert plan.delta.new_count == 2


class TestDownloadFraction:
    def test_parses_percent(self):
        text = format_download_progress("large-v3-turbo", 500_000_000, 1_000_000_000)
        assert download_fraction(text) == pytest.approx(0.5)

    def test_complete(self):
        text = format_download_progress("small", 10, 10)
        assert download_fraction(text) == pytest.approx(1.0)

    def test_unknown_total(self):
        text = format_download_progress("small", 1_000_000, 0)
        assert download_fraction(text) is None

    def test_no_percent(self):
        assert download_fraction("何か別の文言") is None

    def test_empty(self):
        assert download_fraction("") is None


class TestBannerView:
    def test_empty(self):
        view = banner_view(())
        assert view.visible is False
        assert view.text == ""
        assert view.show_progress is False

    def test_joins_banner_texts(self):
        banners = (LiveBanner(BANNER_OLLAMA, "Ollama 不在"), LiveBanner("vad_fallback", "VAD"))
        view = banner_view(banners)
        assert view.visible is True
        assert view.text == "Ollama 不在\nVAD"
        assert view.show_progress is False

    def test_download_banner_shows_progress(self):
        text = format_download_progress("small", 25, 100)
        view = banner_view((LiveBanner(BANNER_MODEL_DOWNLOAD, text),))
        assert view.show_progress is True
        assert view.fraction == pytest.approx(0.25)

    def test_download_unknown_total_indeterminate(self):
        text = format_download_progress("small", 25, 0)
        view = banner_view((LiveBanner(BANNER_MODEL_DOWNLOAD, text),))
        assert view.show_progress is True
        assert view.fraction is None

    def test_strips_emoji(self):
        view = banner_view((LiveBanner("x", "注意\U0001f600です"),))
        assert view.text == "注意です"


class TestMeterFormatting:
    @pytest.mark.parametrize(
        ("db", "expected"),
        [(0.0, 1.0), (LEVEL_FLOOR_DB, 0.0), (LEVEL_FLOOR_DB - 40, 0.0), (6.0, 1.0)],
    )
    def test_level_fraction_bounds(self, db, expected):
        assert level_fraction(db) == pytest.approx(expected)

    def test_level_fraction_middle(self):
        assert level_fraction(LEVEL_FLOOR_DB / 2) == pytest.approx(0.5)

    def test_level_fraction_nan(self):
        assert level_fraction(float("nan")) == 0.0

    def test_format_level(self):
        assert format_level(-35.4) == "-35 dB"
        assert format_level(-100.0) == "-100 dB"

    def test_format_backlog(self):
        assert format_backlog(1.234) == "遅延 1.2秒"
        assert format_backlog(0.0) == "遅延 0.0秒"
        assert format_backlog(-3.0) == "遅延 0.0秒"
        assert format_backlog(None) == "遅延 -"

    def test_format_dropped(self):
        assert format_dropped(0) == "破棄 0"
        assert format_dropped(12) == "破棄 12"
        assert format_dropped(-1) == "破棄 0"
        assert format_dropped(None) == "破棄 -"


class TestMeterView:
    def test_none_reading(self):
        view = meter_view(None)
        assert view.fraction == 0.0
        assert view.level_text == "- dB"
        assert view.backlog_text == "遅延 -"
        assert view.dropped_text == "破棄 -"
        assert view.low_input is False
        assert view.backlog_high is False

    def test_reading(self):
        view = meter_view(_reading(level=-30.0, backlog=2.5, dropped=3))
        assert view.fraction == pytest.approx(level_fraction(-30.0))
        assert view.level_text == "-30 dB"
        assert view.backlog_text == "遅延 2.5秒"
        assert view.dropped_text == "破棄 3"
        assert view.low_input is False
        assert view.backlog_high is False

    def test_low_input_flag(self):
        assert meter_view(_reading(level=-50.0, status="low")).low_input is True

    def test_backlog_high(self):
        assert meter_view(_reading(backlog=BACKLOG_WARN_SEC)).backlog_high is True
        assert meter_view(_reading(backlog=BACKLOG_WARN_SEC - 0.1)).backlog_high is False


class TestNewWarning:
    def test_none_when_empty(self):
        assert new_warning(LiveViewState(), None) is None

    def test_returns_latest_when_unseen(self):
        state = apply_warning(LiveViewState(), "警告A", 0.0)
        assert new_warning(state, None) == "警告A"

    def test_none_when_already_seen(self):
        state = apply_warning(LiveViewState(), "警告A", 0.0)
        assert new_warning(state, "警告A") is None

    def test_returns_newer(self):
        state = apply_warning(LiveViewState(), "警告A", 0.0)
        state = apply_warning(state, "別の警告", 1.0)
        assert new_warning(state, "警告A") == "別の警告"


class TestResultText:
    def test_file_names_only(self, tmp_path):
        # 既存の完了表示と同じくファイル名だけを出す(フルパスは詳細ログに残す)
        wav = tmp_path / "a_recording.wav"
        note = tmp_path / "a_live.md"
        text = result_text(wav, note)
        assert text.splitlines() == ["WAV: a_recording.wav", "ノート: a_live.md"]
        assert str(tmp_path) not in text

    def test_without_note(self, tmp_path):
        wav = tmp_path / "a_recording.wav"
        assert result_text(wav, None) == "WAV: a_recording.wav"


class TestTranscriptLayout:
    def test_toggle_text(self):
        from gui.live_view_logic import transcript_toggle_text

        assert transcript_toggle_text(False) == "文字起こしを隠す"
        assert transcript_toggle_text(True) == "文字起こしを表示"

    def test_textbox_height_grows_with_lines(self):
        from gui.live_view_logic import textbox_height

        assert textbox_height(4, 12) > textbox_height(3, 12) > 0
        assert textbox_height(3, 15) > textbox_height(3, 12)

    def test_textbox_height_four_lines_is_small(self):
        from gui.live_view_logic import TRANSCRIPT_LINES, textbox_height

        # 文字起こしは数行分(副表示)
        assert 3 <= TRANSCRIPT_LINES <= 5
        assert textbox_height(TRANSCRIPT_LINES, 12) <= 100

    @pytest.mark.parametrize("lines, size", [(0, 12), (-1, 12), (3, 0)])
    def test_textbox_height_invalid(self, lines, size):
        from gui.live_view_logic import textbox_height

        with pytest.raises(ValueError):
            textbox_height(lines, size)


class TestTranscriptAutoCollapse:
    """要約(主役)の最小高さを守るため、窓が狭い時は文字起こしを自動で折りたたむ"""

    def _cramped(self, body_height, *, fixed=26, summary_min=83, transcript=100):
        from gui.live_view_logic import is_body_cramped

        return is_body_cramped(
            body_height, fixed_height=fixed, summary_min=summary_min, transcript_height=transcript
        )

    def test_roomy_when_everything_fits(self):
        assert self._cramped(26 + 83 + 100) is False
        assert self._cramped(500) is False

    def test_cramped_when_summary_would_drop_below_min(self):
        assert self._cramped(26 + 83 + 100 - 1) is True
        assert self._cramped(1) is True  # 配置前(高さ 1)も狭いとみなす

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"body_height": -1},
            {"fixed": -1},
            {"summary_min": 0},
            {"transcript": -1},
        ],
    )
    def test_invalid_values(self, kwargs):
        from gui.live_view_logic import is_body_cramped

        args = {"body_height": 100, "fixed": 26, "summary_min": 83, "transcript": 100, **kwargs}
        with pytest.raises(ValueError):
            is_body_cramped(
                args["body_height"],
                fixed_height=args["fixed"],
                summary_min=args["summary_min"],
                transcript_height=args["transcript"],
            )

    def test_summary_min_height_is_three_lines(self):
        from gui.live_view_logic import SUMMARY_MIN_LINES, summary_min_height, textbox_height

        assert SUMMARY_MIN_LINES == 3
        assert summary_min_height(16) == textbox_height(3, 16)


class TestTranscriptPreference:
    """既定(AUTO)・利用者が隠した(HIDDEN)・狭くても表示(SHOWN)の遷移と表示"""

    def test_default_is_auto(self):
        from gui.live_view_logic import TranscriptPreference, transcript_view

        view = transcript_view(TranscriptPreference.AUTO, cramped=False)
        assert view.collapsed is False
        assert view.toggle_text == "文字起こしを隠す"

    def test_auto_collapses_when_cramped_and_says_so(self):
        from gui.live_view_logic import TranscriptPreference, transcript_view

        view = transcript_view(TranscriptPreference.AUTO, cramped=True)
        assert view.collapsed is True
        assert view.auto is True
        assert "表示" in view.toggle_text
        assert "自動" in view.toggle_text  # 自動で隠したことが分かる文言
        assert view.toggle_text != "文字起こしを表示"

    @pytest.mark.parametrize("cramped", [False, True])
    def test_hidden_stays_hidden(self, cramped):
        from gui.live_view_logic import TranscriptPreference, transcript_view

        view = transcript_view(TranscriptPreference.HIDDEN, cramped=cramped)
        assert view.collapsed is True
        assert view.auto is False
        assert view.toggle_text == "文字起こしを表示"

    @pytest.mark.parametrize("cramped", [False, True])
    def test_shown_stays_shown(self, cramped):
        from gui.live_view_logic import TranscriptPreference, transcript_view

        view = transcript_view(TranscriptPreference.SHOWN, cramped=cramped)
        assert view.collapsed is False
        assert view.toggle_text == "文字起こしを隠す"

    @staticmethod
    def _toggle(pref, *, cramped, room=True):
        from gui.live_view_logic import toggle_preference

        return toggle_preference(pref, cramped=cramped, room_for_line=room)

    def test_toggle_from_shown_hides(self):
        from gui.live_view_logic import TranscriptPreference as P

        for pref, cramped in ((P.AUTO, False), (P.SHOWN, True), (P.SHOWN, False)):
            result = self._toggle(pref, cramped=cramped)
            assert result.preference is P.HIDDEN
            assert result.refused is False

    def test_toggle_from_auto_collapsed_forces_show_when_one_line_fits(self):
        from gui.live_view_logic import TranscriptPreference as P

        # 自動で隠れている時に押すと、見出し+1行分が入るなら狭くても表示し直せる
        result = self._toggle(P.AUTO, cramped=True, room=True)
        assert result.preference is P.SHOWN
        assert result.refused is False

    def test_toggle_from_auto_collapsed_refused_when_no_room(self):
        from gui.live_view_logic import TranscriptPreference as P

        # 1行も出せないなら SHOWN にしない(押しても何も見えない状態を作らない)
        result = self._toggle(P.AUTO, cramped=True, room=False)
        assert result.preference is P.AUTO
        assert result.refused is True

    def test_toggle_from_hidden(self):
        from gui.live_view_logic import TranscriptPreference as P

        assert self._toggle(P.HIDDEN, cramped=False).preference is P.AUTO
        # HIDDEN → SHOWN は見出し+1行分が入るときだけ
        assert self._toggle(P.HIDDEN, cramped=True, room=True).preference is P.SHOWN

    def test_toggle_from_hidden_without_room_falls_back_to_auto(self):
        from gui.live_view_logic import TranscriptPreference as P

        # 表示の希望は覚えておき(AUTO)、窓が広がれば自動で表示する
        result = self._toggle(P.HIDDEN, cramped=True, room=False)
        assert result.preference is P.AUTO
        assert result.refused is True

    @pytest.mark.parametrize("cramped", [False, True])
    def test_toggle_flips_visibility_when_line_fits(self, cramped):
        from gui.live_view_logic import TranscriptPreference, transcript_view

        for pref in TranscriptPreference:
            before = transcript_view(pref, cramped=cramped).collapsed
            after = self._toggle(pref, cramped=cramped, room=True).preference
            assert transcript_view(after, cramped=cramped).collapsed is (not before)

    def test_refused_toggle_text_tells_why(self):
        from gui.live_view_logic import TranscriptPreference as P
        from gui.live_view_logic import transcript_view

        view = transcript_view(P.AUTO, cramped=True, refused=True)
        assert view.collapsed is True
        assert "狭いため" in view.toggle_text
        assert "広げて" in view.toggle_text
        # 表示中なら「表示できません」は出さない
        shown = transcript_view(P.AUTO, cramped=False, refused=True)
        assert shown.toggle_text == "文字起こしを隠す"

    def test_room_for_line_uses_one_line_height(self):
        from gui.live_view_logic import has_room_for_line, textbox_height, transcript_min_height

        minimum = transcript_min_height(section_height=94, box_height=62, font_size=12)
        assert minimum == 94 - 62 + textbox_height(1, 12)
        kwargs = {"fixed_height": 28, "summary_min": 83, "transcript_min": minimum}
        assert has_room_for_line(28 + 83 + minimum, **kwargs) is True
        assert has_room_for_line(28 + 83 + minimum - 1, **kwargs) is False

    def test_transcript_min_height_invalid(self):
        from gui.live_view_logic import transcript_min_height

        with pytest.raises(ValueError):
            transcript_min_height(section_height=10, box_height=20, font_size=12)


class TestSettlePreference:
    """SHOWN(狭くても表示)は一時的な希望。窓が広がるか1行も入らなくなれば AUTO に戻す"""

    @staticmethod
    def _settle(pref, *, cramped, room=True):
        from gui.live_view_logic import settle_preference

        return settle_preference(pref, cramped=cramped, room_for_line=room)

    def test_shown_returns_to_auto_when_roomy(self):
        from gui.live_view_logic import TranscriptPreference as P

        assert self._settle(P.SHOWN, cramped=False) is P.AUTO

    def test_shown_kept_while_cramped_with_room(self):
        from gui.live_view_logic import TranscriptPreference as P

        assert self._settle(P.SHOWN, cramped=True, room=True) is P.SHOWN

    def test_shown_returns_to_auto_when_no_line_fits(self):
        from gui.live_view_logic import TranscriptPreference as P

        assert self._settle(P.SHOWN, cramped=True, room=False) is P.AUTO

    @pytest.mark.parametrize("cramped, room", [(False, True), (True, True), (True, False)])
    def test_hidden_and_auto_unchanged(self, cramped, room):
        from gui.live_view_logic import TranscriptPreference as P

        # 利用者が隠した(HIDDEN)は実行をまたいで保持する
        assert self._settle(P.HIDDEN, cramped=cramped, room=room) is P.HIDDEN
        assert self._settle(P.AUTO, cramped=cramped, room=room) is P.AUTO

    def test_invalid_preference(self):
        from gui.live_view_logic import toggle_preference, transcript_view

        with pytest.raises(TypeError):
            transcript_view("auto", cramped=False)
        with pytest.raises(TypeError):
            toggle_preference(None, cramped=False, room_for_line=True)
