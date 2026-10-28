"""gui/live_context_logic.py のテスト(参加者・メモ入力の Tk 非依存ロジック)"""

import pytest

from gui.live_context_logic import (
    DEBOUNCE_MS,
    ContextSync,
    Debouncer,
    build_context,
    limit_notice,
    should_push,
)
from live.context import (
    MAX_MEMO_CHARS,
    MAX_PARTICIPANT_CHARS,
    MAX_PARTICIPANTS,
    LiveContext,
    parse_participants,
)


class FakeScheduler:
    """Tk の after/after_cancel 相当(時間は手動で進める)"""

    def __init__(self):
        self.pending: dict[int, tuple[int, object]] = {}
        self.next_id = 0
        self.cancelled: list[int] = []

    def schedule(self, delay_ms, fn):
        self.next_id += 1
        self.pending[self.next_id] = (delay_ms, fn)
        return self.next_id

    def cancel(self, token):
        self.cancelled.append(token)
        self.pending.pop(token, None)

    def fire_all(self):
        items = list(self.pending.items())
        self.pending.clear()
        for _token, (_delay, fn) in items:
            fn()


def _names(n: int) -> str:
    return "、".join(f"人{i}" for i in range(n))


class TestBuildContext:
    def test_from_texts(self):
        ctx = build_context("田中、佐藤", "議題A")
        assert ctx == LiveContext.from_inputs("田中、佐藤", "議題A")
        assert ctx.participants == ("田中", "佐藤")

    @pytest.mark.parametrize("participants, memo", [(None, None), ("", ""), ("  ", "\n ")])
    def test_empty(self, participants, memo):
        assert build_context(participants, memo).is_empty


class TestShouldPush:
    def test_first_push(self):
        assert should_push(None, LiveContext()) is True

    def test_same_value_skipped(self):
        assert should_push(build_context("a", "m"), build_context("a", "m")) is False

    def test_changed(self):
        assert should_push(build_context("a", "m"), build_context("a", "m2")) is True

    def test_whitespace_only_change_skipped(self):
        # 正規化後に同じなら送り直さない
        assert should_push(build_context("a", "m"), build_context(" a ", "m\n")) is False


class TestLimitNotice:
    def test_none_within_limits(self):
        assert limit_notice("田中、佐藤", "短いメモ") is None

    def test_none_for_empty(self):
        assert limit_notice(None, None) is None
        assert limit_notice("", "") is None

    def test_memo_exactly_limit(self):
        assert limit_notice("", "あ" * MAX_MEMO_CHARS) is None

    def test_memo_over_limit(self):
        notice = limit_notice("", "あ" * (MAX_MEMO_CHARS + 1))
        assert notice is not None
        assert f"先頭{MAX_MEMO_CHARS}文字" in notice

    def test_memo_uses_same_criterion_as_context(self):
        # 前後の空白は数えない(memo_truncated と同じ基準)
        memo = "  " + "あ" * MAX_MEMO_CHARS + "\n\n  "
        assert LiveContext.from_inputs("", memo).memo_truncated is False
        assert limit_notice("", memo) is None

    def test_participants_exactly_limit(self):
        assert limit_notice(_names(MAX_PARTICIPANTS), "") is None

    def test_participants_over_limit(self):
        notice = limit_notice(_names(MAX_PARTICIPANTS + 1), "")
        assert notice is not None
        assert f"{MAX_PARTICIPANTS}人" in notice

    def test_duplicates_not_counted(self):
        text = _names(MAX_PARTICIPANTS) + "、人0、人1"
        assert limit_notice(text, "") is None

    def test_long_name(self):
        notice = limit_notice("あ" * (MAX_PARTICIPANT_CHARS + 1), "")
        assert notice is not None
        assert f"{MAX_PARTICIPANT_CHARS}文字" in notice

    def test_name_exactly_limit(self):
        assert limit_notice("あ" * MAX_PARTICIPANT_CHARS, "") is None

    def test_multiple_notices_joined(self):
        notice = limit_notice(_names(MAX_PARTICIPANTS + 1), "あ" * (MAX_MEMO_CHARS + 1))
        assert notice is not None
        assert f"{MAX_PARTICIPANTS}人" in notice and f"{MAX_MEMO_CHARS}文字" in notice

    def test_notice_has_no_content(self):
        # 入力本文は表示文言に含めない
        notice = limit_notice("秘密太郎" * 20, "機密メモ" * 600)
        assert notice is not None
        assert "秘密" not in notice and "機密" not in notice

    @pytest.mark.parametrize(
        "text",
        ["a,b、c，d\ne", "a、a、b", " 、 ,\n", "x\r\ny", _names(MAX_PARTICIPANTS)],
    )
    def test_split_matches_parse_participants(self, text):
        # 上限以下では、通知の数え方が parse_participants と一致する
        from gui.live_context_logic import count_participants

        assert count_participants(text) == len(parse_participants(text))

    def test_count_participants_none(self):
        from gui.live_context_logic import count_participants

        assert count_participants(None) == 0


class TestDebouncer:
    def test_trigger_schedules_with_delay(self):
        sched = FakeScheduler()
        calls = []
        deb = Debouncer(sched.schedule, sched.cancel, lambda: calls.append(1))
        deb.trigger()
        assert deb.pending is True
        assert list(sched.pending.values())[0][0] == DEBOUNCE_MS
        assert calls == []
        sched.fire_all()
        assert calls == [1]
        assert deb.pending is False

    def test_retrigger_cancels_previous(self):
        sched = FakeScheduler()
        calls = []
        deb = Debouncer(sched.schedule, sched.cancel, lambda: calls.append(1), delay_ms=100)
        deb.trigger()
        deb.trigger()
        deb.trigger()
        assert len(sched.pending) == 1
        assert len(sched.cancelled) == 2
        sched.fire_all()
        assert calls == [1]

    def test_flush_runs_pending_now(self):
        sched = FakeScheduler()
        calls = []
        deb = Debouncer(sched.schedule, sched.cancel, lambda: calls.append(1))
        deb.trigger()
        assert deb.flush() is True
        assert calls == [1]
        assert sched.pending == {}
        assert deb.pending is False

    def test_flush_without_pending(self):
        sched = FakeScheduler()
        calls = []
        deb = Debouncer(sched.schedule, sched.cancel, lambda: calls.append(1))
        assert deb.flush() is False
        assert calls == []

    def test_cancel(self):
        sched = FakeScheduler()
        calls = []
        deb = Debouncer(sched.schedule, sched.cancel, lambda: calls.append(1))
        deb.trigger()
        deb.cancel()
        assert deb.pending is False
        assert sched.pending == {}
        deb.cancel()  # 二重でも安全
        assert calls == []

    @pytest.mark.parametrize("delay", [0, -1])
    def test_invalid_delay(self, delay):
        sched = FakeScheduler()
        with pytest.raises(ValueError):
            Debouncer(sched.schedule, sched.cancel, lambda: None, delay_ms=delay)


class TestCompactLimitNotice:
    """小さい画面(compact)では、複数の上限案内を1行に省略する"""

    _ALL_OVER = (
        "、".join(f"人{i}" for i in range(MAX_PARTICIPANTS + 5))
        + "、"
        + "あ" * (MAX_PARTICIPANT_CHARS + 1),
        "m" * (MAX_MEMO_CHARS + 1),
    )

    def test_compact_all_three_is_short(self):
        notice = limit_notice(*self._ALL_OVER, compact=True)
        assert notice is not None
        # 幅 560px・12pt で1行に収まる長さ(全角換算 40 文字以内)
        assert len(notice) <= 40
        assert f"{MAX_PARTICIPANTS}人" in notice
        assert f"{MAX_PARTICIPANT_CHARS}文字" in notice
        assert f"{MAX_MEMO_CHARS}文字" in notice

    def test_compact_lists_only_exceeded_limits(self):
        notice = limit_notice("", "m" * (MAX_MEMO_CHARS + 1), compact=True)
        assert notice is not None
        assert f"{MAX_MEMO_CHARS}文字" in notice
        assert "人" not in notice

    def test_compact_none_within_limits(self):
        assert limit_notice("田中", "メモ", compact=True) is None

    def test_compact_does_not_include_input_text(self):
        notice = limit_notice("秘密太郎、" + _names(MAX_PARTICIPANTS + 1), "機密", compact=True)
        assert notice is not None
        assert "秘密" not in notice and "機密" not in notice

    def test_default_is_full_text(self):
        full = limit_notice(*self._ALL_OVER)
        assert full is not None
        assert full.count(" / ") == 2

    def test_context_sync_uses_compact_notice(self):
        notices: list = []
        sched = FakeScheduler()
        sync = ContextSync(
            read_inputs=lambda: self._ALL_OVER,
            push=lambda _ctx: None,
            on_notice=notices.append,
            schedule=sched.schedule,
            cancel=sched.cancel,
            compact_notice=True,
        )
        sync.flush()
        assert notices == [limit_notice(*self._ALL_OVER, compact=True)]


class SyncHarness:
    def __init__(self, participants="", memo=""):
        self.inputs = (participants, memo)
        self.pushed: list = []
        self.notices: list = []
        self.sched = FakeScheduler()
        self.sync = ContextSync(
            read_inputs=lambda: self.inputs,
            push=self.pushed.append,
            on_notice=self.notices.append,
            schedule=self.sched.schedule,
            cancel=self.sched.cancel,
        )


class TestContextSync:
    def test_change_is_debounced(self):
        h = SyncHarness("田中", "")
        h.sync.on_input_changed()
        assert h.pushed == []
        h.sched.fire_all()
        assert h.pushed == [build_context("田中", "")]
        assert h.notices == [None]

    def test_burst_of_changes_pushes_once(self):
        h = SyncHarness()
        for text in ("田", "田中", "田中、佐"):
            h.inputs = (text, "")
            h.sync.on_input_changed()
        h.sched.fire_all()
        assert h.pushed == [build_context("田中、佐", "")]

    def test_unchanged_value_not_pushed_again(self):
        h = SyncHarness("田中", "m")
        h.sync.on_input_changed()
        h.sched.fire_all()
        h.inputs = ("田中 ", "m")
        h.sync.on_input_changed()
        h.sched.fire_all()
        assert len(h.pushed) == 1

    def test_flush_pushes_latest_without_waiting(self):
        h = SyncHarness("田中", "")
        h.sync.on_input_changed()
        h.inputs = ("田中、佐藤", "議題")
        h.sync.flush()
        assert h.pushed == [build_context("田中、佐藤", "議題")]
        assert h.sched.pending == {}

    def test_flush_force_pushes_even_if_same(self):
        h = SyncHarness("田中", "")
        h.sync.flush()
        h.sync.flush(force=True)
        h.sync.flush()
        assert len(h.pushed) == 2

    def test_clear_pushes_none_and_cancels(self):
        h = SyncHarness("田中", "メモ")
        h.sync.on_input_changed()
        h.sync.clear()
        assert h.pushed == [None]
        assert h.sched.pending == {}
        assert h.notices[-1] is None

    def test_after_clear_empty_inputs_not_resent(self):
        h = SyncHarness("田中", "メモ")
        h.sync.flush()
        h.sync.clear()
        h.inputs = ("", "")
        h.sync.on_input_changed()  # クリアで入力欄を空にしたことによる変更通知
        h.sched.fire_all()
        assert h.pushed == [build_context("田中", "メモ"), None]

    def test_notice_reported(self):
        h = SyncHarness("", "あ" * (MAX_MEMO_CHARS + 5))
        h.sync.flush()
        assert h.notices[-1] is not None
        assert h.pushed[-1].memo_truncated is True

    def test_cancel(self):
        h = SyncHarness("田中", "")
        h.sync.on_input_changed()
        h.sync.cancel()
        h.sched.fire_all()
        assert h.pushed == []

    def test_push_error_is_swallowed(self, caplog):
        def boom(_ctx):
            raise RuntimeError("down")

        sched = FakeScheduler()
        notices = []
        sync = ContextSync(
            read_inputs=lambda: ("秘密太郎", "機密メモ"),
            push=boom,
            on_notice=notices.append,
            schedule=sched.schedule,
            cancel=sched.cancel,
        )
        sync.on_input_changed()
        sched.fire_all()  # 例外を Tk の after に漏らさない
        assert "秘密" not in caplog.text and "機密" not in caplog.text
        # 失敗した値は「送信済み」にしない(次の変更・開始時に再送する)
        sync.flush()
        assert len(notices) == 2


class TestFlushPending:
    """停止・中断の直前に呼ぶ: デバウンス待ちの編集だけを今すぐ送る"""

    def test_pending_edit_is_pushed_now(self):
        h = SyncHarness("田中", "")
        h.sync.on_input_changed()
        h.inputs = ("田中、佐藤", "議題")
        assert h.sync.flush_pending(force=True) is True
        assert h.pushed == [build_context("田中、佐藤", "議題")]
        assert h.sched.pending == {}

    def test_force_pushes_even_if_same_as_last(self):
        h = SyncHarness("田中", "")
        h.sync.flush()
        h.sync.on_input_changed()
        assert h.sync.flush_pending(force=True) is True
        assert len(h.pushed) == 2

    def test_nothing_pending_sends_nothing(self):
        h = SyncHarness("田中", "")
        h.sync.flush()
        assert h.sync.flush_pending(force=True) is False
        assert len(h.pushed) == 1

    def test_nothing_pending_before_any_edit(self):
        h = SyncHarness("田中", "")
        assert h.sync.flush_pending(force=True) is False
        assert h.pushed == []
        assert h.notices == []


class TestLongNameNoticeUsesNormalizedLength:
    """上限案内の「切り詰めます」は保存値(正規化後)と食い違わない"""

    def test_invisible_chars_do_not_count(self):
        # 40文字 + ゼロ幅スペース: 保存値は40文字で切り詰められないので案内も出さない
        name = "あ" * MAX_PARTICIPANT_CHARS + "​"
        assert build_context(name, "").participants == ("あ" * MAX_PARTICIPANT_CHARS,)
        assert limit_notice(name, "") is None

    def test_control_chars_do_not_count(self):
        name = "\x07" + "あ" * MAX_PARTICIPANT_CHARS
        assert limit_notice(name, "") is None

    def test_really_long_name_still_noticed(self):
        name = "あ" * (MAX_PARTICIPANT_CHARS + 1) + "​"
        notice = limit_notice(name, "")
        assert notice is not None and "切り詰め" in notice
