"""live.summarizer のユニットテスト。"""

import itertools
import logging
import threading

import pytest

from live.summarizer import (
    RollingSummarizer,
    accept_summary,
    build_messages,
    should_summarize,
    strip_think,
)

GOOD_SUMMARY = "現在の話題: 予算の見直し\n進捗:\n- 来期予算の方針を共有した"
# 実機（qwen2.5:7b）で出力された中国語混入の要約
CHINESE_MIXED_SUMMARY = "現在の話題: とても簡単だった\n進発:\n- 过去的会议和讲义非常容易理解"


class TestBuildMessages:
    def test_structure_and_contents(self):
        messages = build_messages("前回の要約", "新しい発言", max_chars=100)
        assert [m["role"] for m in messages] == ["system", "user"]
        system = messages[0]["content"]
        assert "現在の話題" in system
        assert "進捗" in system
        assert "箇条書き" in system
        assert "入力に無い内容を追加しない" in system
        assert "プレーンテキスト" in system
        user = messages[1]["content"]
        assert "前回の要約" in user
        assert "新しい発言" in user

    def test_system_prompt_enforces_japanese_and_headings(self):
        system = build_messages("", "発言", max_chars=100)[0]["content"]
        assert "必ず日本語で出力する" in system
        assert "中国語・英語で書かない" in system
        assert "『現在の話題:』『進捗:』をこの表記のまま" in system
        assert "前回の要約をそのまま返す" in system

    def test_empty_prev_summary_is_marked(self):
        user = build_messages("", "発言", max_chars=100)[1]["content"]
        assert "（なし）" in user

    def test_new_text_within_limit_is_kept(self):
        text = "あ" * 10
        user = build_messages("", text, max_chars=10)[1]["content"]
        assert text in user

    def test_new_text_over_limit_keeps_tail(self):
        text = "古" * 5 + "新" * 10
        user = build_messages("", text, max_chars=10)[1]["content"]
        assert "新" * 10 in user
        assert "古" not in user

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            build_messages("", "x", max_chars=0)

    def test_returns_new_list_each_call(self):
        a = build_messages("p", "n", max_chars=10)
        b = build_messages("p", "n", max_chars=10)
        assert a == b
        assert a is not b


class TestStripThink:
    def test_removes_think_block(self):
        assert strip_think("<think>考え中</think>\n要約です") == "要約です"

    def test_removes_multiline_and_multiple_blocks(self):
        text = "<think>a\nb</think>前半<think>c</think>後半"
        assert strip_think(text) == "前半後半"

    def test_think_only_becomes_empty(self):
        assert strip_think("<think>全部思考</think>") == ""

    def test_unclosed_think_becomes_empty(self):
        assert strip_think("<think>途中で切れた") == ""

    def test_text_without_think_is_unchanged_except_whitespace(self):
        assert strip_think("  そのまま  ") == "そのまま"

    def test_empty(self):
        assert strip_think("") == ""


class TestAcceptSummary:
    def test_accepts_valid_summary(self):
        assert accept_summary(GOOD_SUMMARY, "") == GOOD_SUMMARY

    def test_strips_think_before_accepting(self):
        assert accept_summary(f"<think>x</think>{GOOD_SUMMARY}", "") == GOOD_SUMMARY

    def test_empty_returns_prev(self):
        assert accept_summary("", "前回") == "前回"

    def test_think_only_returns_prev(self):
        assert accept_summary("<think>" + "x" * 100 + "</think>", "前回") == "前回"

    def test_min_chars_boundary(self):
        assert accept_summary("a" * 20, "") == "a" * 20
        assert accept_summary("a" * 19, "") == ""

    def test_custom_min_chars(self):
        assert accept_summary("abc", "prev", min_chars=3, min_ratio=0.0) == "abc"

    def test_ratio_boundary_exact_is_accepted(self):
        prev = "p" * 100
        assert accept_summary("n" * 30, prev) == "n" * 30
        assert accept_summary("n" * 29, prev) == prev

    def test_ratio_not_applied_without_prev(self):
        assert accept_summary("n" * 20, "") == "n" * 20

    def test_none_new_returns_prev(self):
        assert accept_summary(None, "前回") == "前回"

    def test_chinese_mixed_summary_returns_prev(self):
        assert accept_summary(CHINESE_MIXED_SUMMARY, GOOD_SUMMARY) == GOOD_SUMMARY

    def test_chinese_mixed_first_summary_returns_empty_prev(self):
        assert accept_summary(CHINESE_MIXED_SUMMARY, "") == ""

    def test_kanji_dense_japanese_summary_is_accepted(self):
        summary = (
            "現在の話題: 会議と講義の議題整理\n"
            "進捗:\n"
            "- 前回会議の議題と講義資料の進捗を確認した\n"
            "- AWS / GPT-4o の検証は来週"
        )
        assert accept_summary(summary, "") == summary


_BASE = {
    "now": 100.0,
    "last_run": 70.0,
    "interval": 30.0,
    "backlog_sec": 5.0,
    "max_backlog_sec": 10.0,
    "in_flight": False,
    "has_new_text": True,
}


class TestShouldSummarize:
    @pytest.mark.parametrize(
        "elapsed_ok,backlog_ok,in_flight,has_new",
        list(itertools.product([True, False], repeat=4)),
    )
    def test_truth_table(self, elapsed_ok, backlog_ok, in_flight, has_new):
        args = {
            **_BASE,
            "last_run": 70.0 if elapsed_ok else 71.0,
            "backlog_sec": 10.0 if backlog_ok else 10.01,
            "in_flight": in_flight,
            "has_new_text": has_new,
        }
        expected = elapsed_ok and backlog_ok and not in_flight and has_new
        assert should_summarize(**args) is expected

    def test_never_run_is_due(self):
        assert should_summarize(**{**_BASE, "last_run": None}) is True

    def test_interval_boundary(self):
        assert should_summarize(**{**_BASE, "last_run": 70.0}) is True
        assert should_summarize(**{**_BASE, "last_run": 70.5}) is False


class FakeBackend:
    def __init__(self, responses=None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0) if self.responses else GOOD_SUMMARY

    def list_models(self):
        return ["m"]


def _user_content(call):
    return call[1]["content"]


def _make(backend, **kwargs):
    defaults = {"interval_sec": 30.0, "max_backlog_sec": 10.0, "max_chars": 1000}
    return RollingSummarizer(backend, "m", **{**defaults, **kwargs})


class TestRollingSummarizer:
    def test_no_call_without_new_text(self):
        backend = FakeBackend()
        summarizer = _make(backend)
        assert summarizer.maybe_run(now=0.0) is False
        assert backend.calls == []

    def test_first_run_summarizes_added_text(self):
        backend = FakeBackend()
        summarizer = _make(backend)
        summarizer.add_text("最初の発言")

        assert summarizer.maybe_run(now=0.0) is True
        assert summarizer.summary == GOOD_SUMMARY
        assert "最初の発言" in _user_content(backend.calls[0])

    def test_ignores_blank_text(self):
        backend = FakeBackend()
        summarizer = _make(backend)
        summarizer.add_text("   ")
        assert summarizer.maybe_run(now=0.0) is False

    def test_sends_only_text_since_last_success_and_prev_summary(self):
        first = GOOD_SUMMARY
        second = "現在の話題: 採用\n進捗:\n- 採用計画を議論した。面接日程を決めた"
        backend = FakeBackend(responses=[first, second])
        summarizer = _make(backend)
        summarizer.add_text("発言A")
        summarizer.maybe_run(now=0.0)
        summarizer.add_text("発言B")

        assert summarizer.maybe_run(now=30.0) is True
        content = _user_content(backend.calls[1])
        assert "発言B" in content
        assert "発言A" not in content
        assert first in content
        assert summarizer.summary == second

    def test_respects_interval(self):
        backend = FakeBackend()
        summarizer = _make(backend)
        summarizer.add_text("A")
        summarizer.maybe_run(now=0.0)
        summarizer.add_text("B")
        assert summarizer.maybe_run(now=29.9) is False
        assert len(backend.calls) == 1

    def test_skips_when_backlog_too_large(self):
        backend = FakeBackend()
        summarizer = _make(backend)
        summarizer.add_text("A")
        assert summarizer.maybe_run(now=0.0, backlog_sec=10.5) is False
        assert backend.calls == []

    def test_uses_injected_clock(self):
        times = iter([0.0, 10.0, 31.0])
        backend = FakeBackend()
        summarizer = _make(backend, clock=lambda: next(times))
        summarizer.add_text("A")
        assert summarizer.maybe_run() is True
        summarizer.add_text("B")
        assert summarizer.maybe_run() is False
        assert summarizer.maybe_run() is True

    def test_failure_keeps_summary_and_pending_and_warns(self):
        warnings: list[str] = []
        backend = FakeBackend()
        summarizer = _make(backend, on_warning=warnings.append)
        summarizer.add_text("A")
        summarizer.maybe_run(now=0.0)

        backend.error = ConnectionError("down")
        summarizer.add_text("B")
        assert summarizer.maybe_run(now=30.0) is False  # 例外を外に出さない
        assert summarizer.summary == GOOD_SUMMARY
        assert len(warnings) == 1
        assert summarizer.in_flight is False

        # 失敗時は last_run を更新しているので直後には再送しない
        assert summarizer.maybe_run(now=31.0) is False
        backend.error = None
        assert summarizer.maybe_run(now=60.0) is True
        assert "B" in _user_content(backend.calls[-1])

    def test_failure_without_warning_callback_does_not_raise(self):
        summarizer = _make(FakeBackend(error=RuntimeError("x")))
        summarizer.add_text("A")
        assert summarizer.maybe_run(now=0.0) is False
        assert summarizer.summary == ""

    def test_rejected_summary_keeps_prev_and_pending(self):
        warnings: list[str] = []
        backend = FakeBackend(responses=[GOOD_SUMMARY, "<think>だけ</think>", GOOD_SUMMARY])
        summarizer = _make(backend, on_warning=warnings.append)
        summarizer.add_text("A")
        summarizer.maybe_run(now=0.0)
        summarizer.add_text("B")

        assert summarizer.maybe_run(now=30.0) is False
        assert summarizer.summary == GOOD_SUMMARY
        assert len(warnings) == 1
        summarizer.maybe_run(now=60.0)
        assert "B" in _user_content(backend.calls[-1])

    def test_text_added_during_flight_survives(self):
        backend = FakeBackend()
        summarizer = _make(backend)

        def complete_and_add(messages):
            backend.calls.append(list(messages))
            summarizer.add_text("飛行中の発言")
            return GOOD_SUMMARY

        backend.complete = complete_and_add
        summarizer.add_text("消費済みの発言")
        summarizer.maybe_run(now=0.0)

        assert summarizer.has_pending_text is True
        summarizer.maybe_run(now=30.0)
        content = _user_content(backend.calls[1])
        assert "飛行中の発言" in content
        assert "消費済みの発言" not in content

    def test_no_double_send_while_in_flight(self):
        entered = threading.Event()
        release = threading.Event()
        calls: list[int] = []

        class BlockingBackend(FakeBackend):
            def complete(self, messages):
                calls.append(1)
                entered.set()
                release.wait(5)
                return GOOD_SUMMARY

        summarizer = _make(BlockingBackend(), interval_sec=0.0)
        summarizer.add_text("A")
        worker = threading.Thread(target=summarizer.maybe_run, kwargs={"now": 0.0})
        worker.start()
        try:
            assert entered.wait(5)
            assert summarizer.in_flight is True
            summarizer.add_text("B")
            assert summarizer.maybe_run(now=100.0) is False
        finally:
            release.set()
            worker.join(5)
        assert len(calls) == 1
        assert summarizer.in_flight is False

    def test_pending_is_bounded_to_max_chars(self):
        backend = FakeBackend()
        summarizer = _make(backend, max_chars=10, interval_sec=0.0)
        for i in range(100):
            summarizer.add_text(f"発言{i:03d}")
        assert summarizer.pending_chars <= 10

        summarizer.maybe_run(now=0.0)
        content = _user_content(backend.calls[-1])
        assert "発言099" in content
        assert "発言000" not in content
        assert summarizer.pending_chars == 0

    def test_identical_summary_is_accepted_and_consumes_pending(self):
        backend = FakeBackend(responses=[GOOD_SUMMARY, GOOD_SUMMARY])
        summarizer = _make(backend)
        summarizer.add_text("A")
        summarizer.maybe_run(now=0.0)
        summarizer.add_text("B")
        assert summarizer.maybe_run(now=30.0) is True
        assert summarizer.has_pending_text is False

    def test_warning_callback_error_is_swallowed(self):
        def bad_warning(message):
            raise RuntimeError("callback broken")

        summarizer = _make(FakeBackend(error=RuntimeError("x")), on_warning=bad_warning)
        summarizer.add_text("A")
        assert summarizer.maybe_run(now=0.0) is False

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            _make(FakeBackend(), max_chars=0)

    def test_chinese_summary_keeps_prev_and_warns_without_body(self, caplog):
        warnings: list[str] = []
        backend = FakeBackend(responses=[GOOD_SUMMARY, CHINESE_MIXED_SUMMARY])
        summarizer = _make(backend, on_warning=warnings.append)
        summarizer.add_text("最初の発言")
        summarizer.maybe_run(now=0.0)
        summarizer.add_text("崩れた文字起こし")

        with caplog.at_level(logging.DEBUG, logger="voicenote"):
            assert summarizer.maybe_run(now=30.0) is False
        assert summarizer.summary == GOOD_SUMMARY
        assert summarizer.has_pending_text is True
        assert len(warnings) == 1
        assert "日本語" in warnings[0]
        for text in (warnings[0], caplog.text):
            assert "过去" not in text
            assert "崩れた文字起こし" not in text

    @pytest.mark.parametrize(
        "bad_output", ["", "<think>x</think>", "   ", "短すぎる", CHINESE_MIXED_SUMMARY]
    )
    def test_rejected_first_summary_keeps_pending(self, bad_output):
        """前回要約が空の初回でも、不正な出力で未要約テキストを失わない。"""
        warnings: list[str] = []
        backend = FakeBackend(responses=[bad_output, GOOD_SUMMARY])
        summarizer = _make(backend, on_warning=warnings.append)
        summarizer.add_text("最初の発言")

        assert summarizer.maybe_run(now=0.0) is False
        assert summarizer.summary == ""
        assert summarizer.has_pending_text is True
        assert len(warnings) == 1

        assert summarizer.maybe_run(now=30.0) is True
        assert "最初の発言" in _user_content(backend.calls[-1])
