"""gui/live_workflow.py のテスト: 開始前の検証・バナー文言・準備(PREPARING)と取り消し

Fake 群と Harness は tests/live_workflow_fakes.py にある。
"""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

from config import VoiceNoteConfig
from gui.live_state import LivePhase
from gui.live_workflow import (
    BANNER_ENGINE_FALLBACK,
    BANNER_EXTERNAL_SEND,
    BANNER_MODEL_DOWNLOAD,
    BANNER_OLLAMA,
    BANNER_VAD_FALLBACK,
    MeterReading,
    format_download_progress,
    validate_live_start,
)
from gui.live_workflow_support import external_send_banner_text
from tests.live_workflow_fakes import (
    DeferredThread,
    FakeClient,
    FakeFetcher,
    Harness,
    probe_returning,
    reset_deferred_threads,
    run_pending,
)


@pytest.fixture(autouse=True)
def _reset_deferred():
    reset_deferred_threads()
    yield
    reset_deferred_threads()


# ──────────────── validate_live_start ────────────────


class TestValidateLiveStart:
    def test_ok(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        assert validate_live_start(config, "[1] マイク", mic_busy=False) is None

    def test_requires_save_folder(self):
        error = validate_live_start(VoiceNoteConfig(), "[1] マイク", mic_busy=False)
        assert error is not None
        assert error[0] == "設定が必要"

    def test_rejects_missing_device(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(config, "デバイスなし", mic_busy=False)
        assert error is not None
        assert "デバイス" in error[0]

    def test_rejects_when_mic_busy(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(config, "[1] マイク", mic_busy=True)
        assert error is not None
        assert "実行中" in error[1]

    def test_rejects_while_normal_stream_closing(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(config, "[1] マイク", mic_busy=False, stream_closing=True)
        assert error is not None
        assert error[0] == "ストリーム解放中"
        assert "解放" in error[1]

    def test_mic_busy_takes_priority_over_stream_closing(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(config, "[1] マイク", mic_busy=True, stream_closing=True)
        assert error is not None and error[0] == "実行中"

    def test_rejects_invalid_live_settings(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_engine="bogus")
        error = validate_live_start(config, "[1] マイク", mic_busy=False)
        assert error is not None
        assert "bogus" in error[1]


class TestFormatDownloadProgress:
    def test_unknown_total(self):
        assert "small" in format_download_progress("small", 0, 0)

    def test_with_total_shows_percent(self):
        text = format_download_progress("small", 240_000_000, 480_000_000)
        assert "50%" in text
        assert "240 MB" in text and "480 MB" in text

    def test_gigabytes(self):
        text = format_download_progress("large", 800_000_000, 1_600_000_000)
        assert "1.6 GB" in text


class TestExternalSendBannerText:
    def test_openai_transcription_only_mentions_audio(self):
        text = external_send_banner_text("openai", "rule")
        assert text is not None
        assert "このPCの中だけ" in text
        assert "「文字起こしだけする」モード" in text
        assert "音声が OpenAI に送信" in text
        assert "本文" not in text
        assert "設定ダイアログ" in text

    def test_llm_format_only_mentions_text(self):
        text = external_send_banner_text("local", "llm")
        assert text is not None
        assert "文字起こし本文が OpenAI に送信" in text
        assert "音声" not in text

    def test_both_mention_audio_and_text(self):
        text = external_send_banner_text("openai", "llm")
        assert text is not None
        assert "音声と文字起こし本文が OpenAI に送信" in text

    def test_local_settings_have_no_banner(self):
        assert external_send_banner_text("local", "rule") is None

    def test_does_not_use_cli_wording(self):
        text = external_send_banner_text("openai", "llm")
        assert "--config" not in text
        assert "この実行は" not in text


# ──────────────── PREPARING ────────────────


class TestPreparing:
    def test_happy_path_starts_recording(self, tmp_path):
        h = Harness(tmp_path)

        assert h.start() is None

        assert h.state.phase is LivePhase.RECORDING
        assert h.wf.is_active is True
        assert h.spy.recording_started == 1
        assert h.log == ["session.start", "recorder.start"]
        recorder = h.recorders[0]
        assert recorder.device_id == 1 and recorder.dest == tmp_path
        assert any("ライブ" in m and "[1] マイク" in m for m in h.spy.logs)

    def test_preparing_phase_is_reported_before_recording(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)

        assert h.start() is None
        assert h.state.phase is LivePhase.PREPARING
        assert h.wf.is_active is True
        run_pending()
        assert h.state.phase is LivePhase.RECORDING
        phases = [s.phase for s in h.spy.states]
        assert phases.index(LivePhase.PREPARING) < phases.index(LivePhase.RECORDING)

    def test_chunk_sink_feeds_level_meter_and_session(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        chunk = np.full((512, 1), 0.1, dtype=np.float32)

        h.recorders[0].chunk_sink(chunk)

        assert len(h.sessions[0].fed) == 1
        reading = h.wf.poll_meter()
        assert isinstance(reading, MeterReading)
        assert reading.level_db == pytest.approx(-20.0, abs=0.1)
        assert reading.backlog_sec == 1.5
        assert reading.dropped_chunks == 3

    def test_session_receives_engine_vad_backend(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        session = h.sessions[0]
        session.engine.load()
        assert h.engine_api.created[0].loaded is True
        assert session.engine.transcribe(None, "ja") == "faster-whisper:ja"
        assert session.vad.kind == "silero"
        assert isinstance(session.backend, FakeClient)
        assert session.backend.timeout is None  # 本番クライアントは既定タイムアウト

    def test_model_download_success_reports_progress_and_clears_banner(self, tmp_path):
        fetcher = FakeFetcher(cached=False, progress=[(0, 0), (240_000_000, 480_000_000)])
        h = Harness(tmp_path, fetcher=fetcher)

        h.start()

        assert len(fetcher.fetched) == 1
        texts = [b.text for s in h.spy.states for b in s.banners if b.kind == BANNER_MODEL_DOWNLOAD]
        assert any("50%" in t for t in texts)
        assert h.banner(BANNER_MODEL_DOWNLOAD) is None
        assert h.state.phase is LivePhase.RECORDING

    def test_cached_model_is_not_fetched(self, tmp_path):
        fetcher = FakeFetcher(cached=True)
        h = Harness(tmp_path, fetcher=fetcher)
        h.start()
        assert fetcher.fetched == []
        assert all(b.kind != BANNER_MODEL_DOWNLOAD for s in h.spy.states for b in s.banners)

    def test_progress_updates_are_throttled(self, tmp_path):
        progress = [(i * 1_000_000, 100_000_000) for i in range(1, 50)]
        fetcher = FakeFetcher(cached=False, progress=progress)
        h = Harness(tmp_path, fetcher=fetcher)
        h.start()
        texts = {b.text for s in h.spy.states for b in s.banners if b.kind == BANNER_MODEL_DOWNLOAD}
        # 時計が進まないので最初の通知以外は間引かれる
        assert len(texts) <= 3

    def test_model_download_failure_does_not_start_recording(self, tmp_path):
        fetcher = FakeFetcher(cached=False, fail=ConnectionError("offline"))
        h = Harness(tmp_path, fetcher=fetcher)

        h.start()

        assert h.recorders == []
        assert h.sessions == []
        assert h.state.phase is LivePhase.IDLE
        assert h.wf.is_active is False
        assert h.spy.recording_started == 0
        assert len(h.spy.errors) == 1
        assert "ネットワーク" in h.spy.errors[0]
        assert h.banner(BANNER_MODEL_DOWNLOAD) is None

    def test_disk_full_message_uses_required_bytes(self, tmp_path):
        import errno

        fetcher = FakeFetcher(
            cached=False,
            fail=OSError(errno.ENOSPC, "No space"),
            progress=[(0, 1_600_000_000)],
        )
        h = Harness(tmp_path, fetcher=fetcher)
        h.start()
        assert "1.6 GB" in h.spy.errors[0]

    def test_unknown_model_is_setup_error(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_whisper_model="unknown-model")
        h = Harness(tmp_path, config=config)
        h.start()
        assert h.recorders == []
        assert h.state.phase is LivePhase.IDLE
        assert "unknown-model" in h.spy.errors[0]

    def test_mlx_fallback_shows_banner_and_fetches_fallback_model(self, tmp_path):
        config = VoiceNoteConfig(
            save_folder=str(tmp_path), live_engine="mlx", live_whisper_model="large-v3-turbo"
        )
        h = Harness(tmp_path, config=config, unavailable={"mlx"})

        h.start()

        banner = h.banner(BANNER_ENGINE_FALLBACK)
        assert banner is not None and "faster-whisper" in banner.text
        assert h.fetcher.resolved == [("faster-whisper", "small")]
        assert h.state.phase is LivePhase.RECORDING

    def test_engine_unavailable_without_fallback_is_error(self, tmp_path):
        h = Harness(tmp_path, unavailable={"faster-whisper"})
        h.start()
        assert h.recorders == []
        assert h.state.phase is LivePhase.IDLE
        assert "音声認識エンジン" in h.spy.errors[0]

    def test_load_time_fallback_warning_becomes_banner(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_engine="mlx")
        h = Harness(tmp_path, config=config, load_fails={"mlx"})
        h.start()
        assert h.banner(BANNER_ENGINE_FALLBACK) is None

        h.sessions[0].engine.load()  # asr スレッドでの load() 時に mlx が使えないと判明

        assert "faster-whisper" in h.banner(BANNER_ENGINE_FALLBACK).text
        assert h.sessions[0].engine.transcribe(None, "ja") == "faster-whisper:ja"

    def test_engine_use_is_serialized_across_runs(self, tmp_path):
        """放棄した前回の asr スレッドと次の実行の asr が同時にエンジンを使わない"""
        h = Harness(tmp_path)
        h.start()
        h.wf.stop()
        h.start()
        first, second = h.engine_api.created
        first.load_gate = threading.Event()
        old = threading.Thread(target=h.sessions[0].engine.load, daemon=True)
        new = threading.Thread(target=h.sessions[1].engine.load, daemon=True)
        try:
            old.start()
            assert first.load_entered.wait(timeout=2.0)
            new.start()
            assert not second.load_entered.wait(timeout=0.2)
        finally:
            first.load_gate.set()
            old.join(timeout=2.0)
            new.join(timeout=2.0)
        assert second.load_entered.is_set()
        assert not old.is_alive() and not new.is_alive()

    @pytest.mark.parametrize(
        ("status", "message", "hint"),
        [
            ("server_down", "接続できません。`ollama serve` を起動してください。", "ollama serve"),
            ("model_missing", "`ollama pull qwen2.5:7b` を実行してください。", "ollama pull"),
        ],
    )
    def test_ollama_unavailable_shows_banner_and_records(self, tmp_path, status, message, hint):
        h = Harness(tmp_path, probe=probe_returning(status, message))

        h.start()

        banner = h.banner(BANNER_OLLAMA)
        assert banner is not None
        assert hint in banner.text and "要約なし" in banner.text
        assert h.sessions[0].backend is None
        assert h.state.phase is LivePhase.RECORDING

    def test_invalid_llm_url_shows_banner(self, tmp_path):
        config = VoiceNoteConfig(
            save_folder=str(tmp_path), live_llm_base_url="http://example.com/v1"
        )

        def client_factory(*args, **kwargs):
            raise ValueError("ループバック以外の URL は使えません")

        h = Harness(tmp_path, config=config)
        h.wf._deps = replace(h.wf._deps, client_factory=client_factory)
        h.start()
        assert "ループバック" in h.banner(BANNER_OLLAMA).text
        assert h.sessions[0].backend is None

    def test_llm_disabled_has_no_banner(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_llm_enabled=False)
        h = Harness(tmp_path, config=config)
        h.start()
        assert h.banner(BANNER_OLLAMA) is None
        assert h.sessions[0].backend is None

    def test_vad_fallback_shows_banner_and_records(self, tmp_path):
        h = Harness(tmp_path, vad_fallback=True)
        h.start()
        assert "energy" in h.banner(BANNER_VAD_FALLBACK).text
        assert h.state.phase is LivePhase.RECORDING

    def test_external_send_banner(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), transcription_mode="openai")
        h = Harness(tmp_path, config=config)
        h.start()
        assert "OpenAI" in h.banner(BANNER_EXTERNAL_SEND).text

    def test_external_send_banner_uses_gui_wording(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), format_mode="llm")
        h = Harness(tmp_path, config=config)
        h.start()
        assert h.banner(BANNER_EXTERNAL_SEND).text == external_send_banner_text("local", "llm")

    def test_no_external_send_banner_for_local(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        assert h.banner(BANNER_EXTERNAL_SEND) is None

    def test_recorder_start_failure_releases_session(self, tmp_path):
        h = Harness(tmp_path, fail_start=True)

        h.start()

        assert h.state.phase is LivePhase.IDLE
        assert h.wf.is_active is False
        assert h.sessions[0].stop_timeouts  # 起動済みセッションを解放する
        assert "デバイスが使用できません" in h.spy.errors[0]
        assert h.wf.poll_meter() is None

    def test_unexpected_exception_returns_to_idle(self, tmp_path):
        h = Harness(tmp_path, session_factory_error=MemoryError("boom"))
        h.start()
        assert h.state.phase is LivePhase.IDLE
        assert h.recorders == []
        assert len(h.spy.errors) == 1

    def test_restart_after_failure_is_allowed(self, tmp_path):
        h = Harness(tmp_path, fail_start=True)
        h.start()
        h.fail_start = False
        assert h.start() is None
        assert h.state.phase is LivePhase.RECORDING

    def test_new_run_resets_view_state(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.sessions[0].callbacks.on_segment(0, 0.0, 1.0, "前回")
        h.wf.stop()
        h.start()
        assert h.state.segments == ()

    def test_thread_creation_failure_returns_error(self, tmp_path):
        def failing(*args, **kwargs):
            raise RuntimeError("スレッドを生成できません")

        h = Harness(tmp_path, thread_factory=failing)
        error = h.start()
        assert error is not None
        assert h.state.phase is LivePhase.IDLE


class TestPreparingCancel:
    def test_stop_during_preparing_cancels_without_recording(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()

        h.wf.stop()
        run_pending()

        assert h.recorders == []
        assert h.state.phase is LivePhase.IDLE
        assert h.spy.recording_started == 0
        assert h.spy.errors == []
        assert any("取り消" in m for m in h.spy.logs)

    def test_cancel_after_session_start_releases_session(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()
        original = h._make_session

        def make_and_cancel(*args, **kwargs):
            session = original(*args, **kwargs)
            h.wf.stop()  # セッション生成直後に取り消し
            return session

        h.wf._deps = replace(h.wf._deps, session_factory=make_and_cancel)
        run_pending()

        assert h.recorders == []
        assert h.sessions[0].stop_timeouts
        assert h.state.phase is LivePhase.IDLE

    def test_shutdown_during_preparing_never_starts_recorder(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()

        h.wf.shutdown()
        run_pending()

        assert h.recorders == []
        assert h.saved_notes == []
