"""transcriber モジュールのユニットテスト。

`transcribe()` のモード分岐（local/openai ディスパッチ）のみを検証する。
実際の文字起こし処理（faster-whisper・OpenAI API 呼び出し）は monkeypatch で置き換える。
"""

from pathlib import Path

from config import VoiceNoteConfig
from transcriber import transcribe


class TestTranscribe:
    def test_local_mode_calls_transcribe_audio(self, monkeypatch):
        calls = {}

        def fake_transcribe_audio(audio_path, model_name, progress_callback=None, vad_filter=True):
            calls["args"] = (audio_path, model_name, progress_callback, vad_filter)
            return "local result"

        def fake_transcribe_audio_openai(audio_path, progress_callback=None):
            raise AssertionError("openai 版は呼ばれてはいけない")

        monkeypatch.setattr("transcriber.transcribe_audio", fake_transcribe_audio)
        monkeypatch.setattr("transcriber.transcribe_audio_openai", fake_transcribe_audio_openai)

        config = VoiceNoteConfig(
            transcription_mode="local", whisper_model="small", vad_filter=False
        )
        audio_path = Path("/tmp/audio.wav")
        result = transcribe(audio_path, config)

        assert result == "local result"
        assert calls["args"] == (audio_path, "small", None, False)

    def test_openai_mode_calls_transcribe_audio_openai(self, monkeypatch):
        calls = {}

        def fake_transcribe_audio(audio_path, model_name, progress_callback=None, vad_filter=True):
            raise AssertionError("local 版は呼ばれてはいけない")

        def fake_transcribe_audio_openai(audio_path, progress_callback=None):
            calls["args"] = (audio_path, progress_callback)
            return "openai result"

        monkeypatch.setattr("transcriber.transcribe_audio", fake_transcribe_audio)
        monkeypatch.setattr("transcriber.transcribe_audio_openai", fake_transcribe_audio_openai)

        config = VoiceNoteConfig(transcription_mode="openai")
        audio_path = Path("/tmp/audio.wav")
        result = transcribe(audio_path, config)

        assert result == "openai result"
        assert calls["args"] == (audio_path, None)

    def test_progress_callback_is_passed_through(self, monkeypatch):
        received = {}

        def fake_transcribe_audio(audio_path, model_name, progress_callback=None, vad_filter=True):
            received["callback"] = progress_callback
            return "ok"

        monkeypatch.setattr("transcriber.transcribe_audio", fake_transcribe_audio)

        def on_progress(msg: str):
            pass

        config = VoiceNoteConfig(transcription_mode="local")
        transcribe(Path("/tmp/audio.wav"), config, progress_callback=on_progress)

        assert received["callback"] is on_progress
