# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

voicenote is a voice recording and transcription tool that:
- Records audio using sounddevice
- Saves audio files to Desktop as WAV format
- Transcribes using faster-whisper (local) or OpenAI Whisper API (cloud)
- Saves transcriptions as Markdown notes with YAML frontmatter (Obsidian-compatible, but any Markdown note tool works)

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for cloud transcription (optional). When set, enables OpenAI mode selection during `--config`.

## Running the Application

```bash
# First run or reconfigure settings
uv run main.py --config

# Normal run (starts recording immediately)
uv run main.py

# Record only mode (saves to Desktop without transcription)
uv run main.py --record-only

# Transcribe existing audio file (skips recording)
uv run main.py --file path/to/audio.wav
uv run main.py --file path/to/audio.mp3

# List available audio devices
uv run main.py --list-devices

# Record from specific device (name or ID)
uv run main.py --device "BlackHole 2ch"
uv run main.py --device 2

# Live mode: record + real-time local transcription + rolling local-LLM summary (CLI; in the GUI choose the "ライブ" mode)
uv run main_cli.py --live
uv run main_cli.py --live --device "BlackHole 2ch"
```

Live mode setup (all local, nothing leaves the machine):

```bash
uv sync --extra live-mlx      # optional: MLX engine on Apple Silicon (falls back to faster-whisper)
ollama serve                  # local LLM server for summaries (optional)
ollama pull qwen2.5:7b        # default summary model (live_llm_model)
```

`--live` cannot be combined with `--file` or `--record-only`. Live settings (`live_*` keys in config.json) are asked in an optional section of `--config`; old config files without them load with defaults (`faster-whisper` + `small`).

Recommended on Apple Silicon: `uv sync --extra live-mlx`, then in `--config` choose engine `mlx` — the model prompt then defaults to `large-v3-turbo` (`config.default_live_model_for`; the global default stays `small`). Measured on M1 16GB: mlx large-v3-turbo is clearly more accurate than small at p50 ~1.5s per segment, while faster-whisper (CPU int8) large-v3-turbo is not real-time (p50 5.4–9.7s, RTF 3–4), so when mlx falls back to faster-whisper, `turbo` / `large-v3-turbo` are replaced with `small` (`live_cli.CPU_UNSUITABLE_MODELS`). Low microphone input level noticeably hurts accuracy — raise the input volume.

Dependencies are managed in `pyproject.toml`. Use `uv sync` to install dependencies, or `uv run` which automatically syncs before running.

## Architecture

### Module Responsibilities

- **main.py**: GUI entry point (CustomTkinter)
- **main_cli.py**: CLI entry point (Rich)
- **pipeline.py**: Shared business logic — `load_or_configure`, `transcribe_and_save`
- **logging_setup.py**: Logging initialization (shared by GUI and CLI)
- **config.py**: Handles config.json persistence and interactive setup via rich prompts
- **recorder.py**: Real-time audio recording with SIGINT handling for Ctrl+C stop. `ThreadedRecorder` streams chunks straight to disk via `wav_writer.py` instead of buffering them in memory
- **wav_writer.py**: `StreamingWavWriter` — opens a WAV file when recording starts and writes chunks to it incrementally on a background thread, so the file stays valid (playable) even if the recording is interrupted before a clean stop
- **transcriber.py**: Whisper model loading and transcription with progress indicators
- **formatter.py**: Rule-based and LLM-based transcription text formatting
- **note_writer.py**: Markdown note file generation with YAML frontmatter (Obsidian-compatible)
- **live_cli.py**: `--live` entry (`run_live`) — engine creation with faster-whisper fallback, LLM probe, Rich Live display, recording, live note save. Lazily imported by main_cli.py only when `--live` is given
- **live/**: Real-time transcription parts
  - `session.py` (`LiveSession`: segmenter / asr / summary daemon threads), `session_models.py` (`LiveSettings`, `LiveCallbacks`, `LiveSegment`, `LiveResult`)
  - `framer.py`, `vad.py`, `segmenter.py` (chunk → frame → speech segment), `chunk_queue.py` (drop-newest, for the audio callback), `segment_queue.py` (drop-oldest, for pending segments)
  - `engines.py` (`create_engine`: faster-whisper / mlx), `hallucination.py` (filters Whisper boilerplate)
  - `local_llm.py` (loopback-only OpenAI-compatible client, `probe`), `summarizer.py` (`RollingSummarizer`), `language_guard.py` (`is_plausible_japanese`: rejects summaries with Chinese mixed in — Han without any kana, or lines whose Han chars are ≥ `SIMPLIFIED_LINE_RATIO` simplified-only glyphs; the glyph set excludes anything in JIS X 0208)
  - `note.py` (`render_live_note`, pure), `view.py` (`LiveView`, Rich rendering state)
  - `context.py` (pure): `LiveContext` (participants / memo, normalized and capped on construction: `MAX_PARTICIPANTS=30`, `MAX_PARTICIPANT_CHARS=40`, `MAX_MEMO_CHARS=2000`, `memo_truncated`), `LiveContext.from_inputs`, `parse_participants`. Its `repr` hides the content (counts / lengths only); never log the text
  - `bootstrap.py` (shared by CLI and GUI): engine / VAD / local-LLM backend creation and fallback decisions (`build_engine`, `build_vad`, `check_backend`); UI-agnostic, notifies via callbacks, fatal errors raise `LiveSetupError`, all heavy deps injected
  - `model_fetch.py`: Whisper model download from the HuggingFace Hub with byte progress (used by the GUI before recording starts); `huggingface_hub` / `faster_whisper` / `tqdm` are imported lazily inside functions
  - `level_meter.py`: `LevelMeter` (latest RMS dBFS + history, fed from the audio callback via `tee`) and `assess_input_level` (90th percentile < -40 dB → "input too low")
- **gui/**: GUI components — App (main window), SettingsDialog, ThreadSafeUIQueue, constants
  - `mode_logic.py` (pure): window size per mode (`window_spec`: live 760x760 resizable, min 600x640, others fixed 520x380/560), exec-button text for non-live modes, `other_mode_rejection` (blocks normal recording / transcribe-only while live is active)
  - `live_state.py` (pure): `LiveViewState` / `LivePhase` and reducers; `new_display_lines` computes text-box deltas
  - `live_workflow.py` (+ `live_workflow_*.py` helpers): `LiveWorkflow` — Tk-independent lifecycle and threads of GUI live mode (public API: `LiveWorkflow`, `LiveWorkflowCallbacks`, `validate_live_start`, `LiveDeps`, `MeterReading`, `BANNER_*`)
  - `live_view_logic.py` (pure): `button_spec(phase)`, status / banner / meter / delay text, `plan_segment_render` (redraws when a new run resets the segment count)
  - `live_context_logic.py` (Tk-independent): `build_context`, `limit_notice` (over-limit hint text, never contains the input), `Debouncer` (injected `after`/`after_cancel`), `ContextSync` (debounced push to `LiveWorkflow.update_context`, `flush(force=True)` on start / before shutdown, `clear`, skips unchanged values)
  - `live_context_form.py` (widget construction only): participants entry + memo textbox + "クリア" + limit hint; reports changes via Entry `validate="key"` / `<FocusOut>` and Textbox `<<Modified>>`
  - `live_layout.py` (widget construction only; top → bottom: banner, participants/memo form, summary (main, larger font, expands), transcript (3 lines, collapsible via the toggle in the summary header row); when the window is short the transcript shrinks first), `live_panel.py` (`LivePanel`: thin widget-update layer), `live_controller.py` (`LiveController`: wires `LiveWorkflow` to App — every callback goes through `ui_queue.submit`, `poll_meter` every 250 ms, completion / "高精度で文字起こし")
  - `live_settings_form.py`: the "ライブ" tab of SettingsDialog

### Data Flow

Both GUI (`main.py`) and CLI (`main_cli.py`) delegate the core workflow to `pipeline.py`.

**Recording Mode (default)**:
1. **Configuration Phase**: entry → `pipeline.load_or_configure` → `config.json`
2. **Recording Phase**: entry → `recorder.ThreadedRecorder.start()` opens a `wav_writer.StreamingWavWriter` at `~/Desktop/YYYY-MM-DD_HHMMSS_recording.wav` (CLI) or the selected folder (GUI), then streams each sounddevice callback chunk to it incrementally — the WAV file is valid on disk throughout the recording, not just after it ends
3. **Finalize Phase**: entry → `recorder.ThreadedRecorder.finalize_recording()` closes the WAV file and returns its path. This happens *before* `stop_with_timeout()` (which stops the PortAudio stream) so a hang in stream shutdown can no longer lose recorded audio
4. **Transcribe + Note Save Phase**: entry → `pipeline.transcribe_and_save` → `transcriber.py` → `formatter.py` (optional) → `note_writer.save_transcript` → `{save_folder}/YYYY-MM-DD_HHMMSS_raw.md`

**Record-Only Mode (CLI `--record-only` / GUI "録音だけする")**:
1. Configuration → recording → finalize (steps 1–3 above)
2. Transcription is skipped

**File Mode (CLI `--file` / GUI "文字起こしだけする")**:
1. **Configuration Phase**: entry → `pipeline.load_or_configure` → `config.json`
2. **Validation Phase**: entry checks file existence and validates it's a file
3. **Transcribe + Note Save Phase**: entry → `pipeline.transcribe_and_save` → ... → `{save_folder}/YYYY-MM-DD_HHMMSS_raw.md` (supports WAV, MP3, M4A, etc.)

**Live Mode (CLI `--live`)**:
1. **Setup Phase**: `LiveSettings.from_config` → `live.engines.create_engine` (mlx unavailable → faster-whisper + warning) → `local_llm.probe` (server down / model missing → continue without summary)
2. **Recording Phase**: `LiveSession.start()` then `record_audio(chunk_sink=session.feed)`. `feed` only does a non-blocking enqueue; Framer → VAD → Segmenter → engine → hallucination filter run on worker threads, and the summary is held while transcription is behind (`backlog_sec`)
3. **Stop Phase**: `record_audio` returns (WAV finalized, stream stopped) → `session.stop(timeout)` (flush, drain, one final summary) → `note_writer.save_live_note` → `{save_folder}/YYYY-MM-DD_HHMMSS_live.md` (type: live-transcription, tags: [recording, live])
4. The user is told to run `uv run main_cli.py --file <wav>` for an accurate full transcript

**Live Mode (GUI, mode "ライブ")**:
1. **Prepare Phase (PREPARING)**: `validate_live_start` (save folder, device, live settings, mic not busy) → `LiveWorkflow.start()` → worker thread: `bootstrap.build_engine` (mlx → faster-whisper fallback banner) → `model_fetch` download if not cached (progress banner) → `check_backend` (Ollama missing → banner, continue without summary) → `build_vad` → `LiveSession.start()` → `ThreadedRecorder` with `tee(LevelMeter.feed, session.feed)`. The stop button cancels while preparing
2. **Recording Phase (RECORDING)**: segments / summary / warnings arrive as `LiveViewState` via `on_state` (wrapped in `ui_queue.submit`); App polls `poll_meter()` every 250 ms (input level, `backlog_sec`, dropped chunks; low input → banner)
3. **Stop Phase (FINALIZING)**: same invariant as CLI — `finalize_recording()` (WAV on disk) → `stop_with_timeout()` → only then the live drain, on separate threads
4. **Drain Phase (DRAINING)**: `session.stop(timeout)` transcribes the rest and makes the final summary. The button becomes "中断して保存": aborting abandons the wait and saves a partial note (`complete=False`) from the segments/summary collected by the GUI
5. **Save Phase (SAVING → IDLE)**: `note_writer.save_live_note` → `{save_folder}/YYYY-MM-DD_HHMMSS_live.md`; App shows WAV/note paths and "高精度で文字起こし" (switches to "文字起こしだけする" with the WAV path filled in). Closing the window calls `live.shutdown()` (WAV finalized + partial note saved synchronously). The WAV is the source of truth
6. **Participants / memo**: form input (always editable, any phase) → `ContextSync` (600 ms debounce; flushed when "ライブ開始" is pressed and before `shutdown`) → `LiveContext.from_inputs` → `LiveWorkflow.update_context` → summary prompt (from the next summary on; a change alone does not trigger a summary) and the live note (value at save time, including edits made while draining). The workflow never clears the context; the form keeps its content after a run until "クリア" (`update_context(None)`). The form exists only in live mode
7. **Mutual exclusion**: `can_start` = App not busy and `RecordingWorkflow` not recording; normal recording / transcribe-only check `live.is_active` first. Mode selector, device menu and settings are locked while the live phase is not IDLE

### Important Implementation Details

- **Audio Format**: Recording is captured as float32 mono at 16kHz (SAMPLE_RATE constant in recorder.py) and converted to int16 by `wav_writer.float32_to_int16` before being written
- **Signal Handling**: recorder.py's `record_audio()` installs a SIGINT handler for graceful Ctrl+C shutdown; recording state lives on the `ThreadedRecorder` instance, not module globals
- **Audio File Storage**: Recorded WAV files are always saved to Desktop with format `YYYY-MM-DD_HHMMSS_recording.wav` (or `_2`, `_3`, ... if a file with that timestamp already exists). The timestamp reflects when recording *started*, not when it was saved
- **Transcription Modes**: `local` (faster-whisper, CPU, int8) or `openai` (Whisper API). Mode selected via `--config` when `OPENAI_API_KEY` is set.
- **Whisper Configuration**: Local mode uses CPU device with int8 compute_type, auto language detection, beam_size=5. OpenAI mode uses whisper-1 model with 25MB file size limit.
- **Live Mode — WAV is the source of truth**: live text is best-effort (chunks/segments may be dropped under load, engine errors are skipped with a warning). The WAV on Desktop is always complete
- **Live Mode — stop-order invariant**: "WAV finalized → stream stopped → live drain". Never make WAV finalization wait for the live session; `session.stop()` is called only after `record_audio` returns, and is bounded by a timeout even if the engine or LLM hangs
- **Live Mode — VAD**: default is `silero` (`DEFAULT_LIVE_VAD`, selectable in `--config`). The energy VAD (-40 dB threshold) misses most speech on quiet input (real 456 s recording at RMS -47 dB: energy 25 segments / 61 s vs silero 56 / 199 s). If Silero fails to load, `create_vad` falls back to `EnergyVAD` and `live_cli` prints a yellow warning (detected via `vad_kind`)
- **Live Mode — local only**: the summary LLM URL must be loopback (`is_loopback_url`); non-loopback URLs are rejected
- **Transcription Output**: Markdown files named `YYYY-MM-DD_HHMMSS_raw.md` with YAML frontmatter containing created timestamp, type=transcription, tags=[recording, raw]

## Code Modification Guidelines

### When adding features:

- **UI feedback**: Use rich Console for all user-facing messages (already instantiated as `console` in each module)
- **Error handling**: Exit with `sys.exit(1)` on fatal errors after printing red error message
- **File paths**: Always use pathlib.Path, not string concatenation
- **Config changes**: Update both `configure_interactive()` prompts and the config dict structure

### When modifying transcription:

- Language detection is automatic (no hardcoded language parameter)
- Changing model affects accuracy/speed tradeoff (tiny → large-v3)
- Audio must be written to disk before transcription
- OpenAI mode has 25MB file size limit; local mode has no limit

### When changing output format:

- Output filename pattern is in note_writer.py (`{timestamp}_raw.md`)
- Frontmatter structure matches Obsidian conventions (YAML between --- delimiters) — other Markdown note tools (Logseq, Bear, etc.) will also parse it
- The `_raw` suffix indicates untouched transcription (vs potential summarized versions)
