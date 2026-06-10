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
```

Dependencies are managed in `pyproject.toml`. Use `uv sync` to install dependencies, or `uv run` which automatically syncs before running.

## Architecture

### Module Responsibilities

- **main.py**: GUI entry point (CustomTkinter)
- **main_cli.py**: CLI entry point (Rich)
- **pipeline.py**: Shared business logic — `load_or_configure`, `save_wav`, `transcribe_and_save`
- **logging_setup.py**: Logging initialization (shared by GUI and CLI)
- **config.py**: Handles config.json persistence and interactive setup via rich prompts
- **recorder.py**: Real-time audio recording with SIGINT handling for Ctrl+C stop
- **transcriber.py**: Whisper model loading and transcription with progress indicators
- **formatter.py**: Rule-based and LLM-based transcription text formatting
- **note_writer.py**: Markdown note file generation with YAML frontmatter (Obsidian-compatible)
- **gui/**: GUI components — App (main window), SettingsDialog, ThreadSafeUIQueue, constants

### Data Flow

Both GUI (`main.py`) and CLI (`main_cli.py`) delegate the core workflow to `pipeline.py`.

**Recording Mode (default)**:
1. **Configuration Phase**: entry → `pipeline.load_or_configure` → `config.json`
2. **Recording Phase**: entry → `recorder.py` (sounddevice stream with callback) → numpy array
3. **WAV Save Phase**: entry → `pipeline.save_wav` → WAV file at `~/Desktop/YYYY-MM-DD_HHMMSS_recording.wav` (CLI) or selected folder (GUI)
4. **Transcribe + Note Save Phase**: entry → `pipeline.transcribe_and_save` → `transcriber.py` → `formatter.py` (optional) → `note_writer.save_transcript` → `{save_folder}/YYYY-MM-DD_HHMMSS_raw.md`

**Record-Only Mode (CLI `--record-only` / GUI "録音だけする")**:
1. Configuration → recording → WAV save (steps 1–3 above)
2. Transcription is skipped

**File Mode (CLI `--file` / GUI "文字起こしだけする")**:
1. **Configuration Phase**: entry → `pipeline.load_or_configure` → `config.json`
2. **Validation Phase**: entry checks file existence and validates it's a file
3. **Transcribe + Note Save Phase**: entry → `pipeline.transcribe_and_save` → ... → `{save_folder}/YYYY-MM-DD_HHMMSS_raw.md` (supports WAV, MP3, M4A, etc.)

### Important Implementation Details

- **Audio Format**: Recording is float32 mono at 16kHz (SAMPLE_RATE constant in recorder.py)
- **Signal Handling**: recorder.py uses global state (_is_recording, _recording_data) with SIGINT handler for graceful Ctrl+C shutdown
- **Audio File Storage**: Recorded WAV files are always saved to Desktop with format `YYYY-MM-DD_HHMMSS_recording.wav`
- **Transcription Modes**: `local` (faster-whisper, CPU, int8) or `openai` (Whisper API). Mode selected via `--config` when `OPENAI_API_KEY` is set.
- **Whisper Configuration**: Local mode uses CPU device with int8 compute_type, auto language detection, beam_size=5. OpenAI mode uses whisper-1 model with 25MB file size limit.
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
