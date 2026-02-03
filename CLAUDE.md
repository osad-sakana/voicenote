# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

voicenote is a voice recording and transcription tool that:
- Records audio using sounddevice
- Saves audio files to Desktop as WAV format
- Transcribes using faster-whisper (local) or OpenAI Whisper API (cloud)
- Saves transcriptions to Obsidian vault with frontmatter

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
```

Dependencies are managed in `pyproject.toml`. Use `uv sync` to install dependencies, or `uv run` which automatically syncs before running.

## Architecture

### Module Responsibilities

- **main.py**: Orchestrates the complete workflow (config → record → transcribe → save)
- **config.py**: Handles config.json persistence and interactive setup via rich prompts
- **recorder.py**: Real-time audio recording with SIGINT handling for Ctrl+C stop
- **transcriber.py**: Whisper model loading and transcription with progress indicators
- **obsidian.py**: Markdown file generation with YAML frontmatter

### Data Flow

**Recording Mode (default)**:
1. **Configuration Phase**: `main.py` → `config.py` (load/interactive setup) → `config.json`
2. **Recording Phase**: `main.py` → `recorder.py` (sounddevice stream with callback) → numpy array
3. **Save Phase**: `main.py` converts float32 → int16 → WAV file saved to `~/Desktop/YYYY-MM-DD_HHMMSS_recording.wav`
4. **Transcription Phase**: `main.py` → `transcriber.py` (faster-whisper) → text string
5. **Obsidian Save Phase**: `main.py` → `obsidian.py` → `{vault_path}/{save_folder}/YYYY-MM-DD_HHMMSS_raw.md`

**Record-Only Mode (`--record-only` argument)**:
1. **Configuration Phase**: `main.py` → `config.py` (load/interactive setup) → `config.json`
2. **Recording Phase**: `main.py` → `recorder.py` (sounddevice stream with callback) → numpy array
3. **Save Phase**: `main.py` converts float32 → int16 → WAV file saved to `~/Desktop/YYYY-MM-DD_HHMMSS_recording.wav` (transcription skipped)

**File Mode (`--file` argument)**:
1. **Configuration Phase**: `main.py` → `config.py` (load/interactive setup) → `config.json`
2. **Validation Phase**: `main.py` checks file existence and validates it's a file
3. **Transcription Phase**: `main.py` → `transcriber.py` (faster-whisper) → text string (supports WAV, MP3, M4A, etc.)
4. **Save Phase**: `main.py` → `obsidian.py` → `{vault_path}/{save_folder}/YYYY-MM-DD_HHMMSS_raw.md`

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

- Output filename pattern is in obsidian.py (`{timestamp}_raw.md`)
- Frontmatter structure matches Obsidian conventions (YAML between --- delimiters)
- The `_raw` suffix indicates untouched transcription (vs potential summarized versions)
