# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

voicenote is a local voice recording and transcription tool that:
- Records audio using sounddevice
- Transcribes using faster-whisper (completely local, no external APIs)
- Saves transcriptions to Obsidian vault with frontmatter

## Running the Application

```bash
# First run or reconfigure settings
uv run main.py --config

# Normal run (starts recording immediately)
uv run main.py

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
3. **Conversion Phase**: `main.py` converts float32 → int16 → WAV file (`temp_recording.wav`)
4. **Transcription Phase**: `main.py` → `transcriber.py` (faster-whisper) → text string
5. **Save Phase**: `main.py` → `obsidian.py` → `{vault_path}/{save_folder}/YYYY-MM-DD_HHMMSS_raw.md`

**File Mode (`--file` argument)**:
1. **Configuration Phase**: `main.py` → `config.py` (load/interactive setup) → `config.json`
2. **Validation Phase**: `main.py` checks file existence and validates it's a file
3. **Transcription Phase**: `main.py` → `transcriber.py` (faster-whisper) → text string (supports WAV, MP3, M4A, etc.)
4. **Save Phase**: `main.py` → `obsidian.py` → `{vault_path}/{save_folder}/YYYY-MM-DD_HHMMSS_raw.md`

### Important Implementation Details

- **Audio Format**: Recording is float32 mono at 16kHz (SAMPLE_RATE constant in recorder.py)
- **Signal Handling**: recorder.py uses global state (_is_recording, _recording_data) with SIGINT handler for graceful Ctrl+C shutdown
- **Temporary Files**: WAV file is created temporarily for Whisper input, then deleted after transcription
- **Whisper Configuration**: Always uses CPU device with int8 compute_type, Japanese language, beam_size=5
- **Output Format**: Files named `YYYY-MM-DD_HHMMSS_raw.md` with YAML frontmatter containing created timestamp, type=transcription, tags=[recording, raw]

## Code Modification Guidelines

### When adding features:

- **UI feedback**: Use rich Console for all user-facing messages (already instantiated as `console` in each module)
- **Error handling**: Exit with `sys.exit(1)` on fatal errors after printing red error message
- **File paths**: Always use pathlib.Path, not string concatenation
- **Config changes**: Update both `configure_interactive()` prompts and the config dict structure

### When modifying transcription:

- The `language="ja"` parameter in transcriber.py assumes Japanese audio
- Changing model affects accuracy/speed tradeoff (tiny → large-v3)
- Audio must be written to disk as WAV before transcription (faster-whisper limitation)

### When changing output format:

- Output filename pattern is in obsidian.py (`{timestamp}_raw.md`)
- Frontmatter structure matches Obsidian conventions (YAML between --- delimiters)
- The `_raw` suffix indicates untouched transcription (vs potential summarized versions)
