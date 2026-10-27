"""
ノート保存モジュール。

文字起こし結果を YAML frontmatter 付きの Markdown ファイルとして指定フォルダに保存する。
frontmatter は Obsidian の規約に準拠しているが、他の Markdown ベースのノートツールでも
そのまま利用できる。
"""

import re
from datetime import datetime
from pathlib import Path

# 録音 WAV の stem 末尾(`_recording` / `_recording_2` 等)。除くと開始時刻が残る
_RECORDING_SUFFIX_PATTERN = re.compile(r"_recording(?:_\d+)?$")


def save_transcript(save_folder: Path, transcription: str, format_mode: str = "none") -> Path:
    """文字起こし結果を `save_folder/YYYY-MM-DD_HHMMSS_raw.md` として保存する。

    Args:
        save_folder: 保存先フォルダの絶対パス
        transcription: 文字起こしされたテキスト
        format_mode: 使用した整形モード（"none"/"rule"/"llm"）

    Returns:
        保存されたファイルのパス

    Raises:
        RuntimeError: 保存失敗時
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = save_folder / f"{timestamp}_raw.md"

    now = datetime.now().isoformat()
    content = f"""---
created: {now}
type: transcription
format_mode: {format_mode}
tags:
  - recording
  - raw
---
{transcription}
"""

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
    except Exception as e:
        raise RuntimeError(f"保存エラー: {e}") from e


def _live_note_base(audio_file: Path | None, now: datetime) -> str:
    """ライブノート名の基部。WAV(`{開始時刻}_recording[_N].wav`)があればその開始時刻を使う"""
    if audio_file is None:
        return now.strftime("%Y-%m-%d_%H%M%S")
    return _RECORDING_SUFFIX_PATTERN.sub("", Path(audio_file).stem) or Path(audio_file).stem


def _write_new_file(save_folder: Path, base: str, content: str) -> Path:
    """`{base}_live.md` を新規作成する。既にあれば `_2`, `_3`, ... を付けて上書きを避ける"""
    n = 1
    while True:
        suffix = "" if n == 1 else f"_{n}"
        filepath = save_folder / f"{base}_live{suffix}.md"
        try:
            # "x" で排他的に新規作成し、存在確認と書き込みの間の競合でも上書きしない
            with open(filepath, "x", encoding="utf-8") as f:
                f.write(content)
            return filepath
        except FileExistsError:
            n += 1


def save_live_note(save_folder: Path, body: str, audio_file: Path | None = None) -> Path:
    """ライブ文字起こし結果を `save_folder/{開始時刻}_live.md` として保存する。

    Args:
        save_folder: 保存先フォルダの絶対パス
        body: ノート本文（frontmatter を除く Markdown）
        audio_file: 対応する録音 WAV。指定時はその stem（開始時刻）をノート名に使い、
            未指定なら現在時刻を使う。同名ファイルがあれば `_2`, `_3`, ... を付ける

    Returns:
        保存されたファイルのパス

    Raises:
        RuntimeError: 保存失敗時
    """
    save_folder = Path(save_folder)
    now = datetime.now()
    content = f"""---
created: {now.isoformat()}
type: live-transcription
tags:
  - recording
  - live
---
{body}
"""
    try:
        save_folder.mkdir(parents=True, exist_ok=True)
        return _write_new_file(save_folder, _live_note_base(audio_file, now), content)
    except Exception as e:
        raise RuntimeError(f"保存エラー: {e}") from e
