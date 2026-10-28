"""
ノート保存モジュール。

文字起こし結果を YAML frontmatter 付きの Markdown ファイルとして指定フォルダに保存する。
frontmatter は Obsidian の規約に準拠しているが、他の Markdown ベースのノートツールでも
そのまま利用できる。
"""

import contextlib
import json
import os
import re
import tempfile
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


def _participants_yaml(participants: tuple[str, ...]) -> str:
    """frontmatter の participants 行(空なら空文字)

    各要素は JSON 文字列(YAML のダブルクォート文字列としても有効)で書き、`:` `#` `- `
    や yes/null のような値・改行を含む名前でも型が変わったり構造が壊れたりしないようにする。
    """
    if not participants:
        return ""
    items = "".join(f"  - {json.dumps(name, ensure_ascii=False)}\n" for name in participants)
    return f"participants:\n{items}"


def save_live_note(
    save_folder: Path,
    body: str,
    audio_file: Path | None = None,
    participants: tuple[str, ...] = (),
) -> Path:
    """ライブ文字起こし結果を `save_folder/{開始時刻}_live.md` として保存する。

    Args:
        save_folder: 保存先フォルダの絶対パス
        body: ノート本文（frontmatter を除く Markdown）
        audio_file: 対応する録音 WAV。指定時はその stem（開始時刻）をノート名に使い、
            未指定なら現在時刻を使う。同名ファイルがあれば `_2`, `_3`, ... を付ける
        participants: 参加者名。空でなければ frontmatter に `participants:` のリストとして書く

    Returns:
        保存されたファイルのパス

    Raises:
        RuntimeError: 保存失敗時
    """
    save_folder = Path(save_folder)
    now = datetime.now()
    content = _live_note_content(now, body, tuple(participants), refined=False)
    try:
        save_folder.mkdir(parents=True, exist_ok=True)
        return _write_new_file(save_folder, _live_note_base(audio_file, now), content)
    except Exception as e:
        raise RuntimeError(f"保存エラー: {e}") from e


def _live_note_content(
    now: datetime, body: str, participants: tuple[str, ...], *, refined: bool
) -> str:
    """ライブノートの frontmatter + 本文。refined なら tags に refined を足す"""
    refined_tag = "  - refined\n" if refined else ""
    return f"""---
created: {now.isoformat()}
type: live-transcription
tags:
  - recording
  - live
{refined_tag}{_participants_yaml(participants)}---
{body}
"""


def _replace_atomically(path: Path, content: str) -> None:
    """同じフォルダの一時ファイルに書いて fsync し、os.replace で置き換える(元の権限を保つ)"""
    mode = path.stat().st_mode & 0o777
    # .md 以外の名前にして、ノートアプリが書き込み途中の一時ファイルを拾わないようにする
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def overwrite_live_note(
    path: Path,
    body: str,
    participants: tuple[str, ...] = (),
    refined: bool = False,
) -> Path:
    """保存済みのライブノートを原子的に上書きする(途中で失敗しても元のノートは残る)

    Args:
        path: save_live_note が返したノートのパス(存在しなければ上書きしない)
        body: 新しい本文（frontmatter を除く Markdown）
        participants: 参加者名(save_live_note と同じく frontmatter に書く)
        refined: 高精度の再文字起こし結果を含むなら True(tags に refined を足す)

    Returns:
        上書きしたファイルのパス

    Raises:
        RuntimeError: 保存失敗時(対象が無い場合を含む)
    """
    path = Path(path)
    content = _live_note_content(datetime.now(), body, tuple(participants), refined=refined)
    try:
        _replace_atomically(path, content)
        return path
    except Exception as e:
        raise RuntimeError(f"保存エラー: {e}") from e
