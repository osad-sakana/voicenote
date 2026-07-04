# voicenote

ローカルで音声を録音し、faster-whisperで文字起こしを行い、Markdownノートとして保存するPythonツールです (Obsidianなどのノートツールと互換)。

## 必要要件

- Python 3.10以上
- [uv](https://github.com/astral-sh/uv)（Pythonパッケージマネージャー）

## インストール

1. uvをインストールしていない場合は、まずインストールします:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 依存関係をインストールします:

```bash
uv sync
```

### グローバルインストール（オプション）

`uv tool`を使うと、システム全体で`voicenote`コマンドとして使用できます:

```bash
uv tool install . --python 3.12
```

> Python 3.14以降では依存関係がサポートされていないため、3.12または3.13を指定してください。

インストール後は、どこからでも実行可能です:

```bash
voicenote           # 録音開始
voicenote --config  # 設定変更
```

アンインストールする場合:

```bash
uv tool uninstall voicenote
```

## 使い方

### 初回実行

初回実行時は設定を対話的に入力します:

```bash
uv run main.py
```

> `uv run` は実行前に自動的に依存関係を同期します。

以下の項目を入力します:
- ノート保存フォルダの絶対パス（Obsidian Vault内などお好みの場所）
- 文字起こしモード（local: faster-whisper / openai: Whisper API）
- 使用するWhisperモデル（tiny, base, small, medium, large-v3）
- テキスト整形モード（none / rule / llm）

### 2回目以降

設定が保存されているので、すぐに録音が開始されます:

```bash
uv run main.py
```

録音中に `Ctrl+C` を押すと録音が終了し、文字起こしが開始されます。

### 設定の再入力

設定を変更したい場合は `--config` オプションを使用します:

```bash
uv run main.py --config
```

## 出力形式

文字起こし結果は以下の形式でMarkdownノートとして保存されます:

```markdown
---
created: 2026-01-16T12:34:56.789012
type: transcription
tags:
  - recording
  - raw
---

# 録音文字起こし

[文字起こし結果]
```

ファイル名: `YYYY-MM-DD_HHMMSS_raw.md`

## プロジェクト構造

```
voicenote/
├── main.py              # GUIエントリーポイント (CustomTkinter)
├── main_cli.py          # CLIエントリーポイント (Rich)
├── pipeline.py          # GUI/CLI共通の業務ロジック
├── logging_setup.py     # ロギング初期化
├── config.py            # 設定管理モジュール
├── recorder.py          # 録音機能モジュール
├── transcriber.py       # 文字起こし機能モジュール
├── formatter.py         # テキスト整形モジュール
├── note_writer.py       # Markdownノート保存モジュール
├── gui/                 # GUI 関連 (App, SettingsDialog, etc.)
├── tests/               # ユニットテスト
├── pyproject.toml       # プロジェクト設定・依存関係
└── README.md
```

## 機能

- ✅ ローカルで完結（外部APIを使用しない）
- ✅ 対話的な設定管理
- ✅ リアルタイム録音
- ✅ faster-whisperによる高精度な文字起こし
- ✅ Markdownノートとしての自動保存（YAMLフロントマター付き、Obsidian互換）
- ✅ Richライブラリによる美しいUI
- ✅ モジュール化された綺麗なコード構造

## ライセンス

このプロジェクトは[MITライセンス](LICENSE)のもとで公開されています。
