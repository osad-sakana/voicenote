#!/bin/bash
set -e

echo "=== VoiceNote ビルド開始 ==="

# 古いビルド成果物を削除
rm -rf build dist

uv run pyinstaller main.py \
  --windowed \
  --name VoiceNote \
  --collect-all customtkinter \
  --collect-all faster_whisper \
  --collect-all ctranslate2 \
  --collect-all tokenizers \
  --hidden-import sounddevice \
  --hidden-import scipy.io.wavfile \
  --hidden-import av \
  --noconfirm

echo ""
echo "=== ビルド完了 ==="
echo "→ dist/VoiceNote.app"
