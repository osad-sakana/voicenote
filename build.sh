#!/bin/bash
set -e

echo "=== VoiceNote ビルド開始 ==="

# 古いビルド成果物を削除
rm -rf build dist

uv run pyinstaller main.py \
  --windowed \
  --name VoiceNote \
  --icon VoiceNote.icns \
  --collect-all customtkinter \
  --collect-all faster_whisper \
  --collect-all ctranslate2 \
  --collect-all tokenizers \
  --hidden-import sounddevice \
  --hidden-import scipy.io.wavfile \
  --hidden-import av \
  --noconfirm

# マイクアクセス権限のためにInfo.plistへ必須キーを追加
echo "Info.plist更新中..."
/usr/libexec/PlistBuddy -c \
  "Add :NSMicrophoneUsageDescription string '音声録音のためにマイクへのアクセスが必要です。'" \
  dist/VoiceNote.app/Contents/Info.plist

# アドホック署名（entitlements適用）
echo "署名中..."
codesign --deep --force --sign - \
  --entitlements entitlements.plist \
  dist/VoiceNote.app

echo ""
echo "=== ビルド完了 ==="
echo "→ dist/VoiceNote.app"
