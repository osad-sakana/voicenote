"""`import gui.app` が重い依存(音声認識・HF・OpenAI)を読み込まないことのテスト

GUI の起動を速く保つため、これらはライブ開始・文字起こし時に遅延 import する。
別プロセスで import して sys.modules を調べる(このプロセスは他のテストで汚れているため)。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_HEAVY = ("mlx_whisper", "numba", "huggingface_hub", "faster_whisper", "openai")

_SCRIPT = f"""
import json, sys
import gui.app  # noqa: F401
print(json.dumps([name for name in {list(_HEAVY)!r} if name in sys.modules]))
"""


@pytest.fixture(scope="module")
def loaded_heavy_modules() -> list[str]:
    try:
        import customtkinter  # noqa: F401
        import tkinterdnd2  # noqa: F401
    except Exception as e:  # pragma: no cover - GUI 依存が無い環境
        pytest.skip(f"GUI 依存を読み込めません: {e}")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", _HEAVY)
def test_gui_app_import_is_light(loaded_heavy_modules, module):
    assert module not in loaded_heavy_modules
