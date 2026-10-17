"""main.py のエントリーポイント構成に関する回帰テスト (Issue #17)。

PyInstaller 版では multiprocessing の子プロセス (resource_tracker など) が
アプリ自身を再実行する。`freeze_support()` が `main()` より前に呼ばれていないと、
ウィンドウを閉じた後に GUI が再び起動してしまう。
"""

import ast
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent.parent / "main.py"


def _call_names_in_main_guard() -> list[str]:
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "__name__"
    )
    names = []
    for stmt in guard.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            names.append(func.attr if isinstance(func, ast.Attribute) else func.id)
    return names


def test_freeze_support_is_called_before_main():
    names = _call_names_in_main_guard()
    assert "freeze_support" in names
    assert "main" in names
    assert names.index("freeze_support") < names.index("main")
