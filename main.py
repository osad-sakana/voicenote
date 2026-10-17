#!/usr/bin/env python3
"""VoiceNote GUI エントリーポイント (CustomTkinter)"""

import multiprocessing
import socket
import sys

_LOCK_PORT = 47391  # 衝突しにくい固定ポートでシングルインスタンスを保証


def _acquire_instance_lock() -> socket.socket | None:
    """ソケットロックを取得する。既に起動中なら None を返す。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", _LOCK_PORT))
        return sock
    except OSError:
        sock.close()
        return None


def main():
    # 重い import は freeze_support() の後で行う。multiprocessing の子プロセスも
    # main.py を先頭から実行するため、モジュール先頭に置くと子が GUI 一式を読み込む。
    from dotenv import load_dotenv

    from gui.app import App
    from logging_setup import setup_logging

    lock = _acquire_instance_lock()
    if lock is None:
        # 既に起動中 — 何もせず終了
        sys.exit(0)

    try:
        load_dotenv()
        log_file = setup_logging()
        app = App(log_file=log_file)
        app.mainloop()
    finally:
        lock.close()


if __name__ == "__main__":
    # PyInstaller 版では multiprocessing の子プロセス (tqdm のロックが起動する
    # resource_tracker など) が `sys.executable -c ...` でアプリ自身を再実行する。
    # freeze_support() が無いと子が GUI を丸ごと起動してしまうため、最初に呼ぶ。
    multiprocessing.freeze_support()
    main()
