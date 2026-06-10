"""
ロギング初期化モジュール。GUI / CLI のエントリーポイントから呼び出す。

`voicenote` ロガーは DEBUG 以上、ルートロガー (外部ライブラリ) は WARNING 以上を
`logs/YYYY-MM-DD_HHMMSS.log` に出力する。
"""

import logging
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"


def setup_logging() -> Path:
    """ロギングハンドラを登録し、出力先のログファイルパスを返す。"""
    _LOG_DIR.mkdir(exist_ok=True)
    log_file = _LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.getLogger().setLevel(logging.WARNING)
    logger = logging.getLogger("voicenote")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return log_file
