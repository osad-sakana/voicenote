#!/usr/bin/env python3
"""VoiceNote GUI エントリーポイント (CustomTkinter)"""

from dotenv import load_dotenv

from gui.app import App
from logging_setup import setup_logging


def main():
    load_dotenv()
    log_file = setup_logging()
    app = App(log_file=log_file)
    app.mainloop()


if __name__ == "__main__":
    main()
