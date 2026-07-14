"""入力デバイスのラベル生成・解析を行う純関数 (Tkinter 非依存)。"""

from .constants import strip_emoji


def format_device_label(device_id: int, name: str) -> str:
    """'[3] MacBook Air マイク' 形式のラベルを生成する。"""
    return f"[{device_id}] {strip_emoji(name)}"


def build_device_labels(devices: list[dict]) -> list[str]:
    """`recorder.list_devices()` の戻り値からラベル一覧を生成する。"""
    names = [format_device_label(d["id"], d["name"]) for d in devices]
    return names or ["デバイスなし"]


def parse_device_id(label: str) -> int | None:
    """'[3] ...' からデバイスID (3) を逆引きする。解決できなければ None。"""
    if label.startswith("["):
        try:
            return int(label.split("]")[0][1:])
        except ValueError:
            pass
    return None
