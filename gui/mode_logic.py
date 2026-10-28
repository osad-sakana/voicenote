"""モード選択で決まるウィンドウサイズ・実行ボタン・排他の判定(Tk 非依存の純関数)"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import MODE_LIVE, MODE_RECORD_ONLY, MODE_RECORD_TRANSCRIBE, MODE_TRANSCRIBE_ONLY

NORMAL_WIDTH = 520
COMPACT_HEIGHT = 380  # 詳細ログを隠している時
EXPANDED_HEIGHT = 560  # 詳細ログを表示している時
LIVE_WIDTH = 760
# 既定の高さ。13インチ+ドックの画面(縦 約800px 使える)に収める。実App計測(幅760)で、
# 案内・警告が無ければ要約・文字起こしとも全て表示できる
LIVE_HEIGHT = 800
LIVE_MIN_WIDTH = 600
# 最小の高さ。実App計測(幅760)で、案内バナー1件+入力レベル警告+上限案内(3項目すべて)+
# 完了表示を出し、文字起こしを自動で折りたたんだ時の要求高さが 782px(要約欄は最小の3行分)。
# この高さなら要約欄の最小高さ・実行ボタン・参加者・メモ欄・メーター・完了表示が見切れない。
# 案内バナーが3件同時に出ると 799px 必要で、既定の高さ(800)なら収まる
LIVE_MIN_HEIGHT = 790


@dataclass(frozen=True)
class WindowSpec:
    """ウィンドウの大きさとリサイズ可否"""

    width: int
    height: int
    resizable: bool
    min_width: int = 1
    min_height: int = 1

    @property
    def geometry(self) -> str:
        return f"{self.width}x{self.height}"


# メニューバー・タイトルバー・ドックで画面の縦から削られる分の見積り
SCREEN_RESERVED_HEIGHT = 140
# 画面が小さくてもこれ未満にはしない(これ以下だと操作部が見切れる)
FIT_FLOOR_HEIGHT = 560


def fit_to_screen(spec: WindowSpec, screen_height: int) -> WindowSpec:
    """画面の縦に収まらないとき、高さと最小の高さを画面に合わせて縮める

    13インチの標準設定(論理 1440x900)では、既定の高さがほぼ限界になるため。
    縮めた場合、ライブ画面は文字起こしを自動で折りたたんで要約を優先する。
    画面の高さが不明(0以下)のときは何もしない。
    """
    if screen_height <= 0:
        return spec
    available = max(FIT_FLOOR_HEIGHT, screen_height - SCREEN_RESERVED_HEIGHT)
    if spec.height <= available:
        return spec
    return WindowSpec(
        spec.width,
        available,
        spec.resizable,
        spec.min_width,
        min(spec.min_height, available),
    )


def window_spec(mode: str, *, log_visible: bool) -> WindowSpec:
    """モードと詳細ログの表示状態からウィンドウの大きさを決める

    ライブは表示量が多いので広く・リサイズ可にする。他モードは従来どおり固定サイズ。
    """
    if mode == MODE_LIVE:
        return WindowSpec(LIVE_WIDTH, LIVE_HEIGHT, True, LIVE_MIN_WIDTH, LIVE_MIN_HEIGHT)
    height = EXPANDED_HEIGHT if log_visible else COMPACT_HEIGHT
    return WindowSpec(NORMAL_WIDTH, height, False)


def resize_on_log_toggle(mode: str) -> bool:
    """詳細ログの開閉でウィンドウの大きさを変えるか(ライブは利用者が決めた大きさを保つ)"""
    return mode != MODE_LIVE


def normal_exec_spec(mode: str, *, recording: bool) -> tuple[str, bool]:
    """ライブ以外のモードの実行ボタン (文言, 危険色か) を返す"""
    if mode in (MODE_RECORD_TRANSCRIBE, MODE_RECORD_ONLY):
        return ("録音停止", True) if recording else ("実行", False)
    if mode == MODE_TRANSCRIBE_ONLY:
        return ("文字起こし開始", False)
    raise ValueError(f"ライブ以外のモードが必要です: {mode!r}")


def other_mode_rejection(
    *, live_active: bool, live_stream_closing: bool = False
) -> tuple[str, str] | None:
    """ライブ処理中(後処理を含む)なら、通常録音・文字起こしの開始を拒否する理由を返す

    live_stream_closing: ライブで停止がタイムアウトし放棄したストリームが、まだ解放中か
    """
    if live_active:
        return (
            "ライブ実行中",
            "ライブ文字起こし(または前回の後処理)が実行中です。終了してから開始してください",
        )
    if live_stream_closing:
        return (
            "ストリーム解放中",
            "前回のライブ録音のストリームがまだ解放されていません。"
            "しばらく待ってから再試行してください",
        )
    return None
