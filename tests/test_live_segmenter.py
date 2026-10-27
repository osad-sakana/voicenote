"""live.segmenter のテスト"""

import dataclasses

import numpy as np
import pytest

from live.segmenter import Segment, Segmenter

SR = 16000
FRAME = 512
FRAME_SEC = FRAME / SR


def _frames(seconds: float) -> int:
    return round(seconds * SR / FRAME)


def _run(segmenter: Segmenter, pattern: list[tuple[float, bool]]) -> list[Segment]:
    """(秒数, 発話か) の並びを push し、出力区間を集める(flush は含まない)"""
    out: list[Segment] = []
    for seconds, speech in pattern:
        for _ in range(_frames(seconds)):
            frame = np.full(FRAME, 0.1 if speech else 0.0, dtype=np.float32)
            out.extend(segmenter.push(frame, 1.0 if speech else 0.0))
    return out


def test_silence_only_yields_no_segments():
    """無音のみなら区間は出ない"""
    seg = Segmenter()

    out = _run(seg, [(3.0, False)])

    assert out == []
    assert seg.flush() == []


def test_speech_then_silence_yields_one_segment_with_pre_roll():
    """発話→無音で1区間。開始は発話開始の pre_roll 前、終了は hangover 後"""
    seg = Segmenter(onset_frames=2, pre_roll_ms=300, hangover_ms=600)

    out = _run(seg, [(1.0, False), (1.0, True), (1.0, False)])

    assert len(out) == 1
    s = out[0]
    speech_start = _frames(1.0) * FRAME_SEC
    speech_end = _frames(2.0) * FRAME_SEC
    assert s.start_sec == pytest.approx(speech_start - 0.3, abs=FRAME_SEC)
    assert s.start_sec < speech_start
    assert s.end_sec == pytest.approx(speech_end + 0.6, abs=FRAME_SEC)
    assert len(s.samples) == round((s.end_sec - s.start_sec) * SR)
    assert s.samples.dtype == np.float32
    # pre_roll 部分は無音、発話部分は非ゼロ
    pre_roll_samples = round((speech_start - s.start_sec) * SR)
    assert np.all(s.samples[:pre_roll_samples] == 0.0)
    assert s.samples[pre_roll_samples] == pytest.approx(0.1)
    assert seg.flush() == []


def test_pre_roll_does_not_go_before_zero():
    """録音開始直後の発話では start_sec は 0 未満にならない"""
    seg = Segmenter(onset_frames=2)

    out = _run(seg, [(1.0, True), (1.0, False)])

    assert len(out) == 1
    assert out[0].start_sec == 0.0


def test_short_blip_is_discarded():
    """min_segment_ms 未満の短い発話は破棄される(pre_roll/hangover は含めない)"""
    seg = Segmenter(onset_frames=1, min_segment_ms=400)

    out = _run(seg, [(1.0, False), (0.2, True), (2.0, False)])

    assert out == []
    assert seg.flush() == []


def test_speech_at_min_length_is_kept():
    """min_segment_ms 以上の発話は残る"""
    seg = Segmenter(onset_frames=1, min_segment_ms=400)

    out = _run(seg, [(1.0, False), (0.5, True), (2.0, False)])

    assert len(out) == 1


def test_onset_requires_consecutive_speech_frames():
    """onset_frames 未満の単発の発話判定では区間を開始しない"""
    seg = Segmenter(onset_frames=3, min_segment_ms=0)
    pattern = [(FRAME_SEC, True), (FRAME_SEC, False)] * 50

    out = _run(seg, pattern)

    assert out == []
    assert seg.flush() == []


def test_long_continuous_speech_is_split_at_max_segment():
    """20秒の連続発話は max_segment_sec で強制分割される"""
    seg = Segmenter(onset_frames=2, max_segment_sec=15)

    out = _run(seg, [(20.0, True), (1.0, False)])

    assert len(out) == 2
    first, second = out
    assert first.end_sec - first.start_sec == pytest.approx(15.0, abs=FRAME_SEC)
    # 分割点ではギャップも重複もなく連続する
    assert second.start_sec == pytest.approx(first.end_sec)
    assert second.end_sec == pytest.approx(20.0 + 0.6, abs=2 * FRAME_SEC)
    total = len(first.samples) + len(second.samples)
    assert total == round((second.end_sec - first.start_sec) * SR)


def test_short_continuation_after_forced_cut_is_kept():
    """強制分割後の短い続き(min_segment 未満)は破棄せず残す(語尾の欠落防止)"""
    seg = Segmenter(onset_frames=2, max_segment_sec=15, min_segment_ms=400)

    out = _run(seg, [(15.2, True), (1.0, False)])

    assert len(out) == 2
    assert out[1].start_sec == pytest.approx(out[0].end_sec)


def test_cut_during_hangover_does_not_emit_silent_segment():
    """hangover 中に最大長で分割されても、無音だけの区間は出さない"""
    seg = Segmenter(onset_frames=2, max_segment_sec=15, min_segment_ms=0)

    out = _run(seg, [(14.9, True), (1.0, False)]) + seg.flush()

    assert len(out) == 1


def test_flush_mid_speech_returns_open_segment():
    """発話途中で flush すると開いている区間を返し、状態をリセットする"""
    seg = Segmenter(onset_frames=2)

    out = _run(seg, [(0.5, False), (1.0, True)])
    flushed = seg.flush()

    assert out == []
    assert len(flushed) == 1
    assert flushed[0].end_sec == pytest.approx(_frames(1.5) * FRAME_SEC)
    assert seg.flush() == []


def test_flush_mid_short_speech_discards_it():
    """flush 時も min_segment 未満なら破棄"""
    seg = Segmenter(onset_frames=1, min_segment_ms=400)

    _run(seg, [(0.5, False), (0.1, True)])

    assert seg.flush() == []


def test_segments_are_monotonic_and_non_overlapping():
    """複数区間の時刻は単調増加し、重複しない"""
    seg = Segmenter(onset_frames=2, max_segment_sec=3)
    pattern = [
        (0.5, False),
        (1.0, True),
        (1.0, False),
        (0.5, True),
        (0.3, False),
        (0.5, True),
        (1.0, False),
        (7.0, True),
        (0.2, False),
    ]

    out = _run(seg, pattern) + seg.flush()

    assert len(out) >= 4
    for s in out:
        assert s.end_sec > s.start_sec >= 0.0
    for prev, nxt in zip(out[:-1], out[1:], strict=True):
        assert nxt.start_sec >= prev.end_sec


def test_short_silence_within_hangover_keeps_segment_open():
    """hangover より短い無音では区間は切れない"""
    seg = Segmenter(onset_frames=2, hangover_ms=600)

    out = _run(seg, [(1.0, True), (0.3, False), (1.0, True), (1.0, False)])

    assert len(out) == 1


def test_push_does_not_keep_reference_to_input_frame():
    """push 後に入力フレームを書き換えても区間の音声に影響しない"""
    seg = Segmenter(onset_frames=1, min_segment_ms=0)
    frames = [np.full(FRAME, 0.5, dtype=np.float32) for _ in range(5)]
    for f in frames:
        seg.push(f, 1.0)
    for f in frames:
        f[:] = 0.0

    (segment,) = seg.flush()

    assert np.all(segment.samples == 0.5)


def test_segment_is_frozen():
    """Segment は frozen dataclass"""
    s = Segment(start_sec=0.0, end_sec=1.0, samples=np.zeros(4, dtype=np.float32))

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.start_sec = 2.0  # type: ignore[misc]
    assert s.duration_sec == 1.0


def test_custom_speech_threshold():
    """speech_threshold 未満の確率は無音扱い"""
    seg = Segmenter(onset_frames=1, min_segment_ms=0, speech_threshold=0.8)
    for _ in range(30):
        seg.push(np.zeros(FRAME, dtype=np.float32), 0.7)

    assert seg.flush() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"onset_frames": 0},
        {"frame_size": 0},
        {"sample_rate": 0},
        {"max_segment_sec": 0},
        {"hangover_ms": -1},
    ],
)
def test_invalid_parameters_raise(kwargs):
    """不正なパラメータは ValueError"""
    with pytest.raises(ValueError):
        Segmenter(**kwargs)


class TestSkip:
    """skip(n): 破棄された録音サンプル分だけ時刻を進める"""

    def test_later_segment_shifts_by_skipped_samples(self):
        pattern_before = [(1.0, False)]
        pattern_after = [(1.0, False), (1.0, True), (1.0, False)]
        base = Segmenter()
        _run(base, pattern_before)
        expected = _run(base, pattern_after)

        gapped = Segmenter()
        _run(gapped, pattern_before)
        skipped = 8000  # 0.5秒分
        assert gapped.skip(skipped) == []
        actual = _run(gapped, pattern_after)

        assert len(actual) == len(expected) == 1
        assert actual[0].start_sec == pytest.approx(expected[0].start_sec + skipped / SR)
        assert actual[0].end_sec == pytest.approx(expected[0].end_sec + skipped / SR)

    def test_skip_closes_open_segment(self):
        """欠損をまたいで音声を連結しないよう、開いている区間は欠損の手前で閉じる"""
        seg = Segmenter()
        _run(seg, [(1.0, True)])
        closed = seg.skip(1600)
        assert len(closed) == 1
        assert closed[0].end_sec == pytest.approx(_frames(1.0) * FRAME_SEC)
        assert seg.flush() == []

    def test_skip_zero_is_noop(self):
        seg = Segmenter()
        _run(seg, [(1.0, True)])
        assert seg.skip(0) == []
        assert len(seg.flush()) == 1

    def test_negative_skip_raises(self):
        with pytest.raises(ValueError):
            Segmenter().skip(-1)
