"""FunASR provider 单元测试。

真实推理依赖可选安装的 funasr + 模型下载，此处只测纯逻辑
（结果映射、样本片段挑选、未安装时的报错），真机验证见 README。
"""
import asyncio
import importlib.util
from pathlib import Path

import pytest

from app.services.asr import FunASRProvider, get_asr_provider
from app.services.asr.base import ASRSegment
from app.services.asr.funasr_provider import (
    parse_sentence_info,
    pick_speaker_sample_segments,
)

_FUNASR_INSTALLED = importlib.util.find_spec("funasr") is not None


def test_registered_in_factory() -> None:
    assert isinstance(get_asr_provider("funasr"), FunASRProvider)


def test_parse_sentence_info_maps_units_and_labels() -> None:
    raw = [
        {"text": " 大家好。", "start": 0, "end": 1500, "spk": 0},
        {"text": "开始吧。", "start": 1500, "end": 3200, "spk": 1},
        {"text": "   ", "start": 3200, "end": 3300, "spk": 0},  # 空文本应被过滤
        {"text": "好的。", "start": 3300, "end": 4000},  # 缺 spk 默认 0
    ]
    segments = parse_sentence_info(raw)

    assert len(segments) == 3
    assert segments[0].start_time == 0.0 and segments[0].end_time == 1.5  # ms → s
    assert segments[0].speaker_label == "speaker_001"  # spk 0 → 001
    assert segments[1].speaker_label == "speaker_002"
    assert segments[0].text == "大家好。"  # 去除首尾空白
    assert segments[2].speaker_label == "speaker_001"


def test_pick_speaker_sample_prefers_longest_segment() -> None:
    segs = [
        ASRSegment(0.0, 1.0, "speaker_001", "短"),
        ASRSegment(1.0, 5.0, "speaker_001", "这段最长"),
        ASRSegment(5.0, 6.0, "speaker_002", "另一个人"),
    ]
    best = pick_speaker_sample_segments(segs)

    assert set(best) == {"speaker_001", "speaker_002"}
    assert best["speaker_001"].text == "这段最长"


@pytest.mark.skipif(_FUNASR_INSTALLED, reason="funasr installed; error path not applicable")
def test_transcribe_without_funasr_raises_install_hint(tmp_path: Path) -> None:
    provider = FunASRProvider()
    with pytest.raises(RuntimeError, match="uv sync --extra funasr"):
        asyncio.run(provider.transcribe(tmp_path / "a.wav"))
