"""Seed ASR 2.0 provider：utterances 映射、声纹名称优先、配置校验。"""
import asyncio
from pathlib import Path

import pytest

from app.services.asr import SeedASRProvider, get_asr_provider
from app.services.asr.seedasr import parse_utterances


def test_registered_in_factory() -> None:
    assert isinstance(get_asr_provider("seedasr"), SeedASRProvider)


def test_parse_utterances_maps_units_and_speakers() -> None:
    payload = {
        "result": {
            "text": "……",
            "utterances": [
                {
                    "text": "大家好。",
                    "start_time": 0,
                    "end_time": 1500,
                    "additions": {"speaker": "1"},
                },
                {
                    "text": "开始吧。",
                    "start_time": 1600,
                    "end_time": 3000,
                    "additions": {"speaker": "2"},
                },
                {"text": "   ", "start_time": 3000, "end_time": 3100},
            ],
        }
    }
    segments = parse_utterances(payload)

    assert len(segments) == 2
    assert segments[0].start_time == 0.0 and segments[0].end_time == 1.5
    assert segments[0].speaker_label == "speaker_001"
    assert segments[1].speaker_label == "speaker_002"


def test_parse_utterances_prefers_voiceprint_name() -> None:
    payload = {
        "result": {
            "utterances": [
                {
                    "text": "我来同步一下。",
                    "start_time": 0,
                    "end_time": 2000,
                    "additions": {"speaker": "1"},
                    "speaker_info": {"id": "vp-abc", "name": "Tim", "score": 0.92},
                }
            ]
        }
    }
    segments = parse_utterances(payload)
    # 声纹匹配命中：直接用注册名称作为标签（跨会议身份，PRD 二期能力）
    assert segments[0].speaker_label == "Tim"


def test_parse_utterances_tolerates_empty() -> None:
    assert parse_utterances({}) == []
    assert parse_utterances({"result": {"utterances": []}}) == []


def test_transcribe_without_config_raises_hint(tmp_path: Path) -> None:
    provider = SeedASRProvider()
    with pytest.raises(RuntimeError, match="seedasr is not configured"):
        asyncio.run(provider.transcribe(tmp_path / "a.wav"))
