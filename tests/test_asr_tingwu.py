"""听悟 provider：结果映射与配置校验（真实调用需账号，见 README）。"""
import asyncio
from pathlib import Path

import pytest

from app.services.asr import TingwuProvider, get_asr_provider
from app.services.asr.tingwu import parse_transcription


def test_registered_in_factory() -> None:
    assert isinstance(get_asr_provider("tingwu"), TingwuProvider)


def test_parse_transcription_groups_sentences_and_speakers() -> None:
    payload = {
        "Paragraphs": [
            {
                "SpeakerId": "1",
                "Words": [
                    {"SentenceId": 1, "Start": 0, "End": 500, "Text": "大家"},
                    {"SentenceId": 1, "Start": 500, "End": 1200, "Text": "好。"},
                    {"SentenceId": 2, "Start": 1300, "End": 2500, "Text": "开始吧。"},
                ],
            },
            {
                "SpeakerId": "2",
                "Words": [
                    {"SentenceId": 3, "Start": 2600, "End": 4000, "Text": "好的。"},
                    {"SentenceId": 4, "Start": 4100, "End": 4200, "Text": "   "},
                ],
            },
        ]
    }
    segments = parse_transcription(payload)

    assert [s.text for s in segments] == ["大家好。", "开始吧。", "好的。"]
    assert segments[0].start_time == 0.0 and segments[0].end_time == 1.2  # ms → s
    assert segments[0].speaker_label == "speaker_001"
    assert segments[2].speaker_label == "speaker_002"
    # 时间轴有序
    assert all(
        a.start_time <= b.start_time for a, b in zip(segments, segments[1:])
    )


def test_parse_transcription_tolerates_empty() -> None:
    assert parse_transcription({}) == []
    assert parse_transcription({"Paragraphs": [{"SpeakerId": "1", "Words": []}]}) == []


def test_transcribe_without_config_raises_hint(tmp_path: Path) -> None:
    provider = TingwuProvider()
    with pytest.raises(RuntimeError, match="tingwu is not configured"):
        asyncio.run(provider.transcribe(tmp_path / "a.wav"))
