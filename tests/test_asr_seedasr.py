"""Seed ASR 2.0 provider：utterances 映射、声纹名称优先、配置校验。"""
import asyncio
from pathlib import Path

import pytest

from app.services.asr import SeedASRProvider, get_asr_provider
from app.services.asr.seedasr import (
    build_audio_field,
    parse_utterances,
    pick_bitrate,
)


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
    # 命中信息回填，供管道自动绑定 Person
    assert segments[0].voiceprint_id == "vp-abc"
    assert segments[0].voiceprint_confidence == 0.92


def test_build_request_serializes_voiceprint_list_as_json_string() -> None:
    import json

    request = SeedASRProvider()._build_request(
        hotwords=None, voiceprint_ids=["vp-1", "vp-2"]
    )
    # 网关 params store 只收字符串（实测），必须 JSON 编码
    assert json.loads(request["voice_print_list"]) == ["vp-1", "vp-2"]
    # 未传声纹时不携带该字段
    assert "voice_print_list" not in SeedASRProvider()._build_request(hotwords=None)


def test_parse_utterances_tolerates_empty() -> None:
    assert parse_utterances({}) == []
    assert parse_utterances({"result": {"utterances": []}}) == []


def test_pick_bitrate_adapts_to_duration() -> None:
    limit = 11 * 1024 * 1024  # 默认直传上限（网关 16MB 请求体 / base64 4/3）
    # 短音频顶格 64kbps
    assert pick_bitrate(600, limit) == 64_000
    # 40 分钟：64kbps 会超限，需降档且结果落在上限内
    bitrate = pick_bitrate(2400, limit)
    assert 16_000 <= bitrate < 64_000
    assert bitrate * 2400 / 8 <= limit
    # 时长未知：保守用 64kbps（提交失败时错误信息可见）
    assert pick_bitrate(0, limit) == 64_000


def test_pick_bitrate_rejects_overlong_audio() -> None:
    with pytest.raises(RuntimeError, match="too long"):
        pick_bitrate(3 * 3600, 11 * 1024 * 1024)  # 3 小时超出 16kbps 兜底


def test_build_audio_field_small_file_uses_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import make_wav

    wav = make_wav(tmp_path / "small.wav", seconds=1.0)
    field = asyncio.run(build_audio_field(wav))
    assert field["format"] == "wav"
    assert "data" in field and "url" not in field


def test_build_audio_field_large_file_prefers_signed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import settings
    from app.core.security import verify_file_token
    from tests.conftest import make_wav

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "seedasr_max_upload_mb", 0)  # 强制超限
    monkeypatch.setattr(
        settings, "public_base_url", "https://meeting.example.com/"
    )
    wav = make_wav(tmp_path / "transcoded" / "big.wav", seconds=2.0)

    field = asyncio.run(build_audio_field(wav))
    assert field["format"] == "mp3"
    assert "data" not in field
    # URL 指向签名文件路由，token 可验回 data_dir 相对路径
    prefix = "https://meeting.example.com/api/audio/file/"
    assert field["url"].startswith(prefix)
    rel = verify_file_token(field["url"].removeprefix(prefix))
    assert rel == "transcoded/big.upload.mp3"


def test_build_audio_field_no_public_url_falls_back_to_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import settings
    from tests.conftest import make_wav

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "public_base_url", "")
    # 2 秒 wav 约 64KB；上限设 0.05MB 触发压缩路径（mp3 后落回限内）
    monkeypatch.setattr(settings, "seedasr_max_upload_mb", 1)
    wav = make_wav(tmp_path / "transcoded" / "mid.wav", seconds=120.0)
    # 120s wav ≈ 3.8MB > 1MB 上限 → 自适应压缩
    field = asyncio.run(build_audio_field(wav))
    assert field["format"] == "mp3"
    assert "data" in field and "url" not in field


def test_transcribe_without_config_raises_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 隔离本机 .env 里的真实凭证
    monkeypatch.setattr("app.core.config.settings.volc_app_key", "")
    monkeypatch.setattr("app.core.config.settings.volc_access_key", "")
    provider = SeedASRProvider()
    with pytest.raises(RuntimeError, match="seedasr is not configured"):
        asyncio.run(provider.transcribe(tmp_path / "a.wav"))
