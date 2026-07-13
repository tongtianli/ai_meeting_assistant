"""转码前处理：响度归一化把小音量语音拉起来（VAD 误杀空白段的对策）。"""
import asyncio
import math
import struct
import wave
from pathlib import Path

import pytest

from app.core.config import settings
from app.services.transcode import TranscodeError, transcode_to_wav16k_mono


def _write_quiet_sine(path: Path, amplitude: int, seconds: float = 2.0) -> None:
    rate = 16000
    frames = bytearray()
    for i in range(int(rate * seconds)):
        sample = int(amplitude * math.sin(2 * math.pi * 440 * i / rate))
        frames += struct.pack("<h", sample)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _rms(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def test_loudnorm_boosts_quiet_audio(tmp_path) -> None:
    src = tmp_path / "quiet.wav"
    _write_quiet_sine(src, amplitude=300)  # 约 -40dBFS 的小音量
    dest = asyncio.run(transcode_to_wav16k_mono(src, tmp_path / "out"))

    with wave.open(str(dest), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
    # 响度归一化后 RMS 应显著提升（远场小音量被拉起，VAD 才不会当静音丢掉）
    assert _rms(dest) > _rms(src) * 3


def test_filters_can_be_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "transcode_filters", "")
    src = tmp_path / "quiet.wav"
    _write_quiet_sine(src, amplitude=300)
    dest = asyncio.run(transcode_to_wav16k_mono(src, tmp_path / "out"))
    # 无滤镜：纯转码，电平基本不变
    assert _rms(dest) == pytest.approx(_rms(src), rel=0.2)


def test_transcode_error_on_garbage(tmp_path) -> None:
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not-audio")
    with pytest.raises(TranscodeError):
        asyncio.run(transcode_to_wav16k_mono(bad, tmp_path / "out"))
