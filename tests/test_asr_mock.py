import asyncio

from app.services.asr import MockASRProvider, get_asr_provider
from tests.conftest import make_wav


def test_factory_returns_mock_by_default() -> None:
    assert isinstance(get_asr_provider(), MockASRProvider)


def test_factory_rejects_unknown_provider() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown ASR provider"):
        get_asr_provider("nonexistent")


def test_mock_transcribe_produces_aligned_segments(tmp_path) -> None:
    wav = make_wav(tmp_path / "a.wav", seconds=7.0)
    result = asyncio.run(MockASRProvider().transcribe(wav))

    assert len(result.segments) > 0
    prev_end = 0.0
    for seg in result.segments:
        assert seg.start_time >= prev_end - 1e-6  # 时间轴单调不重叠
        assert seg.end_time > seg.start_time
        assert seg.speaker_label.startswith("speaker_")
        assert seg.text
        prev_end = seg.end_time
    # 结尾对齐到音频时长
    assert abs(result.segments[-1].end_time - 7.0) < 0.1


def test_mock_transcribe_produces_speaker_embeddings(tmp_path) -> None:
    wav = make_wav(tmp_path / "a.wav")
    result = asyncio.run(MockASRProvider().transcribe(wav))

    labels_in_segments = {s.speaker_label for s in result.segments}
    labels_in_embeddings = {e.speaker_label for e in result.speaker_embeddings}
    assert labels_in_embeddings == labels_in_segments
    for e in result.speaker_embeddings:
        assert len(e.embedding) == 8
        assert e.model_name and e.model_version  # 暗桩要求：带模型版本号
        assert e.sample_end > e.sample_start
