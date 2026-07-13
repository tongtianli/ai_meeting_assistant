"""确定性 Mock provider：无外部依赖地跑通完整管道（转写→入库→embedding 留存）。

用于本地开发、测试与端到端联调；云 provider 就绪后通过 ASR_PROVIDER 切换。
"""
import hashlib
import wave
from pathlib import Path

from app.services.asr.base import (
    ASRProvider,
    ASRResult,
    ASRSegment,
    SpeakerEmbeddingSample,
)

_SCRIPT = [
    "大家好，我们开始今天的项目周会。",
    "先同步一下上周的进展，后端接口已经完成了八成。",
    "前端这边列表页和详情页联调完毕。",
    "我们计划下周完成上传接口的开发。",
    "好的，那这项任务记给我，周五之前给结论。",
    "关于数据库迁移的方案大家还有问题吗？",
    "没有的话今天就先到这里，散会。",
]
_EMBEDDING_DIM = 8


def _fake_embedding(speaker_label: str) -> list[float]:
    digest = hashlib.sha256(speaker_label.encode()).digest()
    return [b / 255.0 for b in digest[:_EMBEDDING_DIM]]


def _duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, OSError, EOFError):
        return float(len(_SCRIPT) * 5)


class MockASRProvider(ASRProvider):
    name = "mock"

    async def transcribe(
        self,
        audio_path: Path,
        hotwords: list[str] | None = None,
        voiceprint_ids: list[str] | None = None,
    ) -> ASRResult:
        duration = max(_duration_seconds(audio_path), 0.001 * len(_SCRIPT))
        step = duration / len(_SCRIPT)
        segments: list[ASRSegment] = []
        first_seen: dict[str, ASRSegment] = {}
        for i, text in enumerate(_SCRIPT):
            label = f"speaker_{i % 2 + 1:03d}"
            seg = ASRSegment(
                start_time=round(i * step, 3),
                end_time=round((i + 1) * step, 3),
                speaker_label=label,
                text=text,
            )
            segments.append(seg)
            first_seen.setdefault(label, seg)
        embeddings = [
            SpeakerEmbeddingSample(
                speaker_label=label,
                embedding=_fake_embedding(label),
                model_name="mock-diarizer",
                model_version="0.1",
                sample_start=seg.start_time,
                sample_end=seg.end_time,
            )
            for label, seg in first_seen.items()
        ]
        return ASRResult(segments=segments, speaker_embeddings=embeddings)
