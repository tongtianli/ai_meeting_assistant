"""可替换的 ASR provider 接口（云 API ↔ 自建模型可切换，PRD §3/§10）。

provider 负责转写 + 说话人分离 + 对齐三合一，对外输出统一 segment——
ASR 与 diarization 两条时间轴的对齐合并是 provider 内部职责。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ASRSegment:
    """对齐后的统一 segment（PRD Feature 1）。"""

    start_time: float
    end_time: float
    speaker_label: str  # 会议内临时标签，如 "speaker_001"
    text: str


@dataclass
class SpeakerEmbeddingSample:
    """diarization 顺手产出的声纹样本（MVP 必做暗桩，入 voice_samples 表）。"""

    speaker_label: str
    embedding: list[float]
    model_name: str
    model_version: str
    sample_start: float
    sample_end: float


@dataclass
class ASRResult:
    segments: list[ASRSegment]
    speaker_embeddings: list[SpeakerEmbeddingSample] = field(default_factory=list)


class ASRProvider(ABC):
    name: str

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, hotwords: list[str] | None = None
    ) -> ASRResult:
        """输入 16kHz 单声道 wav；hotwords 为热词/术语表注入机制（PRD Feature 1）。"""
