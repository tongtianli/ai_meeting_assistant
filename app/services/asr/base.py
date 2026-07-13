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
    # 云端声纹匹配命中时回填（跨会议身份，PRD 二期）；未命中为 None
    voiceprint_id: str | None = None
    voiceprint_confidence: float | None = None


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
        self,
        audio_path: Path,
        hotwords: list[str] | None = None,
        voiceprint_ids: list[str] | None = None,
    ) -> ASRResult:
        """输入 16kHz 单声道 wav；hotwords 为热词/术语表注入机制（PRD Feature 1）。

        voiceprint_ids：已注册的云端声纹 ID 列表，支持声纹匹配的 provider
        （目前仅 seedasr）据此做跨会议说话人识别；其余 provider 忽略。
        """
