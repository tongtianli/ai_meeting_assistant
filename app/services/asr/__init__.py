from app.core.config import settings
from app.services.asr.base import (
    ASRProvider,
    ASRResult,
    ASRSegment,
    SpeakerEmbeddingSample,
)
from app.services.asr.funasr_provider import FunASRProvider
from app.services.asr.mock import MockASRProvider

_PROVIDERS: dict[str, type[ASRProvider]] = {
    MockASRProvider.name: MockASRProvider,
    FunASRProvider.name: FunASRProvider,
    # 云 provider（阿里云/腾讯云等）在此注册
}


def get_asr_provider(name: str | None = None) -> ASRProvider:
    key = name or settings.asr_provider
    try:
        return _PROVIDERS[key]()
    except KeyError:
        raise ValueError(
            f"unknown ASR provider: {key!r}; available: {sorted(_PROVIDERS)}"
        ) from None


__all__ = [
    "ASRProvider",
    "ASRResult",
    "ASRSegment",
    "SpeakerEmbeddingSample",
    "FunASRProvider",
    "MockASRProvider",
    "get_asr_provider",
]
