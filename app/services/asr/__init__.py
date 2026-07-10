from app.core.config import settings
from app.services.asr.base import (
    ASRProvider,
    ASRResult,
    ASRSegment,
    SpeakerEmbeddingSample,
)
from app.services.asr.funasr_provider import FunASRProvider
from app.services.asr.mock import MockASRProvider
from app.services.asr.tingwu import TingwuProvider

_PROVIDERS: dict[str, type[ASRProvider]] = {
    MockASRProvider.name: MockASRProvider,
    FunASRProvider.name: FunASRProvider,
    TingwuProvider.name: TingwuProvider,
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
    "TingwuProvider",
    "get_asr_provider",
]
