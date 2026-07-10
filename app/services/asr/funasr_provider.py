"""FunASR 本地推理 provider：Paraformer-zh 转写 + CAM++ 说话人分离/声纹。

- 依赖较重（torch 等），作为可选依赖安装：`uv sync --extra funasr`
- 模型首次运行自动从 ModelScope 下载（约 1-2GB），之后走本地缓存
- 数据不出域：录音全程本地处理，合规压力最小（PRD §9.4）
- CPU 可推理但长音频较慢；PRD §9.1 的时延目标按云 API 制定，本地路线需接受更长处理时间
"""
import asyncio
import logging
import wave
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.asr.base import (
    ASRProvider,
    ASRResult,
    ASRSegment,
    SpeakerEmbeddingSample,
)

logger = logging.getLogger(__name__)

_VAD_MODEL = "fsmn-vad"
_PUNC_MODEL = "ct-punc"
_SPK_MODEL = "cam++"
_INSTALL_HINT = (
    "funasr is not installed; install the optional dependency group first: "
    "uv sync --extra funasr"
)


def parse_sentence_info(sentence_info: list[dict[str, Any]]) -> list[ASRSegment]:
    """FunASR sentence_info（毫秒时间戳 + 整型 spk id）→ 统一 ASRSegment。

    文本字段因模型/版本而异：paraformer 系为 "text"，
    Fun-ASR-Nano（funasr>=1.3）为 "sentence"。
    """
    segments: list[ASRSegment] = []
    for item in sentence_info:
        text = (item.get("text") or item.get("sentence") or "").strip()
        if not text:
            continue
        spk = int(item.get("spk", 0))
        segments.append(
            ASRSegment(
                start_time=item["start"] / 1000.0,
                end_time=item["end"] / 1000.0,
                speaker_label=f"speaker_{spk + 1:03d}",
                text=text,
            )
        )
    return segments


def pick_speaker_sample_segments(
    segments: list[ASRSegment],
) -> dict[str, ASRSegment]:
    """每个说话人取时长最长的 segment 作为声纹样本片段（信噪比最好的近似）。"""
    best: dict[str, ASRSegment] = {}
    for seg in segments:
        cur = best.get(seg.speaker_label)
        if cur is None or (seg.end_time - seg.start_time) > (
            cur.end_time - cur.start_time
        ):
            best[seg.speaker_label] = seg
    return best


def _load_wav_clip(audio_path: Path, start: float, end: float):
    """读取 16kHz mono wav 的一个片段，返回 float32 numpy 数组（[-1, 1]）。"""
    import numpy as np

    with wave.open(str(audio_path), "rb") as w:
        rate = w.getframerate()
        w.setpos(int(start * rate))
        frames = w.readframes(max(int((end - start) * rate), 1))
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


class FunASRProvider(ASRProvider):
    name = "funasr"

    # 模型进程内单例：加载一次，多次转写复用
    _pipeline = None
    _spk_encoder = None
    _version: str = "unknown"

    @classmethod
    def _load_models(cls):
        try:
            import funasr
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        if cls._pipeline is None:
            logger.info("loading FunASR models (first run downloads from ModelScope)")
            cls._version = getattr(funasr, "__version__", "unknown")
            cls._pipeline = AutoModel(
                model=settings.funasr_model,
                vad_model=_VAD_MODEL,
                # 会议远场场景放宽语音/噪音阈值，减少小音量语句被 VAD 丢弃
                vad_kwargs={
                    "speech_noise_thres": settings.funasr_speech_noise_thres
                },
                punc_model=_PUNC_MODEL,
                spk_model=_SPK_MODEL,
                disable_update=True,
            )
            cls._spk_encoder = AutoModel(model=_SPK_MODEL, disable_update=True)
        return cls._pipeline, cls._spk_encoder

    async def transcribe(
        self, audio_path: Path, hotwords: list[str] | None = None
    ) -> ASRResult:
        # 推理是重 CPU 任务，放线程池避免阻塞事件循环
        return await asyncio.to_thread(self._transcribe_sync, audio_path, hotwords)

    def _transcribe_sync(
        self, audio_path: Path, hotwords: list[str] | None
    ) -> ASRResult:
        pipeline, spk_encoder = self._load_models()
        kwargs: dict[str, Any] = {"batch_size_s": 300}
        if hotwords:
            kwargs["hotword"] = " ".join(hotwords)  # 热词注入（PRD Feature 1）
        raw = pipeline.generate(input=str(audio_path), **kwargs)
        sentence_info = raw[0].get("sentence_info", []) if raw else []
        if not sentence_info:
            # 常见于 spk/punc 组合未生效（如模型与 funasr 版本不匹配时
            # diarization 被静默禁用）——带上原始输出结构便于诊断
            keys = sorted(raw[0].keys()) if raw else []
            raise RuntimeError(
                "FunASR returned no sentence_info (diarization inactive?); "
                f"raw output keys: {keys}. 若日志中出现 'Missing punc_model' "
                "等提示，请升级 funasr: uv lock --upgrade-package funasr && "
                "uv sync --extra funasr"
            )
        segments = parse_sentence_info(sentence_info)
        embeddings = self._extract_speaker_embeddings(
            spk_encoder, audio_path, segments
        )
        return ASRResult(segments=segments, speaker_embeddings=embeddings)

    def _extract_speaker_embeddings(
        self, spk_encoder, audio_path: Path, segments: list[ASRSegment]
    ) -> list[SpeakerEmbeddingSample]:
        """暗桩：为每个说话人留存一条声纹 embedding（person_id 由二期绑定）。

        提取失败不阻断管道——转录仍是一期核心价值，缺失以 warning 暴露。
        """
        samples: list[SpeakerEmbeddingSample] = []
        for label, seg in sorted(pick_speaker_sample_segments(segments).items()):
            try:
                clip = _load_wav_clip(audio_path, seg.start_time, seg.end_time)
                res = spk_encoder.generate(input=clip, fs=16000)
                emb = res[0].get("spk_embedding")
                if emb is None:
                    emb = res[0].get("embedding")
                if emb is None:
                    raise KeyError(f"no embedding in output keys: {sorted(res[0])}")
                if hasattr(emb, "flatten"):
                    emb = emb.flatten().tolist()
                samples.append(
                    SpeakerEmbeddingSample(
                        speaker_label=label,
                        embedding=[float(x) for x in emb],
                        model_name=_SPK_MODEL,
                        model_version=self._version,
                        sample_start=seg.start_time,
                        sample_end=seg.end_time,
                    )
                )
            except Exception:
                logger.warning(
                    "speaker embedding extraction failed for %s (%s)",
                    label,
                    audio_path,
                    exc_info=True,
                )
        return samples
