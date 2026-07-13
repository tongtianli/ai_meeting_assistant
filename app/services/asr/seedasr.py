"""火山引擎 Seed ASR 2.0 provider（大模型录音文件识别）。

- 异步两段式：submit（base64 直传或 URL）→ 轮询 query → utterances 映射
- 说话人分离：enable_speaker_info + show_utterances，utterance 的
  additions.speaker 为会议内说话人编号
- 声纹匹配（形态 A）：通过 SEEDASR_EXTRA_REQUEST 传入 voice_print_list
  等附加参数；命中的声纹名称若出现在返回中将直接用作 speaker_label
  （完整的 Person 绑定流在声纹集成 PR 中实现）
- 零额外依赖（纯 HTTP）；字段名以控制台示例为准，接口细节可通过
  VOLC_BASE_URL / VOLC_RESOURCE_ID / SEEDASR_EXTRA_REQUEST 调整
"""
import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.services.asr.base import ASRProvider, ASRResult, ASRSegment
from app.services.transcode import TranscodeError

logger = logging.getLogger(__name__)

_CONFIG_HINT = "seedasr is not configured; set VOLC_APP_KEY and VOLC_ACCESS_KEY"
_OK_STATUS = "20000000"


def parse_utterances(payload: dict[str, Any]) -> list[ASRSegment]:
    """查询结果 → 统一 ASRSegment（毫秒 → 秒）。

    speaker 优先级：声纹匹配名称（speaker_info.name）> 分离编号（additions.speaker）。
    """
    result = payload.get("result") or payload
    segments: list[ASRSegment] = []
    for utt in result.get("utterances") or []:
        text = (utt.get("text") or "").strip()
        start, end = utt.get("start_time"), utt.get("end_time")
        if not text or start is None or end is None:
            continue
        additions = utt.get("additions") or {}
        speaker_info = utt.get("speaker_info") or additions.get("speaker_info") or {}
        vp_name = speaker_info.get("name") if isinstance(speaker_info, dict) else None
        raw_speaker = additions.get("speaker") or utt.get("speaker") or "0"
        if vp_name:
            label = str(vp_name)
        elif str(raw_speaker).isdigit():
            label = f"speaker_{int(raw_speaker):03d}"
        else:
            label = f"speaker_{raw_speaker}"
        segments.append(
            ASRSegment(
                start_time=start / 1000.0,
                end_time=end / 1000.0,
                speaker_label=label,
                text=text,
            )
        )
    segments.sort(key=lambda s: (s.start_time, s.end_time))
    return segments


async def _probe_duration_seconds(audio_path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0


def pick_bitrate(duration_seconds: float, limit_bytes: int) -> int:
    """按时长选码率（bps），保证 mp3 落在直传上限内（留 4% 封装余量）。

    上限 64kbps（16k 语音识别的无损级），下限 16kbps；再低识别质量
    不可接受 —— 超长音频应走 URL 模式（对象存储）而非继续压。
    """
    if duration_seconds <= 0:
        return 64_000
    bitrate = int(limit_bytes * 8 * 0.96 / duration_seconds)
    if bitrate < 16_000:
        raise RuntimeError(
            f"audio too long for seedasr direct upload "
            f"({duration_seconds / 60:.0f}min > ~90min at 16kbps); "
            "use URL-based submission (object storage) instead"
        )
    return min(64_000, bitrate)


async def _compress_if_needed(audio_path: Path) -> tuple[bytes, str]:
    """超过直传上限的音频压成单声道 mp3，码率按时长自适应。"""
    data = audio_path.read_bytes()
    limit = settings.seedasr_max_upload_mb * 1024 * 1024
    if len(data) <= limit:
        return data, audio_path.suffix.lstrip(".").lower() or "wav"
    duration = await _probe_duration_seconds(audio_path)
    bitrate = pick_bitrate(duration, limit)
    mp3_path = audio_path.with_suffix(".upload.mp3")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(audio_path), "-ac", "1",
        "-b:a", str(bitrate),
        str(mp3_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise TranscodeError(
            f"compress for upload failed: {stderr.decode(errors='replace')[-300:]}"
        )
    compressed = mp3_path.read_bytes()
    if len(compressed) > limit:
        raise RuntimeError(
            f"seedasr upload still exceeds limit after compression "
            f"({len(compressed) / 1e6:.1f}MB > {limit / 1e6:.1f}MB)"
        )
    logger.info(
        "seedasr: compressed %s (%.1fMB) -> mp3 %dkbps (%.1fMB) for upload",
        audio_path.name,
        len(data) / 1e6,
        bitrate // 1000,
        len(compressed) / 1e6,
    )
    return compressed, "mp3"


class SeedASRProvider(ASRProvider):
    name = "seedasr"

    def _headers(self, request_id: str) -> dict[str, str]:
        return {
            "X-Api-App-Key": settings.volc_app_key,
            "X-Api-Access-Key": settings.volc_access_key,
            "X-Api-Resource-Id": settings.volc_resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json",
        }

    def _build_request(self, hotwords: list[str] | None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_speaker_info": True,  # 说话人分离
            "show_utterances": True,  # 分句 + 时间戳
        }
        if hotwords:
            # 热词注入（PRD Feature 1）；2.0 上下文机制字段以控制台示例为准
            request["corpus"] = {"context": json.dumps(
                {"hotwords": [{"word": w} for w in hotwords[:100]]},
                ensure_ascii=False,
            )}
        if settings.seedasr_extra_request:
            try:
                request.update(json.loads(settings.seedasr_extra_request))
            except json.JSONDecodeError:
                logger.warning("SEEDASR_EXTRA_REQUEST is not valid JSON; ignored")
        return request

    @staticmethod
    def _status_code(resp: httpx.Response) -> str:
        return resp.headers.get("X-Api-Status-Code", "")

    async def transcribe(
        self, audio_path: Path, hotwords: list[str] | None = None
    ) -> ASRResult:
        if not (settings.volc_app_key and settings.volc_access_key):
            raise RuntimeError(_CONFIG_HINT)

        audio_bytes, audio_format = await _compress_if_needed(audio_path)
        request_id = str(uuid.uuid4())
        submit_body = {
            "user": {"uid": "ai-meeting-assistant"},
            "audio": {
                "format": audio_format,
                "data": base64.b64encode(audio_bytes).decode(),
            },
            "request": self._build_request(hotwords),
        }

        # trust_env=False：不继承终端的 http(s)_proxy —— 字节端点国内直连即可，
        # 走本地代理时大体积 POST 常被代理层掐断（表现为 ReadError）
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0), trust_env=False
        ) as client:
            resp = None
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{settings.volc_base_url}/submit",
                        json=submit_body,
                        headers=self._headers(request_id),
                    )
                    break
                except httpx.TransportError as exc:
                    if attempt == 2:
                        raise
                    logger.warning(
                        "seedasr submit transport error (attempt %d/3): %s",
                        attempt + 1, exc,
                    )
                    await asyncio.sleep(2 * (attempt + 1))
            status = self._status_code(resp)
            if resp.status_code != 200 or (status and status != _OK_STATUS):
                raise RuntimeError(
                    f"seedasr submit failed: http={resp.status_code} "
                    f"api_status={status} body={resp.text[:300]}"
                )
            logger.info("seedasr task submitted: %s", request_id)

            elapsed = 0.0
            while True:
                await asyncio.sleep(settings.seedasr_poll_interval_seconds)
                elapsed += settings.seedasr_poll_interval_seconds
                resp = await client.post(
                    f"{settings.volc_base_url}/query",
                    json={},
                    headers=self._headers(request_id),
                )
                status = self._status_code(resp)
                if status == _OK_STATUS:
                    break  # 任务完成
                if status in ("20000001", "20000002"):  # 处理中 / 排队中
                    if elapsed > settings.seedasr_timeout_seconds:
                        raise TimeoutError(
                            f"seedasr task {request_id} timed out after {elapsed}s"
                        )
                    continue
                raise RuntimeError(
                    f"seedasr query failed: http={resp.status_code} "
                    f"api_status={status} body={resp.text[:300]}"
                )

        payload = resp.json()
        segments = parse_utterances(payload)
        # 声纹匹配命中时名称已体现在 speaker_label；embedding 由云端管理，
        # voice_samples 暗桩留空（与听悟相同的缺口，本地 CAM++ 补采样为增强项）
        return ASRResult(segments=segments, speaker_embeddings=[])
