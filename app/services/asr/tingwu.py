"""通义听悟 provider：会议场景云 ASR（远场/抢话/多人分离，PRD §10 云 API 优先路线）。

流程：本地 wav → 上传 OSS → 预签名 URL → 听悟 CreateTask（离线转写 +
DiarizationEnabled）→ 轮询 GetTaskInfo → 拉取转写 JSON → 映射统一 segment。

- 依赖较轻但需账号配置，作为可选依赖安装：`uv sync --extra tingwu`
- 需要环境变量：ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET /
  TINGWU_APP_KEY / OSS_ENDPOINT / OSS_BUCKET
- 注意：听悟不返回 speaker embedding，voice_samples 暗桩在此 provider
  下为空（本地 CAM++ 补采样为后续增强项）
"""
import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.services.asr.base import ASRProvider, ASRResult, ASRSegment

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "tingwu dependencies not installed; run: uv sync --extra tingwu"
)
_CONFIG_HINT = (
    "tingwu is not configured; set ALIYUN_ACCESS_KEY_ID, "
    "ALIYUN_ACCESS_KEY_SECRET, TINGWU_APP_KEY, OSS_ENDPOINT, OSS_BUCKET"
)


def parse_transcription(transcription: dict[str, Any]) -> list[ASRSegment]:
    """听悟转写结果 JSON → 统一 ASRSegment（毫秒 → 秒，SpeakerId → 标签）。

    结构：Paragraphs[].SpeakerId + Words[]{SentenceId, Start, End, Text}，
    按 SentenceId 聚合成句。
    """
    segments: list[ASRSegment] = []
    for paragraph in transcription.get("Paragraphs", []):
        speaker = str(paragraph.get("SpeakerId", "0"))
        label = f"speaker_{int(speaker):03d}" if speaker.isdigit() else f"speaker_{speaker}"
        sentences: dict[Any, dict[str, Any]] = {}
        for word in paragraph.get("Words", []):
            sid = word.get("SentenceId", 0)
            bucket = sentences.setdefault(
                sid, {"text": [], "start": None, "end": None}
            )
            bucket["text"].append(word.get("Text", ""))
            start, end = word.get("Start"), word.get("End")
            if start is not None:
                bucket["start"] = (
                    start if bucket["start"] is None else min(bucket["start"], start)
                )
            if end is not None:
                bucket["end"] = (
                    end if bucket["end"] is None else max(bucket["end"], end)
                )
        for sid in sorted(sentences):
            bucket = sentences[sid]
            text = "".join(bucket["text"]).strip()
            if not text or bucket["start"] is None or bucket["end"] is None:
                continue
            segments.append(
                ASRSegment(
                    start_time=bucket["start"] / 1000.0,
                    end_time=bucket["end"] / 1000.0,
                    speaker_label=label,
                    text=text,
                )
            )
    segments.sort(key=lambda s: (s.start_time, s.end_time))
    return segments


class TingwuProvider(ASRProvider):
    name = "tingwu"

    def _check_config(self) -> None:
        required = (
            settings.aliyun_access_key_id,
            settings.aliyun_access_key_secret,
            settings.tingwu_app_key,
            settings.oss_endpoint,
            settings.oss_bucket,
        )
        if not all(required):
            raise RuntimeError(_CONFIG_HINT)

    def _upload_to_oss(self, audio_path: Path) -> str:
        """上传音频并换发 24h 预签名 URL 供听悟服务端拉取。"""
        try:
            import oss2
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        auth = oss2.Auth(
            settings.aliyun_access_key_id, settings.aliyun_access_key_secret
        )
        bucket = oss2.Bucket(auth, settings.oss_endpoint, settings.oss_bucket)
        key = f"{settings.oss_prefix}{uuid.uuid4().hex}{audio_path.suffix}"
        bucket.put_object_from_file(key, str(audio_path))
        return bucket.sign_url("GET", key, 24 * 3600, slash_safe=True)

    def _acs_request(self, method: str, uri: str, body: dict | None = None) -> dict:
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdkcore.request import CommonRequest
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        client = AcsClient(
            settings.aliyun_access_key_id,
            settings.aliyun_access_key_secret,
            settings.tingwu_region,
        )
        request = CommonRequest()
        request.set_accept_format("json")
        request.set_domain(f"tingwu.{settings.tingwu_region}.aliyuncs.com")
        request.set_version("2023-09-30")
        request.set_protocol_type("https")
        request.set_method(method)
        request.set_uri_pattern(uri)
        request.add_header("Content-Type", "application/json")
        if body is not None:
            request.set_content(json.dumps(body).encode())
        response = client.do_action_with_exception(request)
        return json.loads(response)

    def _create_task(self, file_url: str, hotwords: list[str] | None) -> str:
        transcription: dict[str, Any] = {
            "DiarizationEnabled": True,
            "Diarization": {"SpeakerCount": 0},  # 0 = 自动判定人数
        }
        body: dict[str, Any] = {
            "AppKey": settings.tingwu_app_key,
            "Input": {
                "SourceLanguage": "cn",
                "FileUrl": file_url,
                "TaskKey": f"meeting-{uuid.uuid4().hex}",
            },
            "Parameters": {"Transcription": transcription},
        }
        if hotwords:
            # 热词注入（PRD Feature 1）：听悟词表以自定义热词形式随任务下发
            body["Parameters"]["Transcription"]["PhraseId"] = None
            body["Parameters"]["Transcription"]["Hotwords"] = hotwords[:100]
        data = self._acs_request("PUT", "/openapi/tingwu/v2/tasks?type=offline", body)
        task_id = (data.get("Data") or {}).get("TaskId")
        if not task_id:
            raise RuntimeError(f"tingwu CreateTask failed: {data}")
        return task_id

    def _get_task(self, task_id: str) -> dict:
        data = self._acs_request("GET", f"/openapi/tingwu/v2/tasks/{task_id}")
        return data.get("Data") or {}

    async def transcribe(
        self,
        audio_path: Path,
        hotwords: list[str] | None = None,
        voiceprint_ids: list[str] | None = None,  # 听悟无声纹库能力，忽略
    ) -> ASRResult:
        self._check_config()
        file_url = await asyncio.to_thread(self._upload_to_oss, audio_path)
        task_id = await asyncio.to_thread(self._create_task, file_url, hotwords)
        logger.info("tingwu task created: %s", task_id)

        elapsed = 0.0
        while True:
            await asyncio.sleep(settings.tingwu_poll_interval_seconds)
            elapsed += settings.tingwu_poll_interval_seconds
            info = await asyncio.to_thread(self._get_task, task_id)
            status = (info.get("TaskStatus") or "").upper()
            if status == "COMPLETED":
                break
            if status == "FAILED":
                raise RuntimeError(f"tingwu task failed: {info}")
            if elapsed > settings.tingwu_timeout_seconds:
                raise TimeoutError(f"tingwu task {task_id} timed out after {elapsed}s")

        result_url = (info.get("Result") or {}).get("Transcription")
        if not result_url:
            raise RuntimeError(f"tingwu completed without transcription url: {info}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(result_url)
            resp.raise_for_status()
            transcription = resp.json()

        segments = parse_transcription(
            transcription.get("Transcription", transcription)
        )
        # 听悟不产出 speaker embedding：voice_samples 暗桩留空（见模块注释）
        return ASRResult(segments=segments, speaker_embeddings=[])
