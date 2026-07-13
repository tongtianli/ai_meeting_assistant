"""转码归一化：统一转为 16kHz 单声道 wav 中间格式（PRD §2）。

未来小程序 aac 等新格式仅需扩展此入口。
"""
import asyncio
from pathlib import Path

from app.core.config import settings


class TranscodeError(RuntimeError):
    pass


async def transcode_to_wav16k_mono(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}.wav"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
    if settings.transcode_filters:
        cmd += ["-af", settings.transcode_filters]
    cmd.append(str(dest))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise TranscodeError("ffmpeg not installed") from exc
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace")[-500:]
        raise TranscodeError(f"ffmpeg failed (exit {proc.returncode}): {detail}")
    return dest


def delete_transcoded_artifacts(src: Path, dest_dir: Path) -> None:
    """删除某原始音频派生的全部转码中间产物；幂等。

    覆盖 `<stem>.wav`（转码归一化产物）与 `<stem>.upload.mp3`
    （SeedASR 超限压缩产物，见 app/services/asr/seedasr.py），
    用 glob 一次匹配，无需硬编码后缀。
    """
    if not dest_dir.is_dir():
        return
    for artifact in dest_dir.glob(f"{src.stem}.*"):
        artifact.unlink(missing_ok=True)
