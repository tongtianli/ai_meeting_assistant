"""转码归一化：统一转为 16kHz 单声道 wav 中间格式（PRD §2）。

未来小程序 aac 等新格式仅需扩展此入口。
"""
import asyncio
from pathlib import Path


class TranscodeError(RuntimeError):
    pass


async def transcode_to_wav16k_mono(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}.wav"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            str(dest),
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
