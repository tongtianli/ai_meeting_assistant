"""音频存储抽象。

MVP 用本地磁盘；二期换 S3/OSS（预签名直传、range、生命周期归档，PRD §9.2）
时新增一个实现即可，audio_url 的 URI 形式保持稳定。
音频为不可变资产：只写入、不修改（PRD 设计原则 5）。
"""
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings

_SCHEME = "local://"


class AudioStorage(ABC):
    @abstractmethod
    def save(self, stream: BinaryIO, filename: str) -> str:
        """保存音频，返回可入库的 audio_url（storage URI）。"""

    @abstractmethod
    def resolve(self, audio_url: str) -> Path:
        """将 audio_url 解析为本地可读路径。"""

    @abstractmethod
    def delete(self, audio_url: str) -> None:
        """删除音频（用户显式删除会议时触发，PRD §9.4）；幂等。"""


class LocalAudioStorage(AudioStorage):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, stream: BinaryIO, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        key = f"{uuid.uuid4().hex}{suffix}"
        with (self.root / key).open("wb") as f:
            shutil.copyfileobj(stream, f)
        return f"{_SCHEME}{key}"

    def resolve(self, audio_url: str) -> Path:
        if not audio_url.startswith(_SCHEME):
            raise ValueError(f"unsupported audio_url: {audio_url!r}")
        return self.root / audio_url.removeprefix(_SCHEME)

    def delete(self, audio_url: str) -> None:
        self.resolve(audio_url).unlink(missing_ok=True)


def get_audio_storage() -> AudioStorage:
    return LocalAudioStorage(settings.data_dir / "audio")
