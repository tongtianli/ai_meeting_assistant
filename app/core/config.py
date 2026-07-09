from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/meeting_assistant"
    )
    # 音频与中间产物根目录（MVP 本地磁盘；二期换对象存储时仅换存储实现）
    data_dir: Path = Path("./data")
    # ASR provider 选择（接口可替换，PRD §10）
    asr_provider: str = "mock"


settings = Settings()
