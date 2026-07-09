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

    # 鉴权（PRD §9.2：全站 Bearer Token JWT，MVP 单默认用户）
    # HS256 密钥需 ≥32 字节；生产环境必须通过环境变量覆盖
    auth_secret: str = "dev-only-secret-change-me-0123456789abcdef"
    auth_password: str = "dev-password"
    token_ttl_hours: int = 24
    # 音频播放短时签名 URL 有效期（秒）
    audio_url_ttl_seconds: int = 600


settings = Settings()
