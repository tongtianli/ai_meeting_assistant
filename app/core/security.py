"""Bearer Token（JWT）鉴权与音频短时签名 token（PRD §9.2）。

- API token：Authorization: Bearer <jwt>，scope=api
- 音频 token：签名 URL 的 query 参数，scope=audio 且绑定具体 meeting，
  短时效——<audio> 标签无法携带请求头，故播放走按需换发的签名 URL
"""
import time
import uuid

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)


def create_access_token(user_id: uuid.UUID) -> tuple[str, int]:
    now = int(time.time())
    ttl = settings.token_ttl_hours * 3600
    token = jwt.encode(
        {"sub": str(user_id), "scope": "api", "iat": now, "exp": now + ttl},
        settings.auth_secret,
        algorithm=_ALGORITHM,
    )
    return token, ttl


def create_audio_token(meeting_id: uuid.UUID) -> tuple[str, int]:
    now = int(time.time())
    ttl = settings.audio_url_ttl_seconds
    token = jwt.encode(
        {"sub": str(meeting_id), "scope": "audio", "iat": now, "exp": now + ttl},
        settings.auth_secret,
        algorithm=_ALGORITHM,
    )
    return token, ttl


def verify_audio_token(token: str, meeting_id: uuid.UUID) -> bool:
    try:
        payload = jwt.decode(token, settings.auth_secret, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return False
    return payload.get("scope") == "audio" and payload.get("sub") == str(meeting_id)


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> uuid.UUID:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials, settings.auth_secret, algorithms=[_ALGORITHM]
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if payload.get("scope") != "api":
        raise HTTPException(status_code=401, detail="wrong token scope")
    return uuid.UUID(payload["sub"])
