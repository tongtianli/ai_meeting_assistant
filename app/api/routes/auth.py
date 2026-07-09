"""MVP 鉴权：单默认用户，凭访问口令换取 JWT（PRD §6 不做多用户注册登录）。"""
import hmac

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import create_access_token
from app.models import DEFAULT_USER_ID

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/token", response_model=TokenOut)
async def issue_token(body: TokenRequest) -> TokenOut:
    if not hmac.compare_digest(body.password, settings.auth_password):
        raise HTTPException(status_code=401, detail="invalid password")
    token, ttl = create_access_token(DEFAULT_USER_ID)
    return TokenOut(access_token=token, expires_in=ttl)
