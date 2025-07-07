from typing import Optional
from pydantic import BaseModel, UUID4


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None


class TokenPayload(BaseModel):
    """令牌载荷模型"""
    sub: Optional[str] = None
    exp: Optional[int] = None
    type: Optional[str] = None


class RefreshToken(BaseModel):
    """刷新令牌请求模型"""
    refresh_token: str 