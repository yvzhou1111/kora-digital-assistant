from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class OAuthAccount(Base):
    """OAuth账号模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(20), nullable=False)
    provider_user_id = Column(String(255), nullable=False)
    access_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    # 关系
    user = relationship("User", back_populates="oauth_accounts")
    
    # 联合唯一约束
    __table_args__ = (
        {"UniqueConstraint": ("provider", "provider_user_id")},
    ) 