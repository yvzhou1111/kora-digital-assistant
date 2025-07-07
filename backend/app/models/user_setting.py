from sqlalchemy import Column, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class UserSetting(Base):
    """用户设置模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    notification_preferences = Column(JSON, default={})
    privacy_settings = Column(JSON, default={})
    ui_preferences = Column(JSON, default={})
    
    # 关系
    user = relationship("User", back_populates="settings") 