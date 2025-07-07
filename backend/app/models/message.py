from sqlalchemy import Column, String, ForeignKey, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base


class Message(Base):
    """消息模型"""
    
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    is_from_twin = Column(Boolean, default=False, nullable=False)
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages") 