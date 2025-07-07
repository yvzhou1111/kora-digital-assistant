from sqlalchemy import Column, String, ForeignKey, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base


class Conversation(Base):
    """对话模型"""
    
    twin_id = Column(UUID(as_uuid=True), ForeignKey("digitaltwin.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    
    # 关系
    twin = relationship("DigitalTwin", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """消息模型"""
    
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    recipient_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    content = Column(Text, nullable=False)
    message_type = Column(String(20), default="chat")
    is_from_twin = Column(Boolean, default=False)
    metadata = Column(JSON, default={})
    is_read = Column(Boolean, default=False)
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("User", foreign_keys=[recipient_id], back_populates="received_messages") 