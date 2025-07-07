from sqlalchemy import Column, String, ForeignKey, Text, Boolean, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class KnowledgeBase(Base):
    """知识库模型"""
    
    expert_id = Column(UUID(as_uuid=True), ForeignKey("expertprofile.user_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    expertise_areas = Column(ARRAY(String), nullable=True)
    public_access = Column(Boolean, default=True)
    status = Column(String(20), default="created", nullable=False)
    configuration = Column(JSON, default={})
    
    # 关系
    expert = relationship("ExpertProfile", back_populates="knowledge_bases")
    contents = relationship("KnowledgeContent", back_populates="knowledge_base", cascade="all, delete-orphan")


class KnowledgeContent(Base):
    """知识内容模型"""
    
    knowledge_base_id = Column(UUID(as_uuid=True), ForeignKey("knowledgebase.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    content_type = Column(String(50), nullable=False)
    file_path = Column(String, nullable=True)
    content_text = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    status = Column(String(20), default="processing", nullable=False)
    metadata = Column(JSON, default={})
    
    # 关系
    knowledge_base = relationship("KnowledgeBase", back_populates="contents") 