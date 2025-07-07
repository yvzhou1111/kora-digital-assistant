from sqlalchemy import Column, String, ForeignKey, Text, Boolean, ARRAY, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class ExpertProfile(Base):
    """专家资料模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    title = Column(String(100), nullable=True)
    bio = Column(Text, nullable=True)
    expertise_areas = Column(ARRAY(String), nullable=True)
    years_experience = Column(Integer, nullable=True)
    education = Column(String(255), nullable=True)
    certifications = Column(ARRAY(String), nullable=True)
    is_verified = Column(Boolean, default=False)
    
    # 关系
    user = relationship("User", back_populates="expert_profile")
    knowledge_bases = relationship("KnowledgeBase", back_populates="expert", cascade="all, delete-orphan")
    consultation_services = relationship("ConsultationService", back_populates="expert", cascade="all, delete-orphan") 