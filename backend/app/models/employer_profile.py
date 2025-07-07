from sqlalchemy import Column, String, ForeignKey, Text, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class EmployerProfile(Base):
    """雇主资料模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    company_name = Column(String(100), nullable=False)
    industry = Column(String(100), nullable=True)
    company_size = Column(String(50), nullable=True)
    company_website = Column(String(255), nullable=True)
    company_description = Column(Text, nullable=True)
    company_logo_url = Column(String, nullable=True)
    location = Column(String(255), nullable=True)
    is_verified = Column(Boolean, default=False)
    
    # 关系
    user = relationship("User", back_populates="employer_profile")
    jobs = relationship("Job", back_populates="employer", cascade="all, delete-orphan") 