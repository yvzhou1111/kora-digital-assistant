from sqlalchemy import Column, String, ForeignKey, Date, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class IndividualProfile(Base):
    """个人用户资料模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    headline = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    birth_date = Column(Date, nullable=True)
    education_level = Column(String(50), nullable=True)
    years_experience = Column(Integer, nullable=True)
    current_position = Column(String(255), nullable=True)
    current_company = Column(String(255), nullable=True)
    
    # 关系
    user = relationship("User", back_populates="individual_profile")
    experiences = relationship("Experience", back_populates="individual", cascade="all, delete-orphan")
    educations = relationship("Education", back_populates="individual", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="individual", cascade="all, delete-orphan")
    skills = relationship("UserSkill", back_populates="individual", cascade="all, delete-orphan")
    job_applications = relationship("JobApplication", back_populates="individual", cascade="all, delete-orphan") 