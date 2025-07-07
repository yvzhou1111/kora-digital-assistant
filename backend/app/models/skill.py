from sqlalchemy import Column, String, Text, Float, Integer, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Skill(Base):
    """技能模型"""
    
    name = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(50), nullable=True, index=True)
    description = Column(Text, nullable=True)
    
    # 关系
    user_skills = relationship("UserSkill", back_populates="skill")


class UserSkill(Base):
    """用户技能关联模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    individual_id = Column(UUID(as_uuid=True), ForeignKey("individualprofile.user_id", ondelete="CASCADE"), nullable=True)
    skill_id = Column(UUID(as_uuid=True), ForeignKey("skill.id", ondelete="CASCADE"), primary_key=True)
    proficiency_level = Column(Integer, nullable=True)
    years_experience = Column(Float, nullable=True)
    is_highlighted = Column(Boolean, default=False)
    
    # 关系
    skill = relationship("Skill", back_populates="user_skills")
    individual = relationship("IndividualProfile", back_populates="skills") 