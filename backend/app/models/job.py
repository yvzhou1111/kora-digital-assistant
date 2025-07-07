from sqlalchemy import Column, String, ForeignKey, Text, Integer, Boolean, JSON, Date, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Job(Base):
    """职位模型"""
    
    employer_id = Column(UUID(as_uuid=True), ForeignKey("employerprofile.user_id", ondelete="CASCADE"), nullable=False)
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(ARRAY(String), nullable=True)
    responsibilities = Column(ARRAY(String), nullable=True)
    location = Column(String(255), nullable=True)
    salary_min = Column(Integer, nullable=True)
    salary_max = Column(Integer, nullable=True)
    salary_currency = Column(String(10), nullable=True)
    employment_type = Column(String(50), nullable=True)
    experience_level = Column(String(50), nullable=True)
    status = Column(String(20), default="draft", nullable=False)
    deadline = Column(Date, nullable=True)
    
    # 关系
    employer = relationship("EmployerProfile", back_populates="jobs")
    applications = relationship("JobApplication", back_populates="job", cascade="all, delete-orphan") 