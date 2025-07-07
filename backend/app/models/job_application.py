from sqlalchemy import Column, String, ForeignKey, Text, Boolean, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class JobApplication(Base):
    """职位申请模型"""
    
    job_id = Column(UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), nullable=False)
    individual_id = Column(UUID(as_uuid=True), ForeignKey("individualprofile.user_id", ondelete="CASCADE"), nullable=False)
    cover_letter = Column(Text, nullable=True)
    resume_url = Column(String, nullable=True)
    status = Column(
        String(20), 
        default="applied",
        nullable=False
    )
    use_digital_twin = Column(Boolean, default=False)
    match_score = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    
    # 关系
    job = relationship("Job", back_populates="applications")
    individual = relationship("IndividualProfile", back_populates="job_applications") 