from sqlalchemy import Column, String, ForeignKey, Date, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Education(Base):
    """教育经历模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    individual_id = Column(UUID(as_uuid=True), ForeignKey("individualprofile.user_id", ondelete="CASCADE"), nullable=False)
    institution = Column(String(100), nullable=False)
    degree = Column(String(100), nullable=True)
    field_of_study = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    grade = Column(Float, nullable=True)
    
    # 关系
    individual = relationship("IndividualProfile", back_populates="educations") 