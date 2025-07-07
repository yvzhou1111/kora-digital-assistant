from sqlalchemy import Column, String, ForeignKey, Text, Boolean, JSON, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class ConsultationService(Base):
    """咨询服务模型"""
    
    expert_id = Column(UUID(as_uuid=True), ForeignKey("expertprofile.user_id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    service_type = Column(String(50), nullable=False)
    pricing_model = Column(String(50), nullable=False)
    price_amount = Column(Float, nullable=True)
    price_currency = Column(String(10), nullable=True)
    availability = Column(JSON, default={})
    status = Column(String(20), default="active", nullable=False)
    
    # 关系
    expert = relationship("ExpertProfile", back_populates="consultation_services") 