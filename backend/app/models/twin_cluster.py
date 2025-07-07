from sqlalchemy import Column, String, ForeignKey, DateTime, JSON, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base


class TwinCluster(Base):
    """数字分身聚类模型，对应Second-Me中的L1ClusterModel"""
    
    twin_id = Column(UUID(as_uuid=True), ForeignKey("digitaltwin.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255))
    center_embedding = Column(JSON)  # 聚类中心向量
    memory_ids = Column(JSON)  # 属于该聚类的文档ID列表
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    twin = relationship("DigitalTwin", back_populates="clusters")
    shades = relationship("TwinShade", back_populates="cluster", cascade="all, delete-orphan")
    
    def to_domain_dict(self):
        """转换为领域对象字典"""
        return {
            "id": self.id,
            "name": self.name,
            "clusterCenter": self.center_embedding,
            "memoryList": [{"memoryId": mid} for mid in self.memory_ids],
            "createTime": self.create_time.isoformat() if self.create_time else None,
            "updateTime": self.update_time.isoformat() if self.update_time else None
        }


class TwinShade(Base):
    """数字分身特征信息模型，对应Second-Me中的L1ShadeModel"""
    
    cluster_id = Column(Integer, ForeignKey("twincluster.id", ondelete="CASCADE"), nullable=False)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("digitaltwin.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255))
    aspect = Column(String(50))
    icon = Column(String(50))
    desc_third_view = Column(Text)  # 第三人称视角描述
    content_third_view = Column(Text)  # 第三人称视角内容
    desc_second_view = Column(Text)  # 第二人称视角描述
    content_second_view = Column(Text)  # 第二人称视角内容
    confidence_level = Column(String(20))  # 置信度级别
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    cluster = relationship("TwinCluster", back_populates="shades")
    twin = relationship("DigitalTwin")
    
    def to_domain_dict(self):
        """转换为领域对象字典"""
        return {
            "id": self.id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "descThirdView": self.desc_third_view,
            "contentThirdView": self.content_third_view,
            "descSecondView": self.desc_second_view,
            "contentSecondView": self.content_second_view,
            "confidenceLevel": self.confidence_level,
            "createTime": self.create_time.isoformat() if self.create_time else None,
            "updateTime": self.update_time.isoformat() if self.update_time else None
        } 