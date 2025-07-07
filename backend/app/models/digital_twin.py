from sqlalchemy import Column, String, ForeignKey, DateTime, JSON, Text, Enum, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.db.base_class import Base


class TwinType(str, enum.Enum):
    """数字分身类型枚举"""
    PROFESSIONAL = "professional"  # 职业型数字分身
    EXPERT = "expert"              # 专家型数字分身
    PERSONAL = "personal"          # 个人型数字分身
    RECRUITER = "recruiter"        # 招聘型数字分身


class TwinStatus(str, enum.Enum):
    """数字分身状态枚举"""
    INITIALIZING = "initializing"  # 初始化中
    PROCESSING = "processing"      # 处理中
    ACTIVE = "active"              # 活跃
    INACTIVE = "inactive"          # 不活跃
    FAILED = "failed"              # 失败


class DigitalTwin(Base):
    """数字分身模型"""
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    
    # 分身类型和状态
    twin_type = Column(Enum(TwinType), default=TwinType.PROFESSIONAL, nullable=False)
    status = Column(Enum(TwinStatus), default=TwinStatus.INITIALIZING, nullable=False)
    
    # 交流风格和个性化设置
    communication_style = Column(String(50), default="professional")
    personality_traits = Column(JSON, default={})
    
    # 配置信息
    configuration = Column(JSON, default={})
    
    # 生物特征信息 (从Second-Me的Bio模型引入)
    bio_content = Column(Text, nullable=True)  # 第二人称视角的详细描述
    bio_content_third_view = Column(Text, nullable=True)  # 第三人称视角的详细描述
    bio_summary = Column(Text, nullable=True)  # 第二人称视角的摘要
    bio_summary_third_view = Column(Text, nullable=True)  # 第三人称视角的摘要
    attributes = Column(JSON, default=[])  # 属性列表
    
    # 特征信息 (从Second-Me的Shade模型引入)
    shades = Column(JSON, default=[])  # 特征列表，包含各个方面的特征
    
    # 训练和更新信息
    last_trained_at = Column(DateTime, nullable=True)
    version = Column(Integer, default=1)
    embedding_model = Column(String(100), default="text-embedding-ada-002")
    llm_model = Column(String(100), default="gpt-3.5-turbo")
    
    # 向量数据库集合名称
    vector_collection_name = Column(String(100), nullable=True)
    
    # 是否公开
    is_public = Column(Boolean, default=False)
    
    # 关系
    user = relationship("User", back_populates="digital_twins")
    data_sources = relationship("TwinDataSource", back_populates="twin", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="twin", cascade="all, delete-orphan")
    clusters = relationship("TwinCluster", back_populates="twin", cascade="all, delete-orphan") 