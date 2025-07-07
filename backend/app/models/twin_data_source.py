from sqlalchemy import Column, String, ForeignKey, DateTime, JSON, Text, Integer, Boolean, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

from app.db.base_class import Base


class SourceType(str, enum.Enum):
    """数据源类型枚举"""
    RESUME = "resume"          # 简历
    LINKEDIN = "linkedin"      # LinkedIn数据
    GITHUB = "github"          # GitHub数据
    MANUAL = "manual"          # 手动输入
    TEXT = "text"              # 文本
    MARKDOWN = "markdown"      # Markdown
    PDF = "pdf"                # PDF
    LINK = "link"              # 链接
    CHAT = "chat"              # 聊天记录


class ProcessStatus(str, enum.Enum):
    """处理状态枚举"""
    PENDING = "pending"        # 待处理
    PROCESSING = "processing"  # 处理中
    PROCESSED = "processed"    # 已处理
    FAILED = "failed"          # 失败


class TwinDataSource(Base):
    """数字分身数据源模型"""
    
    twin_id = Column(UUID(as_uuid=True), ForeignKey("digitaltwin.id", ondelete="CASCADE"), nullable=False)
    source_type = Column(String(20), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    file_path = Column(String, nullable=True)
    url = Column(String, nullable=True)
    metadata = Column(JSON, default={})
    status = Column(String(20), default="pending")
    error_message = Column(String, nullable=True)
    
    # 从Second-Me项目引入的字段
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 文档处理相关字段
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    insight = Column(Text, nullable=True)
    tags = Column(JSON, default=[])
    topic = Column(String(255), nullable=True)
    
    # 向量嵌入
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    
    # 处理状态详情
    process_status = Column(String(20), default=ProcessStatus.PENDING)
    processed_at = Column(DateTime, nullable=True)
    chunk_count = Column(Integer, default=0)
    
    # 关系
    twin = relationship("DigitalTwin", back_populates="data_sources")
    chunks = relationship("TwinChunk", back_populates="data_source", cascade="all, delete-orphan")
    
    def to_domain_dict(self):
        """转换为领域对象字典"""
        return {
            "id": self.id,
            "twinId": self.twin_id,
            "sourceType": self.source_type,
            "name": self.name,
            "description": self.description,
            "content": self.content[:1000] + "..." if self.content and len(self.content) > 1000 else self.content,
            "filePath": self.file_path,
            "url": self.url,
            "metadata": self.metadata,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "insight": self.insight,
            "tags": self.tags,
            "topic": self.topic,
            "createTime": self.create_time.isoformat() if self.create_time else None,
            "updateTime": self.update_time.isoformat() if self.update_time else None,
            "processStatus": self.process_status,
            "processedAt": self.processed_at.isoformat() if self.processed_at else None,
            "chunkCount": self.chunk_count
        }


class TwinChunk(Base):
    """数字分身数据块模型，对应Second-Me中的Chunk"""
    
    data_source_id = Column(UUID(as_uuid=True), ForeignKey("twindatasource.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)
    tags = Column(JSON, default=[])
    topic = Column(String(255), nullable=True)
    metadata = Column(JSON, default={})
    sequence = Column(Integer, default=0)  # 在文档中的序号
    relevance_score = Column(Float, nullable=True)  # 相关性分数
    
    # 关系
    data_source = relationship("TwinDataSource", back_populates="chunks")
    
    def to_domain_dict(self):
        """转换为领域对象字典"""
        return {
            "id": self.id,
            "dataSourceId": self.data_source_id,
            "content": self.content,
            "tags": self.tags,
            "topic": self.topic,
            "metadata": self.metadata,
            "sequence": self.sequence,
            "relevanceScore": self.relevance_score
        } 