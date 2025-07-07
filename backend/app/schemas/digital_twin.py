from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, UUID4, validator, Field
import uuid


class TwinDataSourceBase(BaseModel):
    """数据源基础模型"""
    source_type: str
    name: str
    description: Optional[str] = None


class TwinDataSourceCreate(TwinDataSourceBase):
    """数据源创建模型"""
    content: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TwinDataSourceUpdate(BaseModel):
    """数据源更新模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TwinDataSourceInDBBase(TwinDataSourceBase):
    """数据库中的数据源基础模型"""
    id: UUID4
    twin_id: UUID4
    content: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    status: str
    process_status: str
    error_message: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    insight: Optional[str] = None
    tags: List[str] = []
    topic: Optional[str] = None
    chunk_count: int = 0
    processed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class TwinDataSource(TwinDataSourceInDBBase):
    """API响应中的数据源模型"""
    pass


class DigitalTwinBase(BaseModel):
    """数字分身基础模型"""
    name: str
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    communication_style: Optional[str] = "professional"
    twin_type: Optional[str] = "professional"


class DigitalTwinCreate(DigitalTwinBase):
    """数字分身创建模型"""
    personality_traits: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None


class DigitalTwinUpdate(BaseModel):
    """数字分身更新模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    communication_style: Optional[str] = None
    twin_type: Optional[str] = None
    personality_traits: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None


class DigitalTwinInDBBase(DigitalTwinBase):
    """数据库中的数字分身基础模型"""
    id: UUID4
    user_id: UUID4
    status: str
    twin_type: str
    personality_traits: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any] = {}
    bio_content: Optional[str] = None
    bio_summary: Optional[str] = None
    shades: Optional[List[Dict[str, Any]]] = []
    last_trained_at: Optional[datetime] = None
    vector_collection_name: Optional[str] = None
    is_public: bool = False
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class DigitalTwin(DigitalTwinInDBBase):
    """API响应中的数字分身模型"""
    data_sources: Optional[List[TwinDataSource]] = None


class ConversationBase(BaseModel):
    """对话基础模型"""
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    """对话创建模型"""
    twin_id: UUID4


class ConversationInDBBase(ConversationBase):
    """数据库中的对话基础模型"""
    id: UUID4
    twin_id: UUID4
    user_id: UUID4
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class Conversation(ConversationInDBBase):
    """API响应中的对话模型"""
    pass


class MessageBase(BaseModel):
    """消息基础模型"""
    content: str


class MessageCreate(MessageBase):
    """消息创建模型"""
    conversation_id: UUID4


class MessageInDBBase(MessageBase):
    """数据库中的消息基础模型"""
    id: UUID4
    conversation_id: UUID4
    sender_id: Optional[UUID4] = None
    recipient_id: Optional[UUID4] = None
    is_read: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class Message(MessageInDBBase):
    """API响应中的消息模型"""
    pass


class MessageResponse(BaseModel):
    user_message: Dict[str, Any]
    twin_message: Dict[str, Any]


class TwinChunkBase(BaseModel):
    content: str
    tags: Optional[List[str]] = None
    topic: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    sequence: Optional[int] = None


class TwinChunk(TwinChunkBase):
    id: uuid.UUID
    data_source_id: uuid.UUID
    relevance_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class TwinClusterBase(BaseModel):
    name: str
    center_embedding: Optional[List[float]] = None
    memory_ids: List[str] = []


class TwinCluster(TwinClusterBase):
    id: int
    twin_id: uuid.UUID
    create_time: datetime
    update_time: datetime

    class Config:
        orm_mode = True


class TwinShadeBase(BaseModel):
    name: str
    aspect: str
    icon: Optional[str] = None
    desc_third_view: Optional[str] = None
    content_third_view: Optional[str] = None
    desc_second_view: Optional[str] = None
    content_second_view: Optional[str] = None
    confidence_level: Optional[str] = None


class TwinShade(TwinShadeBase):
    id: int
    cluster_id: int
    twin_id: uuid.UUID
    create_time: datetime
    update_time: datetime

    class Config:
        orm_mode = True 