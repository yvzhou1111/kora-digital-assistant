from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, UUID4, validator, Field


class ExpertProfileBase(BaseModel):
    """专家资料基础模型"""
    title: Optional[str] = None
    bio: Optional[str] = None
    expertise_areas: Optional[List[str]] = None
    years_experience: Optional[int] = None
    education: Optional[str] = None
    certifications: Optional[List[str]] = None


class ExpertProfileUpdate(ExpertProfileBase):
    """专家资料更新模型"""
    pass


class ExpertProfileInDBBase(ExpertProfileBase):
    """数据库中的专家资料基础模型"""
    user_id: UUID4
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class ExpertProfile(ExpertProfileInDBBase):
    """API响应中的专家资料模型"""
    username: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None


class KnowledgeBaseBase(BaseModel):
    """知识库基础模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    expertise_areas: Optional[List[str]] = None
    public_access: Optional[bool] = True
    configuration: Optional[Dict[str, Any]] = None


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """知识库创建模型"""
    name: str


class KnowledgeBaseUpdate(KnowledgeBaseBase):
    """知识库更新模型"""
    pass


class KnowledgeBaseInDBBase(KnowledgeBaseBase):
    """数据库中的知识库基础模型"""
    id: UUID4
    expert_id: UUID4
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class KnowledgeBase(KnowledgeBaseInDBBase):
    """API响应中的知识库模型"""
    expert_name: Optional[str] = None
    content_count: Optional[int] = None


class KnowledgeContentBase(BaseModel):
    """知识内容基础模型"""
    title: Optional[str] = None
    content_type: Optional[str] = None
    file_path: Optional[str] = None
    content_text: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeContentCreate(KnowledgeContentBase):
    """知识内容创建模型"""
    title: str
    content_type: str


class KnowledgeContentUpdate(KnowledgeContentBase):
    """知识内容更新模型"""
    pass


class KnowledgeContentInDBBase(KnowledgeContentBase):
    """数据库中的知识内容基础模型"""
    id: UUID4
    knowledge_base_id: UUID4
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class KnowledgeContent(KnowledgeContentInDBBase):
    """API响应中的知识内容模型"""
    pass


class ConsultationServiceBase(BaseModel):
    """咨询服务基础模型"""
    title: Optional[str] = None
    description: Optional[str] = None
    service_type: Optional[str] = None
    pricing_model: Optional[str] = None
    price_amount: Optional[float] = None
    price_currency: Optional[str] = None
    availability: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class ConsultationServiceCreate(ConsultationServiceBase):
    """咨询服务创建模型"""
    title: str
    service_type: str
    pricing_model: str


class ConsultationServiceUpdate(ConsultationServiceBase):
    """咨询服务更新模型"""
    pass


class ConsultationServiceInDBBase(ConsultationServiceBase):
    """数据库中的咨询服务基础模型"""
    id: UUID4
    expert_id: UUID4
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class ConsultationService(ConsultationServiceInDBBase):
    """API响应中的咨询服务模型"""
    expert_name: Optional[str] = None 