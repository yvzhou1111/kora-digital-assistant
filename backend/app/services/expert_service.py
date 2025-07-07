from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import uuid

from app.models.expert_profile import ExpertProfile
from app.models.knowledge_base import KnowledgeBase, KnowledgeContent
from app.models.consultation_service import ConsultationService
from app.schemas.expert import (
    ExpertProfileUpdate, 
    KnowledgeBaseCreate, 
    KnowledgeBaseUpdate,
    KnowledgeContentCreate,
    ConsultationServiceCreate,
    ConsultationServiceUpdate
)


def get_expert_profile(db: Session, user_id: uuid.UUID) -> Optional[ExpertProfile]:
    """获取专家资料"""
    return db.query(ExpertProfile).filter(ExpertProfile.user_id == user_id).first()


def update_expert_profile(db: Session, profile: ExpertProfile, profile_in: ExpertProfileUpdate) -> ExpertProfile:
    """更新专家资料"""
    update_data = profile_in.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(profile, field) and value is not None:
            setattr(profile, field, value)
    
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def get_knowledge_base(db: Session, knowledge_base_id: uuid.UUID) -> Optional[KnowledgeBase]:
    """获取知识库"""
    return db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()


def get_expert_knowledge_bases(
    db: Session, 
    expert_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100
) -> List[KnowledgeBase]:
    """获取专家的所有知识库"""
    return db.query(KnowledgeBase).filter(
        KnowledgeBase.expert_id == expert_id
    ).order_by(KnowledgeBase.created_at.desc()).offset(skip).limit(limit).all()


def create_knowledge_base(db: Session, knowledge_base_in: KnowledgeBaseCreate, expert_id: uuid.UUID) -> KnowledgeBase:
    """创建知识库"""
    db_knowledge_base = KnowledgeBase(
        expert_id=expert_id,
        name=knowledge_base_in.name,
        description=knowledge_base_in.description,
        expertise_areas=knowledge_base_in.expertise_areas,
        public_access=knowledge_base_in.public_access if knowledge_base_in.public_access is not None else True,
        configuration=knowledge_base_in.configuration or {},
        status="created"
    )
    db.add(db_knowledge_base)
    db.commit()
    db.refresh(db_knowledge_base)
    return db_knowledge_base


def update_knowledge_base(db: Session, knowledge_base: KnowledgeBase, knowledge_base_in: KnowledgeBaseUpdate) -> KnowledgeBase:
    """更新知识库"""
    update_data = knowledge_base_in.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(knowledge_base, field) and value is not None:
            setattr(knowledge_base, field, value)
    
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


def delete_knowledge_base(db: Session, knowledge_base: KnowledgeBase) -> None:
    """删除知识库"""
    db.delete(knowledge_base)
    db.commit()


def add_knowledge_content(db: Session, knowledge_base_id: uuid.UUID, content_in: KnowledgeContentCreate) -> KnowledgeContent:
    """添加知识内容"""
    db_content = KnowledgeContent(
        knowledge_base_id=knowledge_base_id,
        title=content_in.title,
        content_type=content_in.content_type,
        file_path=content_in.file_path,
        content_text=content_in.content_text,
        description=content_in.description,
        tags=content_in.tags,
        metadata=content_in.metadata or {},
        status="processing"
    )
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    
    # 在实际应用中，这里应该启动一个异步任务来处理内容
    # 处理完成后更新状态为"processed"
    
    return db_content


def get_knowledge_contents(
    db: Session, 
    knowledge_base_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100
) -> List[KnowledgeContent]:
    """获取知识库的所有内容"""
    return db.query(KnowledgeContent).filter(
        KnowledgeContent.knowledge_base_id == knowledge_base_id
    ).order_by(KnowledgeContent.created_at.desc()).offset(skip).limit(limit).all()


def delete_knowledge_content(db: Session, content_id: uuid.UUID) -> None:
    """删除知识内容"""
    content = db.query(KnowledgeContent).filter(KnowledgeContent.id == content_id).first()
    if content:
        db.delete(content)
        db.commit()


def create_consultation_service(db: Session, service_in: ConsultationServiceCreate, expert_id: uuid.UUID) -> ConsultationService:
    """创建咨询服务"""
    db_service = ConsultationService(
        expert_id=expert_id,
        title=service_in.title,
        description=service_in.description,
        service_type=service_in.service_type,
        pricing_model=service_in.pricing_model,
        price_amount=service_in.price_amount,
        price_currency=service_in.price_currency,
        availability=service_in.availability or {},
        status=service_in.status or "active"
    )
    db.add(db_service)
    db.commit()
    db.refresh(db_service)
    return db_service


def get_expert_consultation_services(
    db: Session, 
    expert_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100
) -> List[ConsultationService]:
    """获取专家的所有咨询服务"""
    return db.query(ConsultationService).filter(
        ConsultationService.expert_id == expert_id
    ).order_by(ConsultationService.created_at.desc()).offset(skip).limit(limit).all()


def update_consultation_service(db: Session, service: ConsultationService, service_in: ConsultationServiceUpdate) -> ConsultationService:
    """更新咨询服务"""
    update_data = service_in.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(service, field) and value is not None:
            setattr(service, field, value)
    
    db.add(service)
    db.commit()
    db.refresh(service)
    return service 