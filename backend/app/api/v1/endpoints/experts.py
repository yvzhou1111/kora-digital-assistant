from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Body
from sqlalchemy.orm import Session
import uuid

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.expert_profile import ExpertProfile
from app.models.knowledge_base import KnowledgeBase, KnowledgeContent
from app.schemas.expert import (
    ExpertProfile as ExpertProfileSchema,
    ExpertProfileUpdate,
    KnowledgeBase as KnowledgeBaseSchema,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeContent as KnowledgeContentSchema,
    KnowledgeContentCreate
)
from app.services.expert_service import (
    get_expert_profile,
    update_expert_profile,
    get_knowledge_base,
    get_expert_knowledge_bases,
    create_knowledge_base,
    update_knowledge_base,
    delete_knowledge_base,
    add_knowledge_content,
    get_knowledge_contents,
    delete_knowledge_content
)

router = APIRouter()


@router.get("/profile", response_model=ExpertProfileSchema)
async def read_expert_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取当前专家资料
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    profile = get_expert_profile(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="专家资料不存在"
        )
    
    return profile


@router.put("/profile", response_model=ExpertProfileSchema)
async def update_expert_profile_details(
    profile_in: ExpertProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    更新当前专家资料
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    profile = get_expert_profile(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="专家资料不存在"
        )
    
    return update_expert_profile(db, profile, profile_in)


@router.get("/knowledge-bases", response_model=List[KnowledgeBaseSchema])
async def read_knowledge_bases(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取当前专家的知识库列表
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    profile = get_expert_profile(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="专家资料不存在"
        )
    
    return get_expert_knowledge_bases(db, profile.user_id, skip=skip, limit=limit)


@router.post("/knowledge-bases", response_model=KnowledgeBaseSchema)
async def create_new_knowledge_base(
    knowledge_base_in: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    创建新知识库
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    profile = get_expert_profile(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="专家资料不存在"
        )
    
    return create_knowledge_base(db, knowledge_base_in, profile.user_id)


@router.get("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseSchema)
async def read_knowledge_base(
    knowledge_base_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取特定知识库
    """
    knowledge_base = get_knowledge_base(db, knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 检查访问权限
    if not knowledge_base.public_access and str(knowledge_base.expert_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此知识库"
        )
    
    return knowledge_base


@router.put("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseSchema)
async def update_knowledge_base_details(
    knowledge_base_id: uuid.UUID,
    knowledge_base_in: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    更新知识库
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    knowledge_base = get_knowledge_base(db, knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 检查是否为知识库创建者
    if str(knowledge_base.expert_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有知识库创建者可以更新知识库"
        )
    
    return update_knowledge_base(db, knowledge_base, knowledge_base_in)


@router.delete("/knowledge-bases/{knowledge_base_id}")
async def delete_knowledge_base_endpoint(
    knowledge_base_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    删除知识库
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    knowledge_base = get_knowledge_base(db, knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 检查是否为知识库创建者
    if str(knowledge_base.expert_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有知识库创建者可以删除知识库"
        )
    
    delete_knowledge_base(db, knowledge_base)
    return {"message": "知识库已删除"}


@router.post("/knowledge-bases/{knowledge_base_id}/contents", response_model=KnowledgeContentSchema)
async def add_content_to_knowledge_base(
    knowledge_base_id: uuid.UUID,
    content_type: str = Form(...),
    title: str = Form(...),
    content_text: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    向知识库添加内容
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    knowledge_base = get_knowledge_base(db, knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 检查是否为知识库创建者
    if str(knowledge_base.expert_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有知识库创建者可以添加内容"
        )
    
    # 处理标签
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # 处理文件上传
    file_path = None
    if file:
        # 保存文件逻辑，这里简化处理
        file_path = f"uploads/knowledge/{knowledge_base_id}/{file.filename}"
        # 实际应用中需要保存文件
    
    content_in = KnowledgeContentCreate(
        content_type=content_type,
        title=title,
        content_text=content_text,
        file_path=file_path,
        description=description,
        tags=tag_list
    )
    
    return add_knowledge_content(db, knowledge_base_id, content_in)


@router.get("/knowledge-bases/{knowledge_base_id}/contents", response_model=List[KnowledgeContentSchema])
async def read_knowledge_contents(
    knowledge_base_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取知识库的所有内容
    """
    knowledge_base = get_knowledge_base(db, knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 检查访问权限
    if not knowledge_base.public_access and str(knowledge_base.expert_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此知识库"
        )
    
    return get_knowledge_contents(db, knowledge_base_id, skip=skip, limit=limit)


@router.delete("/knowledge-bases/{knowledge_base_id}/contents/{content_id}")
async def delete_knowledge_content_endpoint(
    knowledge_base_id: uuid.UUID,
    content_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    删除知识内容
    """
    # 检查用户是否为专家
    if current_user.role != "expert":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有专家可以访问此功能"
        )
    
    knowledge_base = get_knowledge_base(db, knowledge_base_id)
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 检查是否为知识库创建者
    if str(knowledge_base.expert_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有知识库创建者可以删除内容"
        )
    
    delete_knowledge_content(db, content_id)
    return {"message": "知识内容已删除"}
