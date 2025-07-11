from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
import uuid

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.message import Message
from app.schemas.message import (
    Message as MessageSchema,
    MessageCreate,
    MessageUpdate
)
from app.services.message_service import (
    get_user_messages,
    create_message,
    mark_message_as_read,
    delete_message
)

router = APIRouter()


@router.get("", response_model=List[MessageSchema])
async def read_messages(
    skip: int = 0,
    limit: int = 100,
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取当前用户的消息
    """
    return get_user_messages(db, current_user.id, skip=skip, limit=limit, unread_only=unread_only)


@router.post("", response_model=MessageSchema)
async def create_new_message(
    message_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    创建新消息
    """
    return create_message(db, message_in, sender_id=current_user.id)


@router.put("/{message_id}/read", response_model=MessageSchema)
async def mark_as_read(
    message_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    将消息标记为已读
    """
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="消息不存在"
        )
    
    # 检查是否为消息接收者
    if str(message.recipient_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限标记此消息"
        )
    
    return mark_message_as_read(db, message)


@router.delete("/{message_id}")
async def delete_message_endpoint(
    message_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    删除消息
    """
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="消息不存在"
        )
    
    # 检查是否为消息发送者或接收者
    if str(message.sender_id) != str(current_user.id) and str(message.recipient_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此消息"
        )
    
    delete_message(db, message)
    return {"message": "消息已删除"}
