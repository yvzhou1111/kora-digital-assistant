from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.schemas.user import User as UserSchema, UserUpdate, PasswordChange
from app.services.user_service import update_user, get_user
from app.core.security import verify_password, get_password_hash

router = APIRouter()


@router.get("/me", response_model=UserSchema)
async def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取当前用户信息
    """
    return current_user


@router.put("/me", response_model=UserSchema)
async def update_user_me(
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    更新当前用户信息
    """
    user = update_user(db, current_user, user_in)
    return user


@router.put("/me/password", response_model=dict)
async def change_password(
    password_in: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    修改当前用户密码
    """
    # 验证当前密码
    if not verify_password(password_in.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前密码不正确"
        )
    
    # 更新密码
    current_user.password_hash = get_password_hash(password_in.new_password)
    db.add(current_user)
    db.commit()
    
    return {"message": "密码修改成功"}
