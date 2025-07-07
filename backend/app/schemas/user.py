from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, UUID4, validator, Field


class UserBase(BaseModel):
    """用户基础模型"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None
    role: Optional[str] = None
    avatar_url: Optional[str] = None
    is_active: Optional[bool] = True


class UserCreate(UserBase):
    """用户创建模型"""
    username: str
    email: EmailStr
    password: str
    role: str = "individual"
    register_source: str = "direct"
    
    @validator("role")
    def validate_role(cls, v):
        allowed_roles = ["individual", "employer", "expert"]
        if v not in allowed_roles:
            raise ValueError(f"角色必须是以下之一: {', '.join(allowed_roles)}")
        return v


class UserUpdate(UserBase):
    """用户更新模型"""
    password: Optional[str] = None


class UserInDBBase(UserBase):
    """数据库中的用户基础模型"""
    id: UUID4
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class User(UserInDBBase):
    """API响应中的用户模型"""
    pass


class UserInDB(UserInDBBase):
    """数据库中的完整用户模型"""
    password_hash: str


class UserWithProfile(User):
    """带有资料的用户模型"""
    individual_profile: Optional[dict] = None
    employer_profile: Optional[dict] = None
    expert_profile: Optional[dict] = None


class PasswordChange(BaseModel):
    """密码修改模型"""
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator("confirm_password")
    def passwords_match(cls, v, values, **kwargs):
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("两次输入的密码不匹配")
        return v