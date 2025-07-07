from typing import Optional
from sqlalchemy.orm import Session

from app.core.security import verify_password, get_password_hash
from app.models.user import User
from app.models.user_setting import UserSetting
from app.models.individual_profile import IndividualProfile
from app.models.employer_profile import EmployerProfile
from app.models.expert_profile import ExpertProfile
from app.schemas.user import UserCreate, UserUpdate


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """通过邮箱获取用户"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """通过用户名获取用户"""
    return db.query(User).filter(User.username == username).first()


def get_user(db: Session, user_id: str) -> Optional[User]:
    """通过ID获取用户"""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username_or_email: str, password: str) -> Optional[User]:
    """验证用户"""
    # 尝试通过用户名查找
    user = get_user_by_username(db, username_or_email)
    
    # 如果未找到，尝试通过邮箱查找
    if not user:
        user = get_user_by_email(db, username_or_email)
    
    # 如果仍未找到或密码不正确，返回None
    if not user or not verify_password(password, user.password_hash):
        return None
    
    return user


def create_user(db: Session, user_in: UserCreate) -> User:
    """创建新用户"""
    # 创建用户
    db_user = User(
        username=user_in.username,
        email=user_in.email,
        password_hash=get_password_hash(user_in.password),
        role=user_in.role,
        registration_source=user_in.register_source,
    )
    db.add(db_user)
    db.flush()  # 获取用户ID
    
    # 创建用户设置
    db_settings = UserSetting(user_id=db_user.id)
    db.add(db_settings)
    
    # 根据角色创建相应的资料
    if user_in.role == "individual":
        db_profile = IndividualProfile(user_id=db_user.id)
        db.add(db_profile)
    elif user_in.role == "employer":
        db_profile = EmployerProfile(user_id=db_user.id, company_name="未命名企业")
        db.add(db_profile)
    elif user_in.role == "expert":
        db_profile = ExpertProfile(user_id=db_user.id)
        db.add(db_profile)
    
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user: User, user_in: UserUpdate) -> User:
    """更新用户信息"""
    # 更新基本信息
    update_data = user_in.dict(exclude_unset=True)
    
    # 如果包含密码，则更新密码哈希
    if "password" in update_data and update_data["password"]:
        update_data["password_hash"] = get_password_hash(update_data.pop("password"))
    
    # 更新用户属性
    for field, value in update_data.items():
        if hasattr(user, field) and value is not None:
            setattr(user, field, value)
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user 