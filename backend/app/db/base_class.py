import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import as_declarative, declared_attr


@as_declarative()
class Base:
    """SQLAlchemy模型的基类"""
    
    id: Any
    __name__: str
    
    # 生成表名
    @declared_attr
    def __tablename__(cls) -> str:
        """自动生成表名为类名的小写形式"""
        return cls.__name__.lower()
    
    # 通用列
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 