from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from app.models.digital_twin import DigitalTwin, TwinType, TwinStatus
from app.models.twin_data_source import TwinDataSource, TwinChunk, SourceType, ProcessStatus
from app.models.twin_cluster import TwinCluster, TwinShade
from app.models.conversation import Conversation, Message
from app.schemas.digital_twin import DigitalTwinCreate, DigitalTwinUpdate, TwinDataSourceCreate
from app.ai.twin_engine import TwinEngine


class DigitalTwinService:
    """数字分身服务类，集成Second-Me功能"""
    
    def __init__(self, db: Session):
        """
        初始化数字分身服务
        
        参数:
        - db: 数据库会话
        """
        self.db = db
        self.twin_engine = TwinEngine(db)
    
    def create_digital_twin(self, obj_in: DigitalTwinCreate, user_id: uuid.UUID) -> DigitalTwin:
        """
        创建数字分身
        
        参数:
        - obj_in: 创建数字分身的输入数据
        - user_id: 用户ID
        
        返回:
        - 创建的数字分身对象
        """
        twin_data = obj_in.dict(exclude_unset=True)
        
        # 设置默认值
        if "twin_type" not in twin_data:
            twin_data["twin_type"] = TwinType.PROFESSIONAL
        
        if "status" not in twin_data:
            twin_data["status"] = TwinStatus.INITIALIZING
        
        # 创建数字分身
        twin = DigitalTwin(
            **twin_data,
            user_id=user_id
        )
        self.db.add(twin)
        self.db.commit()
        self.db.refresh(twin)
        
        return twin
    
    def get_digital_twin(self, twin_id: uuid.UUID) -> Optional[DigitalTwin]:
        """
        获取数字分身
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 数字分身对象，如果不存在则返回None
        """
        return self.db.query(DigitalTwin).filter(DigitalTwin.id == twin_id).first()
    
    def get_user_digital_twins(self, user_id: uuid.UUID) -> List[DigitalTwin]:
        """
        获取用户的所有数字分身
        
        参数:
        - user_id: 用户ID
        
        返回:
        - 数字分身列表
        """
        return self.db.query(DigitalTwin).filter(DigitalTwin.user_id == user_id).all()
    
    def update_digital_twin(self, twin_id: uuid.UUID, obj_in: DigitalTwinUpdate) -> Optional[DigitalTwin]:
        """
        更新数字分身
        
        参数:
        - twin_id: 数字分身ID
        - obj_in: 更新数据
        
        返回:
        - 更新后的数字分身对象，如果不存在则返回None
        """
        twin = self.get_digital_twin(twin_id)
        if not twin:
            return None
        
        update_data = obj_in.dict(exclude_unset=True)
        
        # 更新数字分身
        for field, value in update_data.items():
            setattr(twin, field, value)
        
        self.db.commit()
        self.db.refresh(twin)
        
        return twin
    
    def delete_digital_twin(self, twin_id: uuid.UUID) -> bool:
        """
        删除数字分身
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 是否删除成功
        """
        twin = self.get_digital_twin(twin_id)
        if not twin:
            return False
        
        self.db.delete(twin)
        self.db.commit()
        
        return True
    
    def add_data_source(self, twin_id: uuid.UUID, source_type: str, name: str, content: str = None, 
                       file_path: str = None, url: str = None, metadata: Dict = None) -> Optional[TwinDataSource]:
        """
        添加数据源
        
        参数:
        - twin_id: 数字分身ID
        - source_type: 数据源类型
        - name: 数据源名称
        - content: 数据源内容
        - file_path: 文件路径
        - url: URL
        - metadata: 元数据
        
        返回:
        - 创建的数据源对象，如果数字分身不存在则返回None
        """
        twin = self.get_digital_twin(twin_id)
        if not twin:
            return None
        
        # 创建数据源
        data_source = TwinDataSource(
            twin_id=twin_id,
            source_type=source_type,
            name=name,
            content=content,
            file_path=file_path,
            url=url,
            metadata=metadata or {},
            process_status=ProcessStatus.PENDING
        )
        self.db.add(data_source)
        self.db.commit()
        self.db.refresh(data_source)
        
        # 异步处理数据源（在实际应用中应该使用异步任务）
        self.twin_engine.process_data_source(data_source.id)
        
        return data_source
    
    def get_data_sources(self, twin_id: uuid.UUID) -> List[TwinDataSource]:
        """
        获取数字分身的所有数据源
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 数据源列表
        """
        return self.db.query(TwinDataSource).filter(TwinDataSource.twin_id == twin_id).all()
    
    def create_conversation(self, twin_id: uuid.UUID, title: str = None) -> Optional[Conversation]:
        """
        创建对话
        
        参数:
        - twin_id: 数字分身ID
        - title: 对话标题
        
        返回:
        - 创建的对话对象，如果数字分身不存在则返回None
        """
        twin = self.get_digital_twin(twin_id)
        if not twin:
            return None
        
        # 创建对话
        conversation = Conversation(
            twin_id=twin_id,
            title=title or f"与{twin.name}的对话"
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        
        return conversation
    
    def get_conversations(self, twin_id: uuid.UUID) -> List[Conversation]:
        """
        获取数字分身的所有对话
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 对话列表
        """
        return self.db.query(Conversation).filter(Conversation.twin_id == twin_id).all()
    
    def add_message(self, conversation_id: uuid.UUID, content: str, is_from_twin: bool = False) -> Optional[Message]:
        """
        添加消息
        
        参数:
        - conversation_id: 对话ID
        - content: 消息内容
        - is_from_twin: 是否来自数字分身
        
        返回:
        - 创建的消息对象，如果对话不存在则返回None
        """
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return None
        
        # 创建消息
        message = Message(
            conversation_id=conversation_id,
            content=content,
            is_from_twin=is_from_twin
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        
        return message
    
    def get_messages(self, conversation_id: uuid.UUID) -> List[Message]:
        """
        获取对话的所有消息
        
        参数:
        - conversation_id: 对话ID
        
        返回:
        - 消息列表
        """
        return self.db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    
    def send_message_to_twin(self, conversation_id: uuid.UUID, content: str) -> Dict[str, Any]:
        """
        向数字分身发送消息并获取回复
        
        参数:
        - conversation_id: 对话ID
        - content: 消息内容
        
        返回:
        - 包含用户消息和数字分身回复的字典
        """
        # 获取对话
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return {"error": "对话不存在"}
        
        # 添加用户消息
        user_message = self.add_message(conversation_id, content, is_from_twin=False)
        
        # 生成数字分身回复
        twin_response = self.twin_engine.generate_twin_response(
            conversation.twin_id, 
            content, 
            conversation_id
        )
        
        # 添加数字分身回复
        twin_message = self.add_message(conversation_id, twin_response, is_from_twin=True)
        
        return {
            "user_message": {
                "id": str(user_message.id),
                "content": user_message.content,
                "created_at": user_message.created_at.isoformat()
            },
            "twin_message": {
                "id": str(twin_message.id),
                "content": twin_message.content,
                "created_at": twin_message.created_at.isoformat()
            }
        }
    
    def generate_twin_bio(self, twin_id: uuid.UUID) -> bool:
        """
        生成数字分身的生物特征信息
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 生成是否成功
        """
        return self.twin_engine._update_twin_bio(twin_id)
    
    def generate_twin_clusters(self, twin_id: uuid.UUID) -> bool:
        """
        为数字分身生成聚类
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 生成是否成功
        """
        return self.twin_engine.generate_clusters(twin_id)
    
    def get_twin_clusters(self, twin_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        获取数字分身的所有聚类
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 聚类列表
        """
        clusters = self.db.query(TwinCluster).filter(TwinCluster.twin_id == twin_id).all()
        return [cluster.to_domain_dict() for cluster in clusters]
    
    def get_twin_shades(self, twin_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        获取数字分身的所有特征
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 特征列表
        """
        shades = self.db.query(TwinShade).filter(TwinShade.twin_id == twin_id).all()
        return [shade.to_domain_dict() for shade in shades] 