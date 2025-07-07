import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建所有表
async def init_db() -> None:
    """初始化数据库"""
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表创建成功")
        
        # 这里可以添加初始数据，如管理员账号等
        # await create_initial_data()
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise

# 创建初始数据
async def create_initial_data() -> None:
    """创建初始数据"""
    # 这里可以添加初始数据，如预设技能、系统角色等
    pass 