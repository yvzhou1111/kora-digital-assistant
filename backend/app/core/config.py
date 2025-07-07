import os
import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置类"""
    # 应用配置
    APP_NAME: str = "个人数字分身平台"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # 安全配置
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # 数据库配置
    DATABASE_URL: PostgresDsn
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # 文件存储配置
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # AI模型配置
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = "https://api.openai.com/v1"
    DOUBAO_API_KEY: Optional[str] = None
    DOUBAO_API_BASE: Optional[str] = None
    
    # 向量数据库配置
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # 跨域配置
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Second-Me 集成配置
    APP_ROOT: str = os.getenv("APP_ROOT", os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    PREFER_LANGUAGE: str = os.getenv("PREFER_LANGUAGE", "Chinese")
    L1_GENERATOR_TEMPERATURE: float = float(os.getenv("L1_GENERATOR_TEMPERATURE", "0"))
    L1_GENERATOR_MAX_TOKENS: int = int(os.getenv("L1_GENERATOR_MAX_TOKENS", "2000"))
    L1_GENERATOR_TOP_P: float = float(os.getenv("L1_GENERATOR_TOP_P", "0"))
    L1_GENERATOR_FREQUENCY_PENALTY: float = float(os.getenv("L1_GENERATOR_FREQUENCY_PENALTY", "0"))
    L1_GENERATOR_PRESENCE_PENALTY: float = float(os.getenv("L1_GENERATOR_PRESENCE_PENALTY", "0"))
    L1_GENERATOR_SEED: int = int(os.getenv("L1_GENERATOR_SEED", "42"))
    L1_GENERATOR_TIMEOUT: int = int(os.getenv("L1_GENERATOR_TIMEOUT", "45"))
    
    # 文档处理配置
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # MCP 集成配置
    MCP_LOCAL_URL: str = os.getenv("MCP_LOCAL_URL", "localhost:8002")
    MCP_LOCAL_PATH: str = os.getenv("MCP_LOCAL_PATH", "/api/kernel2/chat")
    MCP_PUBLIC_URL: str = os.getenv("MCP_PUBLIC_URL", "app.secondme.io")
    MCP_PUBLIC_PATH: str = os.getenv("MCP_PUBLIC_PATH", "/api/chat")
    
    # Twin Types
    TWIN_TYPES: List[str] = ["professional", "expert", "personal", "recruiter"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 创建配置实例
settings = Settings() 