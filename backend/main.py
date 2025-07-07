import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging

# 设置Python路径，确保可以导入Second-Me模块
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.append(backend_path)

# 导入核心配置
from app.core.config import settings
from app.api.v1.api import api_router
from app.db.init_db import init_db

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="个人数字分身平台API",
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含API路由
app.include_router(api_router, prefix="/api/v1")

# 创建Second-Me Flask应用实例
try:
    from lpm_kernel.app import app as flask_app
    import flask
    from fastapi.middleware.wsgi import WSGIMiddleware
    
    # 挂载Flask应用到/second-me路径
    app.mount("/second-me", WSGIMiddleware(flask_app))
    logger.info("Successfully mounted Second-Me Flask app at /second-me")
except ImportError as e:
    logger.warning(f"Could not import Second-Me Flask app: {str(e)}")
except Exception as e:
    logger.error(f"Failed to mount Second-Me Flask app: {str(e)}")

# 创建MCP服务
try:
    from mcp.mcp_local import mindv
    from mcp.mcp_public import mindverse
    from fastapi import Request
    from fastapi.responses import JSONResponse
    import json
    
    # 挂载MCP服务到/mcp路径
    @app.post("/mcp/local")
    async def mcp_local_endpoint(request: Request):
        try:
            # 获取请求数据
            data = await request.json()
            query = data.get("query", "")
            session_id = data.get("session_id", "default")
            
            if not query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "缺少查询参数"}
                )
                
            # 调用MCP服务
            response = await mindv.get_response(query=query, session_id=session_id)
            return JSONResponse(content={"response": response})
            
        except Exception as e:
            logger.error(f"MCP Local服务处理请求失败: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"处理请求失败: {str(e)}"}
            )
    
    @app.post("/mcp/public")
    async def mcp_public_endpoint(request: Request):
        try:
            # 获取请求数据
            data = await request.json()
            query = data.get("query", "")
            instance_id = data.get("instance_id", "")
            session_id = data.get("session_id", "default")
            
            if not query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "缺少查询参数"}
                )
                
            if not instance_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "缺少instance_id参数"}
                )
                
            # 调用MCP服务
            response = await mindverse.get_response(
                query=query, 
                instance_id=instance_id,
                session_id=session_id
            )
            return JSONResponse(content={"response": response})
            
        except Exception as e:
            logger.error(f"MCP Public服务处理请求失败: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"处理请求失败: {str(e)}"}
            )
            
    @app.get("/mcp/public/instances")
    async def get_public_instances():
        try:
            # 获取可用实例列表
            instances = await mindverse.get_online_instances()
            return JSONResponse(content={"instances": json.loads(instances)})
            
        except Exception as e:
            logger.error(f"获取MCP Public实例列表失败: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"获取实例列表失败: {str(e)}"}
            )
    
    logger.info("成功设置MCP服务: /mcp/local, /mcp/public, /mcp/public/instances")
except ImportError as e:
    logger.warning(f"无法导入MCP服务: {str(e)}")
except Exception as e:
    logger.error(f"设置MCP服务失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """应用启动时执行的事件"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    logger.info("Database initialized")

@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "status": "ok",
        "message": f"欢迎使用{settings.APP_NAME} API",
        "version": settings.APP_VERSION,
        "docs_url": "/api/docs",
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    ) 