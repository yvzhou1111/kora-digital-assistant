from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, digital_twins, jobs, experts, messages

api_router = APIRouter()

# 认证路由
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])

# 用户路由
api_router.include_router(users.router, prefix="/users", tags=["用户"])

# 数字分身路由
api_router.include_router(digital_twins.router, prefix="/digital-twins", tags=["数字分身"])

# 职位路由
api_router.include_router(jobs.router, prefix="/jobs", tags=["职位"])

# 专家路由
api_router.include_router(experts.router, prefix="/experts", tags=["专家"])

# 消息路由
api_router.include_router(messages.router, prefix="/messages", tags=["消息"]) 