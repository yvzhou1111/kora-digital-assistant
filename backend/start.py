"""
启动脚本，用于启动后端服务
"""
import os
import sys
import uvicorn

# 设置Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 启动服务
if __name__ == "__main__":
    print("正在启动个人数字分身平台后端服务...")
    print("访问API文档: http://localhost:8000/api/docs")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    ) 