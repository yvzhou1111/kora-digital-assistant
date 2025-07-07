# 个人数字分身平台 API 测试

这个目录包含了用于测试个人数字分身平台后端 API 的工具和脚本。

## 环境设置

1. 创建并激活虚拟环境:
   ```
   python -m venv backend_env
   .\backend_env\Scripts\activate
   ```

2. 安装依赖:
   ```
   pip install -r ../backend/requirements.txt
   ```

3. 配置环境变量 (.env 文件):
   ```
   # Database settings
   DATABASE_URL=sqlite:///./test.db
   
   # Security settings
   SECRET_KEY=testsecretkey123456789
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   
   # AI Model settings
   AI_MODEL_NAME=gpt-3.5-turbo
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Vector DB settings
   VECTOR_DB_PATH=./vector_db
   
   # Testing mode
   TESTING=true
   ```

## 测试文件

1. **app.py**: 简化版的 FastAPI 应用，用于测试 API 端点
2. **test_app.py**: 使用 pytest 和 TestClient 测试 API 端点
3. **api_client.py**: 使用 requests 库测试 API 端点

## 运行测试

1. 运行单元测试:
   ```
   python test_app.py
   ```

2. 启动 API 服务器:
   ```
   uvicorn app:app --reload
   ```

3. 使用 API 客户端测试端点:
   ```
   python api_client.py
   ```

## API 端点

- **GET /**: 根端点，返回 API 状态和版本信息
- **GET /api/v1/health**: 健康检查端点
- **POST /api/v1/auth/token**: 认证端点，用于获取访问令牌
- **GET /api/v1/users/me**: 获取当前用户信息
- **GET /api/v1/digital-twins**: 获取用户的数字分身列表

## 测试结果

所有测试均已通过，API 端点按预期工作。

- 根端点返回正确的状态和版本信息
- 健康检查端点返回状态 "ok"
- 认证端点能够正确处理有效和无效的凭据
- 用户信息端点能够返回当前用户的详细信息
- 数字分身端点能够返回用户的数字分身列表 