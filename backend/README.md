# 个人数字分身平台后端服务

本项目是个人数字分身平台的后端服务，集成了Second-Me项目的功能，实现了数字分身的创建、训练和交互，为前端提供完整的API支持。

## 项目概述

个人数字分身平台致力于赋能个体，连接未来。通过为专业人士打造智能的"数字分身"，让其知识和经验能够7x24小时不间断地与世界沟通、创造价值。同时，为用人方提供前所未有的高效人才筛选和连接工具，彻底改变传统招聘和咨询的模式。

核心功能包括：
- 数字分身创建与养成
- 智能求职与匹配
- 能力可视化
- 知识咨询与变现

## 项目结构

```
backend/
├── app/                  # 主应用目录
│   ├── ai/               # AI引擎相关
│   │   ├── doubao_client.py  # 豆包API客户端
│   │   └── twin_engine.py    # 数字分身引擎
│   ├── api/              # API接口
│   ├── core/             # 核心配置
│   ├── db/               # 数据库模型
│   ├── models/           # 数据模型
│   ├── schemas/          # Pydantic模式
│   ├── services/         # 业务服务
│   └── utils/            # 工具函数
├── dependencies/         # 项目依赖
├── lpm_kernel/           # Second-Me核心组件
├── mcp/                  # MCP服务
│   ├── mcp_local.py      # 本地MCP服务
│   └── mcp_public.py     # 公共MCP服务
├── main.py               # 主入口文件
├── start.py              # 启动脚本
├── requirements.txt      # 依赖列表
└── 环境变量.md            # 环境变量说明
```

## 技术栈

- **FastAPI**: Web框架，提供高性能API接口
- **SQLAlchemy**: ORM，数据库交互
- **PostgreSQL**: 关系型数据库，存储用户和分身数据
- **ChromaDB**: 向量数据库，支持语义检索
- **OpenAI/豆包API**: 大模型服务，提供AI能力
- **Second-Me**: 数字分身引擎，提供L1生成器和MCP服务

## 环境配置

1. 创建并激活虚拟环境

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

2. 安装依赖

```bash
# Linux/macOS
pip install -r requirements.txt

# Windows (解决编码问题)
pip install -r requirements-windows.txt
```

3. 创建`.env`文件，参考`环境变量.md`进行配置

## 项目启动

使用启动脚本运行服务:

```bash
python start.py
```

或者直接使用uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Second-Me集成详解

本项目深度集成了Second-Me的核心组件，实现了从个人数据到数字分身的完整流程：

### 1. L1 Generator集成

`app/ai/twin_engine.py`中集成了Second-Me的L1Generator组件，实现：

- 数据源的处理和分块
- 向量嵌入和语义索引
- 自动生成个性化Bio（生物信息）
- 特征识别和归类（Shades）
- 集群分析和主题生成

示例代码：
```python
# L1Generator初始化
self.l1_generator = L1Generator()
self.l1_generator.preferred_language = self.preferred_language
self.l1_generator.bio_model_params = {
    "temperature": settings.L1_GENERATOR_TEMPERATURE,
    "max_tokens": settings.L1_GENERATOR_MAX_TOKENS,
    # 其他参数...
}

# 生成数字分身Bio
bio = self.l1_generator.gen_global_biography(old_bio, l1_clusters)
```

### 2. MCP服务集成

`main.py`中集成了MCP服务，提供与本地和公共Second-Me模型的交互：

- `/mcp/local`: 与本地Second-Me模型交互
- `/mcp/public`: 与公共Second-Me模型交互
- `/mcp/public/instances`: 获取可用的公共模型列表

这些服务支持会话管理、流式响应和错误处理。

### 3. Flask应用挂载

将Second-Me的Flask应用挂载到FastAPI中：

```python
# 创建Second-Me Flask应用实例
from lpm_kernel.app import app as flask_app
from fastapi.middleware.wsgi import WSGIMiddleware

# 挂载Flask应用到/second-me路径
app.mount("/second-me", WSGIMiddleware(flask_app))
```

## API文档

启动服务后，访问以下URL查看API文档:

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 主要端点

- `/api/v1/twins`: 数字分身管理
- `/api/v1/conversations`: 对话管理
- `/api/v1/datasources`: 数据源管理
- `/second-me`: Second-Me Web界面
- `/mcp/local`: 本地MCP服务
- `/mcp/public`: 公共MCP服务
- `/mcp/public/instances`: 公共实例列表

## 项目目标符合性

本后端实现满足"最后目标项目.md"中定义的需求：

1. **数字分身创建与养成**：通过twin_engine.py实现数据处理、分身生成
2. **一站式信息聚合**：支持多种数据源的处理和分析
3. **分身"试驾"与调优**：通过MCP服务提供即时对话功能
4. **智能求职中心**：实现能力模型和职位匹配
5. **能力可视化**：通过集群和特征分析提供能力洞察

## 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request 