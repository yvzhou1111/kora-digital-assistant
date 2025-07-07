# 后端API设计

## API架构概述

本系统采用RESTful API架构风格，主要特点包括：

- **资源导向**：API围绕资源设计，使用HTTP方法表示操作
- **无状态交互**：每个请求包含所有必要信息
- **标准HTTP方法**：GET(查询)、POST(创建)、PUT(更新)、DELETE(删除)
- **JSON数据格式**：请求和响应均使用JSON
- **版本控制**：API路径包含版本号

## API基础路径

```
https://api.digital-twin-platform.com/v1/
```

## HTTP状态码使用

| 状态码 | 使用场景 |
|-------|---------|
| 200 OK | 成功的GET、PUT、PATCH请求 |
| 201 Created | 成功的POST请求(创建资源) |
| 204 No Content | 成功的DELETE请求 |
| 400 Bad Request | 客户端请求错误 |
| 401 Unauthorized | 身份验证失败 |
| 403 Forbidden | 权限不足 |
| 404 Not Found | 资源不存在 |
| 422 Unprocessable Entity | 请求参数验证失败 |
| 429 Too Many Requests | 请求频率超限 |
| 500 Internal Server Error | 服务器内部错误 |

## 认证与授权

### 认证方式

系统使用Bearer Token认证：

```
Authorization: Bearer {token}
```

### 授权流程

1. 客户端调用登录接口获取访问令牌(JWT)
2. 客户端在后续请求中携带令牌
3. 服务器验证令牌有效性和权限

## 标准响应格式

### 成功响应

```json
{
  "success": true,
  "data": {
    // 实际数据
  },
  "message": "操作成功",
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 100,
      "total_pages": 10
    }
  }
}
```

### 错误响应

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": [
      // 详细错误信息
    ]
  }
}
```

## API端点定义

### 1. 用户认证与授权API

#### 1.1 用户注册

- **端点**: `POST /auth/register`
- **描述**: 创建新用户账号
- **请求体**:

```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "phone": "string",
  "role": "individual|employer|expert",
  "register_source": "direct|google|linkedin|wechat"
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "user_id": "uuid",
    "username": "string",
    "email": "string",
    "role": "string",
    "created_at": "datetime",
    "access_token": "string",
    "token_type": "Bearer",
    "expires_in": 3600
  },
  "message": "注册成功"
}
```

#### 1.2 用户登录

- **端点**: `POST /auth/login`
- **描述**: 用户登录并获取访问令牌
- **请求体**:

```json
{
  "username": "string", // 用户名或邮箱
  "password": "string",
  "remember_me": true
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "user_id": "uuid",
    "username": "string",
    "role": "string",
    "access_token": "string",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "string"
  },
  "message": "登录成功"
}
```

#### 1.3 第三方登录

- **端点**: `POST /auth/oauth/{provider}`
- **描述**: 第三方OAuth登录(Google、LinkedIn、微信等)
- **请求体**:

```json
{
  "access_token": "string", // 第三方平台获取的token
  "user_info": {
    // 第三方返回的用户信息
  }
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "user_id": "uuid",
    "username": "string",
    "is_new_user": true, // 是否新用户
    "access_token": "string",
    "token_type": "Bearer",
    "expires_in": 3600
  },
  "message": "登录成功"
}
```

#### 1.4 刷新令牌

- **端点**: `POST /auth/refresh-token`
- **描述**: 使用刷新令牌获取新的访问令牌
- **请求体**:

```json
{
  "refresh_token": "string"
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "access_token": "string",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "string"
  },
  "message": "令牌刷新成功"
}
```

#### 1.5 退出登录

- **端点**: `POST /auth/logout`
- **描述**: 用户退出登录，使当前令牌失效
- **响应**: 204 No Content

### 2. 用户管理API

#### 2.1 获取当前用户信息

- **端点**: `GET /users/me`
- **描述**: 获取当前登录用户的详细信息
- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "user_id": "uuid",
    "username": "string",
    "email": "string",
    "phone": "string",
    "role": "individual|employer|expert",
    "avatar": "url",
    "created_at": "datetime",
    "updated_at": "datetime",
    "settings": {
      // 用户设置
    }
  },
  "message": "获取用户信息成功"
}
```

#### 2.2 更新用户信息

- **端点**: `PUT /users/me`
- **描述**: 更新当前用户的个人信息
- **请求体**:

```json
{
  "username": "string",
  "email": "string",
  "phone": "string",
  "avatar": "file",
  "settings": {
    // 用户设置
  }
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "user_id": "uuid",
    "username": "string",
    "email": "string",
    "phone": "string",
    "avatar": "url",
    "updated_at": "datetime"
  },
  "message": "用户信息更新成功"
}
```

#### 2.3 更改密码

- **端点**: `PUT /users/me/password`
- **描述**: 更改当前用户密码
- **请求体**:

```json
{
  "current_password": "string",
  "new_password": "string",
  "confirm_password": "string"
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "message": "密码修改成功"
}
```

### 3. 数字分身API

#### 3.1 创建数字分身

- **端点**: `POST /digital-twins`
- **描述**: 为当前用户创建数字分身
- **请求体**:

```json
{
  "name": "string",
  "description": "string",
  "avatar": "file",
  "communication_style": "professional|friendly|casual"
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "twin_id": "uuid",
    "name": "string",
    "description": "string",
    "avatar_url": "url",
    "communication_style": "string",
    "created_at": "datetime",
    "status": "initializing"
  },
  "message": "数字分身创建成功"
}
```

#### 3.2 获取数字分身详情

- **端点**: `GET /digital-twins/{twin_id}`
- **描述**: 获取指定数字分身的详细信息
- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "twin_id": "uuid",
    "name": "string",
    "description": "string",
    "avatar_url": "url",
    "communication_style": "string",
    "created_at": "datetime",
    "updated_at": "datetime",
    "status": "active|training|paused",
    "data_sources": [
      {
        "source_id": "uuid",
        "source_type": "resume|social_media|manual",
        "name": "string",
        "status": "processed|pending|failed"
      }
    ],
    "stats": {
      "training_progress": 80, // percentage
      "knowledge_coverage": 75, // percentage
      "total_conversations": 42
    }
  },
  "message": "获取数字分身成功"
}
```

#### 3.3 更新数字分身

- **端点**: `PUT /digital-twins/{twin_id}`
- **描述**: 更新数字分身的基本信息
- **请求体**:

```json
{
  "name": "string",
  "description": "string",
  "avatar": "file",
  "communication_style": "professional|friendly|casual"
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "twin_id": "uuid",
    "name": "string",
    "description": "string",
    "avatar_url": "url",
    "communication_style": "string",
    "updated_at": "datetime"
  },
  "message": "数字分身更新成功"
}
```

#### 3.4 删除数字分身

- **端点**: `DELETE /digital-twins/{twin_id}`
- **描述**: 删除指定的数字分身
- **响应**: 204 No Content

#### 3.5 上传数字分身数据

- **端点**: `POST /digital-twins/{twin_id}/data`
- **描述**: 为数字分身上传各类数据(简历、社交媒体数据等)
- **请求体**:

```json
{
  "data_type": "resume|linkedin|github|blog|manual",
  "content": "file|url|text",
  "metadata": {
    // 附加信息
  }
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "data_id": "uuid",
    "data_type": "string",
    "status": "processing",
    "upload_time": "datetime",
    "estimated_completion": "datetime"
  },
  "message": "数据上传成功，正在处理"
}
```

#### 3.6 与数字分身对话

- **端点**: `POST /digital-twins/{twin_id}/conversations`
- **描述**: 创建与数字分身的新对话
- **请求体**:

```json
{
  "message": "string",
  "context": {
    // 可选的上下文信息
  }
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "conversation_id": "uuid",
    "message": "string",
    "response": "string",
    "timestamp": "datetime"
  },
  "message": "对话成功"
}
```

#### 3.7 优化数字分身回答

- **端点**: `PUT /digital-twins/{twin_id}/responses/{response_id}`
- **描述**: 对数字分身的特定回答进行纠正和优化
- **请求体**:

```json
{
  "correct_response": "string",
  "feedback": "string"
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "response_id": "uuid",
    "original_response": "string",
    "corrected_response": "string",
    "updated_at": "datetime"
  },
  "message": "回答已更新，分身将优化此类问题的响应"
}
```

### 4. 职位管理API（雇主视角）

#### 4.1 创建职位

- **端点**: `POST /employer/jobs`
- **描述**: 雇主创建新职位
- **请求体**:

```json
{
  "title": "string",
  "description": "string",
  "requirements": ["string"],
  "responsibilities": ["string"],
  "location": "string",
  "salary_range": {
    "min": 0,
    "max": 0,
    "currency": "string"
  },
  "employment_type": "full_time|part_time|contract",
  "experience_level": "entry|mid|senior",
  "deadline": "date"
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "job_id": "uuid",
    "title": "string",
    "status": "draft|published|closed",
    "created_at": "datetime"
  },
  "message": "职位创建成功"
}
```

#### 4.2 获取雇主发布的所有职位

- **端点**: `GET /employer/jobs`
- **描述**: 获取当前雇主发布的所有职位列表
- **查询参数**:
  - `status`: 职位状态筛选
  - `page`: 页码
  - `per_page`: 每页条数
- **响应**: 200 OK

```json
{
  "success": true,
  "data": [
    {
      "job_id": "uuid",
      "title": "string",
      "location": "string",
      "status": "string",
      "created_at": "datetime",
      "applications_count": 0
    }
  ],
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 100,
      "total_pages": 10
    }
  },
  "message": "获取职位列表成功"
}
```

#### 4.3 获取候选人列表

- **端点**: `GET /employer/jobs/{job_id}/candidates`
- **描述**: 获取特定职位的所有候选人列表
- **查询参数**:
  - `status`: 申请状态筛选
  - `sort_by`: 排序字段
  - `page`: 页码
  - `per_page`: 每页条数
- **响应**: 200 OK

```json
{
  "success": true,
  "data": [
    {
      "application_id": "uuid",
      "candidate_id": "uuid",
      "name": "string",
      "match_score": 85,
      "status": "applied|reviewing|interviewed|offered|rejected",
      "applied_at": "datetime",
      "has_digital_twin": true
    }
  ],
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 50,
      "total_pages": 5
    }
  },
  "message": "获取候选人列表成功"
}
```

#### 4.4 与候选人数字分身对话

- **端点**: `POST /employer/candidates/{candidate_id}/twin/conversations`
- **描述**: 雇主与求职者的数字分身进行对话
- **请求体**:

```json
{
  "message": "string",
  "context": {
    "job_id": "uuid"
  }
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "conversation_id": "uuid",
    "message": "string",
    "response": "string",
    "timestamp": "datetime"
  },
  "message": "对话成功"
}
```

#### 4.5 批量群面数字分身

- **端点**: `POST /employer/jobs/{job_id}/batch-interview`
- **描述**: 向多个候选人的数字分身提出同一个问题
- **请求体**:

```json
{
  "candidate_ids": ["uuid"],
  "question": "string"
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "interview_id": "uuid",
    "job_id": "uuid",
    "question": "string",
    "responses": [
      {
        "candidate_id": "uuid",
        "name": "string",
        "response": "string",
        "response_time": "datetime"
      }
    ]
  },
  "message": "批量群面已发起"
}
```

### 5. 求职管理API（个人用户视角）

#### 5.1 获取推荐职位列表

- **端点**: `GET /individual/job-recommendations`
- **描述**: 获取基于用户数字分身自动推荐的职位列表
- **查询参数**:
  - `page`: 页码
  - `per_page`: 每页条数
- **响应**: 200 OK

```json
{
  "success": true,
  "data": [
    {
      "job_id": "uuid",
      "title": "string",
      "company": "string",
      "location": "string",
      "match_score": 92,
      "match_reasons": ["string"],
      "posted_at": "datetime"
    }
  ],
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 50,
      "total_pages": 5
    }
  },
  "message": "获取推荐职位成功"
}
```

#### 5.2 投递职位申请

- **端点**: `POST /individual/jobs/{job_id}/apply`
- **描述**: 向特定职位提交申请
- **请求体**:

```json
{
  "cover_letter": "string",
  "use_digital_twin": true,
  "resume_id": "uuid" // 可选，特定简历ID
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "application_id": "uuid",
    "job_id": "uuid",
    "job_title": "string",
    "company": "string",
    "status": "applied",
    "applied_at": "datetime"
  },
  "message": "职位申请已提交"
}
```

#### 5.3 获取模拟面试

- **端点**: `POST /individual/mock-interviews`
- **描述**: 创建模拟面试会话
- **请求体**:

```json
{
  "job_id": "uuid", // 可选，基于特定职位
  "position_type": "string", // 可选，职位类型
  "difficulty": "easy|medium|hard"
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "interview_id": "uuid",
    "questions": [
      {
        "question_id": "uuid",
        "content": "string",
        "type": "technical|behavioral|experience"
      }
    ],
    "expires_at": "datetime"
  },
  "message": "模拟面试已创建"
}
```

#### 5.4 提交面试答案

- **端点**: `POST /individual/mock-interviews/{interview_id}/answers`
- **描述**: 提交模拟面试答案
- **请求体**:

```json
{
  "question_id": "uuid",
  "answer": "string"
}
```

- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "question_id": "uuid",
    "feedback": {
      "score": 85,
      "strengths": ["string"],
      "weaknesses": ["string"],
      "improvement_suggestions": ["string"]
    },
    "next_question": {
      "question_id": "uuid",
      "content": "string",
      "type": "string"
    }
  },
  "message": "答案已评估"
}
```

### 6. 专家服务API

#### 6.1 创建专家知识库

- **端点**: `POST /expert/knowledge-bases`
- **描述**: 创建专家知识库
- **请求体**:

```json
{
  "name": "string",
  "description": "string",
  "expertise_areas": ["string"],
  "public_access": true
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "knowledge_base_id": "uuid",
    "name": "string",
    "status": "created",
    "created_at": "datetime"
  },
  "message": "知识库创建成功"
}
```

#### 6.2 上传专业内容

- **端点**: `POST /expert/knowledge-bases/{knowledge_base_id}/contents`
- **描述**: 向知识库上传专业内容
- **请求体**:

```json
{
  "title": "string",
  "content_type": "paper|article|book|slides|faq",
  "file": "file",
  "description": "string",
  "tags": ["string"]
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "content_id": "uuid",
    "title": "string",
    "content_type": "string",
    "status": "processing",
    "upload_time": "datetime"
  },
  "message": "内容上传成功，正在处理"
}
```

#### 6.3 创建咨询服务

- **端点**: `POST /expert/consultation-services`
- **描述**: 创建专家咨询服务
- **请求体**:

```json
{
  "title": "string",
  "description": "string",
  "service_type": "qa|consultation|project",
  "pricing": {
    "model": "free|per_question|hourly|subscription",
    "amount": 0,
    "currency": "string"
  },
  "availability": {
    "days": ["monday", "tuesday"],
    "hours": ["09:00-12:00", "14:00-18:00"]
  }
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "service_id": "uuid",
    "title": "string",
    "status": "active",
    "created_at": "datetime"
  },
  "message": "咨询服务创建成功"
}
```

#### 6.4 获取影响力分析

- **端点**: `GET /expert/analytics`
- **描述**: 获取专家影响力统计和分析
- **查询参数**:
  - `period`: 时间范围
  - `metrics`: 统计指标
- **响应**: 200 OK

```json
{
  "success": true,
  "data": {
    "profile_views": {
      "total": 1250,
      "trend": "+15%",
      "history": [
        {
          "date": "date",
          "count": 42
        }
      ]
    },
    "queries": {
      "total": 568,
      "unique_users": 234,
      "top_topics": [
        {
          "topic": "string",
          "count": 75
        }
      ]
    },
    "paid_consultations": {
      "total": 28,
      "revenue": 1400,
      "currency": "USD"
    }
  },
  "message": "获取分析数据成功"
}
```

### 7. 通用消息API

#### 7.1 获取消息列表

- **端点**: `GET /messages`
- **描述**: 获取用户的消息列表
- **查询参数**:
  - `type`: 消息类型
  - `status`: 消息状态
  - `page`: 页码
  - `per_page`: 每页条数
- **响应**: 200 OK

```json
{
  "success": true,
  "data": [
    {
      "message_id": "uuid",
      "sender": {
        "id": "uuid",
        "name": "string",
        "avatar": "url",
        "type": "user|system|twin"
      },
      "content": "string",
      "created_at": "datetime",
      "read": true,
      "message_type": "chat|notification|alert"
    }
  ],
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 100,
      "total_pages": 10
    },
    "unread_count": 5
  },
  "message": "获取消息列表成功"
}
```

#### 7.2 发送消息

- **端点**: `POST /messages`
- **描述**: 发送消息给特定用户
- **请求体**:

```json
{
  "recipient_id": "uuid",
  "content": "string",
  "message_type": "chat|consultation|job_related"
}
```

- **响应**: 201 Created

```json
{
  "success": true,
  "data": {
    "message_id": "uuid",
    "sent_at": "datetime",
    "delivered": true
  },
  "message": "消息发送成功"
}
```

#### 7.3 标记消息为已读

- **端点**: `PUT /messages/{message_id}/read`
- **描述**: 将特定消息标记为已读
- **响应**: 200 OK

```json
{
  "success": true,
  "message": "消息已标记为已读"
}
```

## API文档和测试

- 使用OpenAPI(Swagger)自动生成API文档
- 提供Postman集合用于API测试
- 提供示例请求和响应 