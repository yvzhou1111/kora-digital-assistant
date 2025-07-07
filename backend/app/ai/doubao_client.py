"""
豆包API客户端模块，提供与豆包大模型API的交互功能
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import httpx
from httpx import Response
from pydantic import BaseModel

from app.core.config import settings

# 设置日志
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str  # 'system', 'user', 'assistant'
    content: str


class ChatCompletionResponse(BaseModel):
    """聊天补全响应模型"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class DoubaoClient:
    """豆包API客户端类"""
    
    def __init__(self):
        """初始化豆包API客户端"""
        self.api_key = os.getenv("DOUBAO_API_KEY") or settings.OPENAI_API_KEY
        self.base_url = os.getenv("DOUBAO_API_BASE") or settings.OPENAI_API_BASE
        self.model = os.getenv("DOUBAO_MODEL") or settings.LLM_MODEL
        self.timeout = 60  # 默认超时时间
        
        # 创建HTTP客户端
        self.client = httpx.Client(timeout=self.timeout)
        
        # 设置请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        调用豆包聊天补全API
        
        参数:
            messages: 消息列表，每个消息包含role和content
            model: 模型名称，如不指定则使用默认模型
            temperature: 温度参数，控制随机性，0-1之间
            max_tokens: 最大生成令牌数
            stream: 是否使用流式响应
            
        返回:
            API响应内容
        """
        url = f"{self.base_url}/chat/completions"
        
        # 构建请求体
        data = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            # 发送请求
            response = self.client.post(
                url,
                headers=self.headers,
                json=data
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return result
            
        except Exception as e:
            logger.error(f"豆包API调用失败: {str(e)}")
            # 返回错误信息
            return {
                "error": {
                    "message": str(e),
                    "type": "api_error"
                }
            }
    
    def parse_html(self, html_content: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        解析HTML内容
        
        参数:
            html_content: HTML内容字符串
            query: 可选的查询字符串，用于指导解析
            
        返回:
            解析结果
        """
        # 构建系统提示
        system_prompt = "你是一个网页内容解析助手，专长于从HTML中提取结构化信息。请分析以下HTML内容。"
        
        # 构建用户提示
        user_prompt = f"请解析以下HTML内容，提取其中的主要信息:\n\n{html_content[:10000]}"
        if query:
            user_prompt += f"\n\n特别关注与以下查询相关的信息: {query}"
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用聊天API
        response = self.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        
        # 处理响应
        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        # 提取解析结果
        parsed_content = response["choices"][0]["message"]["content"]
        
        return {
            "success": True,
            "parsed_content": parsed_content
        }
    
    def process_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        处理API响应
        
        参数:
            response: API响应字典
            
        返回:
            生成的文本，如果出错则返回None
        """
        try:
            if "error" in response:
                logger.error(f"API响应错误: {response['error']}")
                return None
                
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0]["message"]
                if "content" in message:
                    return message["content"]
            
            logger.error(f"API响应格式不正确: {response}")
            return None
            
        except Exception as e:
            logger.error(f"处理API响应时出错: {str(e)}")
            return None


# 创建客户端实例
doubao_client = DoubaoClient() 