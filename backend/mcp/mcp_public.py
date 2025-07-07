from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
import http.client
import json
import requests
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 创建MCP实例
mindverse = FastMCP("mindverse_public")

# 配置参数
url = "app.secondme.io"

# 维护会话上下文
conversation_history = {}

@mindverse.tool()
async def get_response(query: str, instance_id: str, session_id: str = "default") -> str | None:
    """
    获取基于公共Second-Me模型的响应
    
    参数:
        query (str): 用户针对secondme模型提出的问题
        instance_id (str): 用于标识secondme模型的ID或URL
        session_id (str): 会话ID，用于维护对话上下文
        
    返回:
        str: 模型生成的回复
    """
    # 提取ID
    id = instance_id.split('/')[-1]
    path = f"/api/chat/{id}"
    
    # 初始化会话历史
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # 添加用户消息到历史
    conversation_history[session_id].append({"role": "user", "content": query})
    
    # 构建请求头
    headers = {"Content-Type": "application/json"}
    
    # 构建请求体
    data = {
        "messages": conversation_history[session_id],
        "metadata": {
            "enable_l0_retrieval": False,
            "role_id": "default_role"
        },
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": True
    }
    
    try:
        # 创建HTTPS连接
        conn = http.client.HTTPSConnection(url)
        
        # 发送POST请求
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        
        # 获取响应
        response = conn.getresponse()
        
        # 检查响应状态
        if response.status != 200:
            logger.error(f"API请求失败，状态码: {response.status}")
            return f"请求失败，状态码: {response.status}"
        
        full_content = ""
        
        # 处理流式响应
        for line in response:
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line == 'data: [DONE]':
                    break
                if decoded_line.startswith('data: '):
                    try:
                        json_str = decoded_line[6:]
                        chunk = json.loads(json_str)
                        content = chunk['choices'][0]['delta'].get('content', '')
                        if content:
                            full_content += content
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {str(e)}")
                    except KeyError as e:
                        logger.error(f"响应格式错误: {str(e)}")
        
        # 关闭连接
        conn.close()
        
        # 处理响应结果
        if full_content:
            # 添加助手回复到历史
            conversation_history[session_id].append({"role": "assistant", "content": full_content})
            return full_content
        else:
            logger.warning("从模型获取的响应为空")
            return "抱歉，我无法生成回复。请稍后再试。"
            
    except Exception as e:
        logger.error(f"调用Public Second-Me API时出错: {str(e)}")
        return f"发生错误: {str(e)}"


@mindverse.tool()
async def get_online_instances():
    """
    获取可用于在线聊天的secondme模型列表
    
    返回:
        str: 可用模型的JSON字符串
    """
    try:
        api_url = "https://app.secondme.io/api/upload/list?page_size=100"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("data", {}).get("items", [])
            
            online_items = [
                {
                    "upload_name": item["upload_name"],
                    "instance_id": item["instance_id"],
                    "description": item.get("description", ""),
                    "created_at": item.get("created_at", ""),
                    "avatar_url": item.get("avatar_url", "")
                }
                for item in items if item.get("status") == "online"
            ]
            
            return json.dumps(online_items, ensure_ascii=False, indent=2)
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
            return json.dumps({"error": f"请求失败，状态码: {response.status_code}"})
            
    except Exception as e:
        logger.error(f"获取在线实例时出错: {str(e)}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # 初始化并运行服务器
    mindverse.run(transport='stdio')



