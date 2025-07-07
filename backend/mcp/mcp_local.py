from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
import http.client
import json
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 创建MCP实例
mindv = FastMCP("mindverse")

# 配置参数
url = "localhost:8002"
path = "/api/kernel2/chat"

# 维护会话上下文
conversation_history = {}

@mindv.tool()
async def get_response(query: str, session_id: str = "default") -> str | None | Any:
    """
    获取基于本地Second-Me模型的响应
    
    参数:
        query (str): 用户针对secondme模型提出的问题
        session_id (str): 会话ID，用于维护对话上下文
        
    返回:
        str: 模型生成的回复
    """
    # 初始化会话历史
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # 添加用户消息到历史
    conversation_history[session_id].append({"role": "user", "content": query})
    
    # 构建请求头
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    # 构建请求体
    data = {
        "messages": conversation_history[session_id],
        "stream": True
    }
    
    try:
        # 创建HTTP连接
        conn = http.client.HTTPConnection(url)
        
        # 发送POST请求
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        
        # 获取响应
        response = conn.getresponse()
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
        logger.error(f"调用Second-Me API时出错: {str(e)}")
        return f"发生错误: {str(e)}"


if __name__ == "__main__":
    # 初始化并运行服务器
    mindv.run(transport='stdio')
