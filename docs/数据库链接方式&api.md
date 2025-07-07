### 数据库链接都用本地链接（下面是正确的使用方式，都经过测试）

1.
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="postgres",
        user="postgres",
        password="yili142857"
    )
    print("PostgreSQL连接成功！")
    conn.close()
except Exception as e:
    print(f"连接失败：{e}")

2.
import redis

try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('test_key', 'Hello Redis!')
    print(r.get('test_key'))  # 输出：b'Hello Redis!'
except Exception as e:
    print(f"连接失败：{e}")


3.
chroma数据库在需要的虚拟环境中本地创建使用


4.AI api 使用方法如下  包括fanction_call功能的使用实例

豆包大模型Rest API 调用示例

curl https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 7efdde88-d1b7-4ca3-add9-e9e01be8e012" \
  -d $'{
    "model": "doubao-seed-1-6-250615",
    "messages": [
        {
            "content": [
                {
                    "text": "图片主要讲了什么?",
                    "type": "text"
                },
                {
                    "image_url": {
                        "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
                    },
                    "type": "image_url"
                }
            ],
            "role": "user"
        }
    ]
}'




网页解析插件功能说明
最近更新时间：2025.06.18 11:42:13
首次发布时间：2024.06.27 16:17:37



一次调用最多可解析 3 个 url 内容。
支持网页、pdf、txt、csv、docx、doc、xlsx、xls、pptx、ppt、md、mobi、epub格式文件的内容获取。
当前支持接入 Doubao 系列拥有Function calling能力的模型，详见支持模型。
1QPS
60RPM

限时免费；使用插件可能产生模型更多 tokens 消耗

注：当前网页解析插件暂不支持与其他插件同时开启使用。


功能优势
精准解析网页： 支持通过url 快速、准确地解析网页、pdf、txt、csv、docx、doc、xlsx、xls、pptx、ppt、md、mobi、epub格式文件的内容，可接入模型进行高效使用。
高效操作流程：用户仅需提供url链接，即可完成网页的完整解析，无需繁琐复杂的操作步骤。

应用场景
可高效满足广泛的业务需求，无论是内容聚合、市场分析还是自动化报告生成，本插件都能提供强大支持，接入大模型可助力业务决策和信息整合

市场分析： 可用于解析网站信息，帮助企业进行市场趋势调研和竞品分析，为市场定位和策略制定提供支持
新闻聚合： 整合来自不同内容源的最新新闻信息，形成信息概览
学习教育： 获取教育资料和学术文章，支持学习教育场景下的高效学习和研究

基本用法

通过零代码应用调用
您可在控制台的直接创建带网页解析插件的应用，无需自己实现插件调用代码。详情请查看 零代码应用操作指南。
后续您可通过应用(Bot) API 调用已添加网页解析插件的零代码应用，详情请查看 Bot API 文档 与 Bot SDK 文档。

自行实现插件调用
您也可以自行大模型调用插件的代码，下面是示例代码。


Function Calling调用




适用于支持Function Calling的模型，下面是示例代码。

需安装arkitect 0.1.x版本（同时要求python版本小于3.12.0），pip install 'arkitect<0.2.0' --index-url https://pypi.org/simple

import os
from typing import AsyncIterable, Union

from arkitect.core.component.llm import BaseChatLanguageModel

from arkitect.core.component.llm.model import (
    ArkChatCompletionChunk,
    ArkChatParameters,
    ArkChatRequest,
    ArkChatResponse,
    Response,
)
from arkitect.core.component.tool import LinkReader, ToolPool
from arkitect.launcher.local.serve import launch_serve
from arkitect.telemetry.trace import task

endpoint_id = "<YOUR ENDPOINT ID>"

@task()
async def default_model_calling(
    request: ArkChatRequest,
) -> AsyncIterable[Union[ArkChatCompletionChunk, ArkChatResponse]]:
    parameters = ArkChatParameters(**request.__dict__)
    ToolPool.register(LinkReader())

    llm = BaseChatLanguageModel(
        endpoint_id=endpoint_id,
        messages=request.messages,
        parameters=parameters,
    )
    if request.stream:
        async for resp in llm.astream(functions=ToolPool.all()):
            yield resp
    else:
        yield await llm.arun(functions=ToolPool.all())


@task()
async def main(request: ArkChatRequest) -> AsyncIterable[Response]:
    async for resp in default_model_calling(request):
        yield resp


if __name__ == "__main__":
    port = os.getenv("_FAAS_RUNTIME_PORT")
    launch_serve(
        package_path="main",
        port=int(port) if port else 8080,
        health_check_path="/v1/ping",
        endpoint_path="/api/v3/bots/chat/completions",
        clients={},
    )
启动服务

# # 替换为您的方舟API Key https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey?apikey=%7B%7D
export ARK_API_KEY=<YOUR APIKEY>
python3 main.py
调用方式

curl --location 'http://localhost:8080/api/v3/bots/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "",
    "messages": [
        {
            "role": "user",
            "content": "https://www.toutiao.com/ 这个网站讲了什么"
        }
    ]
}'
返回结果预览

{
    "error": null,
    "id": "0217***",
    "choices": [
        {
            "finish_reason": "stop",
            "moderation_hit_type": null,
            "index": 0,
            "logprobs": null,
            "message": {
                "content": "\n用户想获取网站内容，调用LinkReader/LinkReader函数获取信息。这是今日头条的网页信息，包含了多个热点事件和人物故事的讨论：\n- **政策与建议**：全国人大代表李镇西建议将寒假设为法定节假日；\n- **财经与商业**：多家企业发布动态，如巴菲特旗下伯克希尔·哈撒韦公司市值蒸发、苹果股价创新高、欧洲央行降息等；《哪吒2》票房突破146亿元。\n- **科技与航空**：直捣“星链”第八次试飞现场，“悟空号”成功再现，但二级火箭失联；马斯克欲发射飞船接人，却遭美国宇航局拒绝。\n- **人物故事**：介绍了超级富豪马斯克，虽身价高昂，但他自称像普通打工仔，成长经历坎坷，母亲梅耶·马斯克给予其重要支持；梅耶·马斯克一生传奇，48岁重新学习考试，60岁回归T台，69岁登上美国时代广场，她的自传《人生由我》鼓励人们积极面对生活。 ",
                "role": "assistant",
                "function_call": null,
                "tool_calls": null,
                "audio": null,
                "reasoning_content": null
            }
        }
    ],
    "created": 1741840746,
    "model": "doubao-seed-1-6-250615",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 340,
        "prompt_tokens": 13116,
        "total_tokens": 13456,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "bot_usage": null,
    "metadata": null
}  
