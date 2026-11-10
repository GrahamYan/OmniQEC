# OpenRouter 快速使用指南

OpenRouter 的 API 地址：

```text
https://openrouter.ai/api/v1
```

## 1. 配置 API key

在以下页面创建 key：

```text
https://openrouter.ai/settings/keys
```

在终端中配置：

```bash
export OPENROUTER_API_KEY="sk-or-v1-替换成你的key"
```

不要把真实 key 写进源码或提交到 Git。

## 2. 查看和选择模型

### 网页查看

打开模型目录：

```text
https://openrouter.ai/models
```

网页中可以查看模型名称、价格、上下文长度和支持的功能。

### 命令行查看

列出所有模型：

```bash
curl -s https://openrouter.ai/api/v1/models \
  | jq -r '.data[] | [.id, .context_length, .name] | @tsv'
```

搜索 DeepSeek 模型：

```bash
curl -s https://openrouter.ai/api/v1/models \
  | jq -r '.data[].id' \
  | grep -i deepseek
```

查看某个模型的完整信息：

```bash
MODEL="deepseek/deepseek-chat"

curl -s https://openrouter.ai/api/v1/models \
  | jq --arg model "$MODEL" '.data[] | select(.id == $model)'
```

调用时使用模型的 `id`，例如：

```bash
export OPENROUTER_MODEL="deepseek/deepseek-chat"
```

模型和价格可能变化，正式使用前建议重新查询模型目录。

## 3. 调用模型

### cURL

```bash
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model "${OPENROUTER_MODEL:-deepseek/deepseek-chat}" \
    '{
      model: $model,
      messages: [
        {role: "user", content: "Hello, introduce yourself briefly."}
      ]
    }')" \
  | jq -r '.choices[0].message.content'
```

### Python Requests

安装：

```bash
python -m pip install requests
```

调用：

```python
import os
import requests

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    },
    json={
        "model": os.getenv(
            "OPENROUTER_MODEL",
            "deepseek/deepseek-chat",
        ),
        "messages": [
            {
                "role": "user",
                "content": "Hello, introduce yourself briefly.",
            }
        ],
    },
    timeout=60,
)

response.raise_for_status()
data = response.json()

print(data["choices"][0]["message"]["content"])
print("实际模型：", data.get("model"))
print("用量：", data.get("usage"))
```

### OpenAI Python SDK

安装：

```bash
python -m pip install --upgrade openai
```

调用：

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model=os.getenv(
        "OPENROUTER_MODEL",
        "deepseek/deepseek-chat",
    ),
    messages=[
        {
            "role": "user",
            "content": "Hello, introduce yourself briefly.",
        }
    ],
)

print(response.choices[0].message.content)
print("实际模型：", response.model)
print("用量：", response.usage)
```

官方文档：

- [Quickstart](https://openrouter.ai/docs/quickstart)
- [模型目录](https://openrouter.ai/models)
- [Models API](https://openrouter.ai/docs/api/api-reference/models/get-models)

