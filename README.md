# Research Copilot MVP

一个本地运行的 Research Copilot MVP：支持模型配置、知识库管理、TXT/PDF 上传、启用知识库检索、基于 LangGraph 的 RAG 问答和历史会话。

## 启动

```powershell
pip install -r requirements.txt
python main.py
```

如果当前机器没有把 Python 放进 PATH，可以用 Codex 自带 Python 运行：

```powershell
& "C:\Users\zyc2025\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r requirements.txt
& "C:\Users\zyc2025\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" main.py
```

启动后访问：

```text
http://127.0.0.1:8000
```

## 使用顺序

1. 点击右下角模型管理，填写 `base_url`、`api_key`、`model`，例如 `https://api.deepseek.com`。
2. 点击左上角知识库管理，新建知识库。Embedding 可选择本地 `local-hash`，也可以填写 OpenAI-compatible embedding 接口。
3. 进入知识库详情，上传 TXT 或 PDF。
4. 保持知识库启用，在中间对话框提问。

## 数据位置

所有数据保存在本地 `data/` 目录：

- `data/app.db`：SQLite 数据库
- `data/files/`：上传的原始文档

## 说明

MVP 默认提供 `local-hash` embedding，方便无额外 embedding API 时也能跑通。生产或高质量检索建议创建知识库时配置真实 embedding 模型；知识库创建后 embedding 配置会被锁定。
