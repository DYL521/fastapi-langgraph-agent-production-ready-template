# FastAPI LangGraph Agent 模板

[English](README.md) | **简体中文**

一个用于构建 AI Agent 后端的生产就绪模板，基于 FastAPI 与 LangGraph。它替你搞定了那些"难做的部分"——有状态对话、长期记忆、工具调用、可观测性、限流、鉴权——让你专注于自己的 Agent 业务逻辑。

**为 AI 工程师打造**：要的是一个扎实的地基，而不是一个教学玩具项目。

---

## 由 Atlas Cloud 驱动 —— 面向 LangGraph Agent 的即插即用 LLM 后端

<div align="center">
  <a href="https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=fastapi-langgraph-agent-production-ready-template">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/atlas-cloud-logo-dark.png"/>
      <img src="docs/atlas-cloud-logo.png" alt="Atlas Cloud" width="200"/>
    </picture>
  </a>
</div>

[**Atlas Cloud**](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=fastapi-langgraph-agent-production-ready-template) 提供一个 **OpenAI 兼容的 LLM API**，可无缝接入本 FastAPI + LangGraph 模板——无需改动你的 Agent 图逻辑。只需替换 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，即可通过单一统一端点访问 **DeepSeek、Qwen、GLM、Kimi、MiniMax、Gemini、Claude、GPT** 等众多模型。

本模板中的 `LLMRegistry` 使用 `langchain_openai.ChatOpenAI`——Atlas Cloud 与之线缆级兼容，因此你无需触碰任何 LangGraph 逻辑即可即刻使用 59+ 款精选推理模型。

### 快速接入

**第 1 步 —— 获取免费 API Key：** [atlascloud.ai/console/coding-plan](https://www.atlascloud.ai/console/coding-plan)

**第 2 步 —— 更新 `.env.development`：**

```env
OPENAI_API_KEY=<your-atlascloud-key>
OPENAI_BASE_URL=https://api.atlascloud.ai/v1
DEFAULT_LLM_MODEL=deepseek-ai/deepseek-v4-pro
```

**第 3 步 —— 或直接在代码中使用：**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-ai/deepseek-v4-pro",
    openai_api_base="https://api.atlascloud.ai/v1",
    openai_api_key="<your-atlascloud-key>",
    max_tokens=512,  # 推理模型要求 max_tokens >= 512
)
```

它可作为任何使用 `ChatOpenAI` 之处的即插即用替代品——包括 `LLMRegistry`、环形 fallback 服务，以及 mem0 长期记忆。

<details>
<summary>📋 完整模型目录（提供 59 款 LLM）</summary>

| 模型 ID | 提供方 |
|---|---|
| `deepseek-ai/DeepSeek-V3-0324` | DeepSeek |
| `deepseek-ai/deepseek-r1-0528` | DeepSeek |
| `deepseek-ai/DeepSeek-V3.1` | DeepSeek |
| `deepseek-ai/DeepSeek-V3.1-Terminus` | DeepSeek |
| `deepseek-ai/DeepSeek-V3.2-Exp` | DeepSeek |
| `deepseek-ai/deepseek-v3.2` | DeepSeek |
| `qwen/qwen3-32b` | Alibaba Qwen |
| `qwen/qwen3-8b` | Alibaba Qwen |
| `qwen/qwen3-235b-a22b-thinking-2507` | Alibaba Qwen |
| `qwen/qwen3-30b-a3b` | Alibaba Qwen |
| `qwen/qwen3-30b-a3b-thinking-2507` | Alibaba Qwen |
| `Qwen/Qwen3-Coder` | Alibaba Qwen |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Alibaba Qwen |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | Alibaba Qwen |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | Alibaba Qwen |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | Alibaba Qwen |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | Alibaba Qwen |
| `moonshotai/Kimi-K2-Instruct` | Moonshot AI |
| `moonshotai/Kimi-K2-Instruct-0905` | Moonshot AI |
| `moonshotai/Kimi-K2-Thinking` | Moonshot AI |
| `moonshotai/kimi-k2.5` | Moonshot AI |
| `zai-org/GLM-4.6` | Zhipu AI |
| `zai-org/glm-4.7` | Zhipu AI |
| `MiniMaxAI/MiniMax-M2` | MiniMax |
| `minimaxai/minimax-m2.1` | MiniMax |
| `google/gemini-2.5-flash` | Google |
| `google/gemini-2.5-flash-preview-202509` | Google |
| `google/gemini-2.5-flash-lite` | Google |
| `google/gemini-2.5-flash-lite-preview-202509` | Google |
| `google/gemini-2.5-pro` | Google |
| `google/gemini-3-flash-preview` | Google |
| `google/gemini-2.0-flash` | Google |
| `google/gemini-2.0-flash-lite` | Google |
| `openai/gpt-5.1` | OpenAI |
| `openai/gpt-5.1-chat` | OpenAI |
| `openai/gpt-5.1-codex` | OpenAI |
| `openai/gpt-5.1-codex-mini` | OpenAI |
| `openai/gpt-5.1-codex-max` | OpenAI |
| `openai/gpt-4o` | OpenAI |
| `openai/gpt-4o-mini` | OpenAI |
| `openai/gpt-4.1` | OpenAI |
| `openai/gpt-4.1-mini` | OpenAI |
| `openai/gpt-4.1-nano` | OpenAI |
| `openai/o1` | OpenAI |
| `openai/o3` | OpenAI |
| `openai/o3-mini` | OpenAI |
| `openai/o4-mini` | OpenAI |
| `openai/o3-pro` | OpenAI |
| `openai/gpt-5` | OpenAI |
| `openai/gpt-5-chat` | OpenAI |
| `openai/gpt-5-codex` | OpenAI |
| `openai/gpt-5-mini` | OpenAI |
| `openai/gpt-5-nano` | OpenAI |
| `openai/gpt-5-pro` | OpenAI |
| `openai/gpt-5.2` | OpenAI |
| `openai/gpt-5.2-chat` | OpenAI |
| `anthropic/claude-sonnet-4-20250514` | Anthropic |
| `anthropic/claude-haiku-4.5-20251001` | Anthropic |
| `anthropic/claude-sonnet-4.5-20250929` | Anthropic |
| `anthropic/claude-opus-4.1-20250805` | Anthropic |
| `anthropic/claude-opus-4-20250514` | Anthropic |
| `anthropic/claude-opus-4.5-20251101` | Anthropic |

[查看实时模型列表 →](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=fastapi-langgraph-agent-production-ready-template)

</details>

---

## 包含哪些能力

- **LangGraph** 有状态 Agent，支持检查点、工具调用与 human-in-the-loop
- **长期记忆**：mem0 + pgvector，按用户语义检索，带缓存兜底
- **LLM 服务**：环形模型 fallback、指数退避重试、总超时预算
- **Langfuse** 对所有 LLM 调用追踪；Prometheus 指标 + Grafana 仪表盘
- **JWT 鉴权** 与会话管理；通过 slowapi 限流
- **Alembic** 迁移；可选 Valkey/Redis 缓存层
- **结构化日志**：每行都带 request/session/user 上下文

## 快速开始

```bash
git clone <repo-url> my-agent && cd my-agent
cp .env.example .env.development   # 填入你的密钥
make install
make docker-up                     # 启动 API + PostgreSQL
```

打开 [http://localhost:8000/docs](http://localhost:8000/docs) 查看交互式 API。

> 关于不使用 Docker 的本地开发，见 [docs/getting-started.md](docs/getting-started.md)。

## 文档

| 指南 | 涵盖内容 |
|---|---|
| [Getting Started](docs/getting-started.md) | 前置条件、本地搭建、首个 API 调用 |
| [Architecture](docs/architecture.md) | 系统设计、请求流、组件图 |
| [Configuration](docs/configuration.md) | 所有环境变量及默认值 |
| [Authentication](docs/authentication.md) | JWT 流程、会话、端点参考 |
| [Database & Migrations](docs/database.md) | Schema、Alembic 迁移、pgvector |
| [LLM Service](docs/llm-service.md) | 模型、重试、fallback、超时预算 |
| [Memory](docs/memory.md) | mem0 长期记忆、缓存层 |
| [Observability](docs/observability.md) | Langfuse、结构化日志、Prometheus、性能剖析 |
| [Evaluation](docs/evaluation.md) | 评测框架、自定义指标、报告 |
| [Docker](docs/docker.md) | Docker、Compose、完整监控栈 |

## 项目结构

```
src/agent/
  api/v1/          # 路由处理器
  core/
    langgraph/     # Agent 图 + 工具
    prompts/       # 系统提示词模板
    cache.py       # Valkey/Redis + 内存兜底
    config.py      # 配置
    middleware.py  # 指标、日志上下文、性能剖析
    limiter.py     # 限流
  models/          # SQLModel ORM 模型
  schemas/         # Pydantic 请求/响应模型
  services/        # LLM、数据库、记忆服务
alembic/           # 数据库迁移
evals/             # LLM 评测框架
```

## 贡献

欢迎提交 PR。请先阅读 [docs/getting-started.md](docs/getting-started.md) 搭好环境，再遵循 [AGENTS.md](AGENTS.md) 中的代码规范。

安全问题请私下上报——见 [SECURITY.md](SECURITY.md)。

## 许可证

见 [LICENSE](LICENSE)。

## 常见问题

### 通用

**这个模板是什么？**
一个基于 FastAPI + LangGraph 的生产就绪 AI Agent 后端地基。它打包了那些你本来需要手工拼装的组件：有状态对话、长期记忆、工具调用、可观测性、限流和 JWT 鉴权。

**它和基础的 LangGraph 搭建有何不同？**
LangGraph 的基础快速上手到"Agent 能在本地跑起来"就停了。本模板在此之上增加了 Alembic 迁移、mem0 + pgvector 长期记忆、Langfuse 追踪、Prometheus + Grafana 仪表盘、JWT 会话、slowapi 限流、带每请求上下文的结构化日志，以及一个环形 fallback 的 LLM 服务——这些都是你本来要单独构建的生产级关注点。

### 搭建与配置

**我需要 Docker 吗？**
推荐但非必需。`make docker-up` 会同时启动 API + PostgreSQL。仅本地搭建见 [docs/getting-started.md](docs/getting-started.md)。

**支持哪些 LLM 提供方？**
目前：通过 `src/agent/services/llm/registry.py` 中的 `LLMRegistry` **仅支持 OpenAI**。通过 LangChain 的 `init_chat_model` 实现多提供方支持（Anthropic、Google、OpenRouter）已在规划中——见 [#51](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template/issues/51)。通过 `.env.development` 中的 `DEFAULT_LLM_MODEL` 配置你的模型。

**如何配置长期记忆？**
长期记忆是自托管的：mem0 在进程内运行，并通过 pgvector 持久化进你现有的 PostgreSQL——无需单独的 mem0 云账号或 API Key。你只需要一个可用的 `OPENAI_API_KEY`（用于事实抽取 + 向量化）以及启用 pgvector 扩展。详见 [docs/memory.md](docs/memory.md)。

### 开发

**如何添加自定义工具？**
在 `src/agent/core/langgraph/tools/` 中放一个 `@tool` 装饰的函数，并注册进该包导出的 `tools` 列表。Agent 在下次启动时自动识别；无需改动图。

**LLM 服务如何处理失败？**
两层：(1) 通过 `tenacity` 进行每次调用的指数退避重试；(2) **环形 fallback**——若当前模型耗尽重试，服务会轮转到 `LLMRegistry` 中的下一个模型并继续。一个总超时预算为整次调用封顶，使延迟保持有界。详见 [docs/llm-service.md](docs/llm-service.md)。

**可以不使用 Langfuse 吗？**
可以。设置 `LANGFUSE_TRACING_ENABLED=false`（或省略 Langfuse 密钥）。Agent 照常运行；结构化日志仍会捕获 request/session/user 上下文。

### 故障排查

**API 无法启动**
- 确保 PostgreSQL 正在运行（`make docker-up` 会随 API 一起拉起）
- 确认 `.env.development` 存在——从 `.env.example` 复制并填入必需密钥
- 应用迁移：`make migrate`

**记忆 / 语义检索返回空**
- 确认 PostgreSQL 实例已启用 `pgvector` 扩展
- 确认 `OPENAI_API_KEY` 有效（mem0 会调用 OpenAI 做事实抽取 + 向量化）
- 检查 `.env.development` 中已设置 `LONG_TERM_MEMORY_MODEL` 与 `LONG_TERM_MEMORY_EMBEDDER_MODEL`

**限流过于激进**
限流规则定义在 `src/agent/core/limiter.py`（slowapi）。调整每路由装饰器或该文件中的默认速率。相关环境变量见 [docs/configuration.md](docs/configuration.md)。
