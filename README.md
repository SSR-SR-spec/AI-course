# 🎓 校园智能助手 — Campus AI Agent

基于 **FastAPI + FAISS + ReAct Loop + DeepSeek** 的多智能体问答系统。

> 本项目为一个真正的 **LLM 驱动的多 Agent 系统**，而非传统的 RAG 管道。
> LLM 自主决策工具调用、任务分解、多智能体协作与答案反思。

---

## ✨ 五大核心能力

| 能力 | 实现 | 说明 |
|------|------|------|
| 🔧 **工具调用** | OpenAI function calling + ToolRegistry | LLM 自主决定调用知识库搜索、网络搜索、绩点计算、校园指南等工具 |
| 📋 **规划与任务分解** | ReAct 循环（Thought→Action→Observation） | Agent 在每步自动思考并拆解复杂问题 |
| 👥 **多智能体协作** | Supervisor→Sub-agent 委托机制 | Supervisor 可委托学术/生活/行政子智能体独立处理任务 |
| 🔍 **自主反思与纠错** | ReflectionEngine（LLM 驱动审核） | 自动检查回答的完整性/准确性，不通过则改进 |
| 🧠 **记忆系统** | LongTermMemory + 会话存储 | 自动提取用户画像（姓名/年级/专业等），跨会话保持 |

## 🏗 架构总览

```
用户 ──→ ReActAgent(Supervisor)
            ├─ LLM 自主决策工具调用
            │   ├─ search_knowledge_base   ← 语义检索知识库
            │   ├─ web_search              ← 联网搜索(Bing备线)
            │   ├─ get_current_time        ← 获取当前时间
            │   ├─ calculate_gpa           ← 计算绩点
            │   ├─ get_campus_guide        ← 校园生活指南
            │   └─ delegate_to_agent ──→ 子智能体(独立 ReAct 循环)
            │        ├─ academic_agent     ← 学术事务
            │        ├─ life_agent         ← 校园生活
            │        └─ admin_agent        ← 行政事务
            ├─ ReflectionEngine           ← LLM 审核回答质量
            └─ LongTermMemory             ← 用户画像提取
```

## 📁 项目结构

```
├── backend/
│   ├── app/
│   │   ├── main.py                 # API 入口
│   │   ├── config.py               # 配置管理(YAML + 环境变量)
│   │   ├── schemas.py              # 数据模型
│   │   ├── settings.yaml           # 配置文件
│   │   ├── agents/                 # 🆕 智能体系统
│   │   │   ├── react.py            # ReAct 循环引擎(核心)
│   │   │   ├── supervisor.py       # 主管智能体(工厂)
│   │   │   └── reflection.py       # 反思审核引擎
│   │   ├── tools/                  # 🆕 工具系统
│   │   │   ├── __init__.py         # ToolRegistry(全局注册表)
│   │   │   ├── kb_tools.py         # 知识库检索工具
│   │   │   ├── campus_tools.py     # 校园工具(绩点/指南/时间)
│   │   │   ├── delegation.py       # 🆕 Agent委托工具
│   │   │   └── web_tools.py        # 🆕 联网搜索工具
│   │   ├── memory/                 # 🆕 记忆系统
│   │   │   └── long_term.py        # 长期记忆(用户画像提取)
│   │   ├── rag/                    # RAG 知识库
│   │   │   ├── kb.py               # FAISS 向量库
│   │   │   ├── chunking.py         # 🆕 三种分块策略
│   │   │   └── document_loader.py  # 文档加载
│   │   ├── llm/client.py           # LLM 调用(支持 tool calling)
│   │   ├── citations.py            # 引用格式化
│   │   ├── sessions.py             # 会话存储
│   │   └── infor/                  # 源文档
│   ├── scripts/
│   │   └── download_embedding_model.py
│   └── requirements.txt
├── frontend/
│   ├── chat.html                   # 🆕 Agent 追踪面板(展示 ReAct 步骤)
│   └── assets/
│       └── hbue-logo.webp
├── Dockerfile                      # Docker 构建
├── docker-compose.yml              # Docker 部署
├── package.bat                     # 🆕 Windows 打包脚本
├── package.sh                      # 🆕 Unix 打包脚本
├── pyproject.toml                  # 🆕 Python 包元数据
├── deploy.sh                       # 一键部署(Ubuntu)
└── .env.example                    # 环境变量模板
```

## 🚀 快速开始

### 方式一：本地运行（推荐）

**环境要求：** Python 3.10+

```bash
# 1. 进入项目根目录
cd AI-course

# 2. 创建虚拟环境
python -m venv backend/.venv

# Windows:
backend\.venv\Scripts\activate
# Linux/Mac:
source backend/.venv/bin/activate

# 3. 安装依赖（国内用阿里云镜像更快）
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r backend/requirements.txt

# 4. 下载 embedding 模型（首次需要，约 500MB）
python backend/scripts/download_embedding_model.py

# 5. 启动服务
cd backend
uvicorn app.main:app --reload --port 8000

# 6. 打开浏览器访问:
#    http://127.0.0.1:8000/chat.html
```

### 方式二：Docker 部署（适合服务器）

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API Key

# 2. 一键启动
docker compose up -d

# 3. 访问 http://localhost
```

### 方式三：从打包文件部署

```bash
# Windows
# 1. 解压 campus-ai-agent_v2.0.0.zip
# 2. 双击 setup.bat    ← 自动配置环境
# 3. 双击 start_backend.bat  ← 启动服务
# 4. 打开 http://127.0.0.1:8000/chat.html

# Linux/Mac
# 1. tar -xzf campus-ai-agent_v2.0.0.tar.gz
# 2. bash setup.sh     ← 自动配置环境
# 3. bash start.sh     ← 启动服务
```

## 🔧 配置说明

### LLM 配置（两种模式）

**Mock 模式**（默认，无需 API Key）：
```bash
# 直接启动即可，LLM 行为由模拟器驱动
# 适合验证 ReAct 循环和前端交互
```

**真实模式**（需要 DeepSeek API Key）：
```bash
# 方法 1: 环境变量
export OPENAI_API_KEY="sk-xxx"

# 方法 2: settings.local.yaml（推荐）
# 编辑 backend/app/settings.local.yaml:
llm:
  openai_api_key: "sk-xxx"
```

### 文本分块策略

| 策略 | 配置值 | 说明 |
|------|--------|------|
| 简单滑动窗口 | `simple` | 固定字符窗口，效率最高 |
| 递归分段 | `recursive`（默认） | 段落→句子→字符三级降级，保语义完整 |
| 语义分块 | `semantic` | 基于嵌入向量的边界检测，效果最好 |

```yaml
# settings.yaml
rag:
  chunk_strategy: recursive    # simple | recursive | semantic
  chunk_size: 700
  chunk_overlap: 100
  similarity_threshold: 0.75   # semantic 策略的边界阈值
```

### 联网搜索

Agent 会自动联网搜索知识库中不存在的信息，无需额外配置。
搜索引擎自动回退：DuckDuckGo → Bing（BeautifulSoup 解析）。

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/agent/chat` | POST | **Agent 智能体聊天**（推荐，展示 ReAct 追踪） |
| `/chat` | POST | 传统 RAG 聊天（向后兼容） |
| `/health` | GET | 健康检查 |
| `/kb/stats` | GET | 知识库统计 |
| `/ingest/local-folder` | POST | 导入文档到知识库 |

## 📦 打包项目

项目支持两种打包方式：

### Windows 打包

```bash
# 双击 package.bat，或在终端运行：
.\package.bat
```

### Unix 打包

```bash
bash package.sh
```

输出文件位于 `dist/campus-ai-agent_v2.0.0.zip`（及 `.tar.gz`）。

## ❓ 常见问题

**Q: 为什么回答显示的是 Mock 模型？**
未配置 API Key，系统自动降级到 Mock 模式。配置 `settings.local.yaml` 即可。

**Q: 网页上能看到 Agent 的思考过程吗？**
可以。在 `chat.html` 中，每条回答下方都有一个可展开的 **Agent 追踪面板**，显示：
💭 思考过程 → 🔧 工具调用 → 📎 工具结果 → ✅ 最终回答

**Q: 知识库里没有的数据怎么办？**
Agent 会自动调用 `web_search` 工具联网搜索，无需手动干预。

**Q: 下载 embedding 模型太慢？**
脚本自动使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。
也可手动设置：`export HF_ENDPOINT=https://hf-mirror.com`

**Q: 如何提交课程作业？**
运行 `package.bat`（Windows）或 `package.sh`（Linux/Mac）打包后，提交 `dist/campus-ai-agent_v2.0.0.zip`。
