# 模型对决评测平台 (Model Duel Evaluation)

输入模型 API 配置，自动完成「出题 → 作答 → 评审 → 报告」的评测闭环，支持**双模型对决**与 **N 模型横向 benchmark（排行榜）**。用户操作指南见 [USER_MANUAL.md](USER_MANUAL.md)。

## 功能特性

- **八维能力体系**：知识 / 数学 / 逻辑推理 / 代码 / 语言 / 指令遵循与对齐 / 安全与价值观（单独展示不计总分）/ 长文本与效率稳定性
- **双任务类型**：判别式（expected 比对）与生成式（rubric 评审）一等公民
- **三类评测入口**：双模型对决 / N 模型 benchmark 批次（1 评测集 × N 模型 × M 轮，单臂评审）/ 对抗扰动与偏见测试（改写、噪声注入、属性扰动，衰减曲线 + 歧视检测）
- **三种评审方式**：人工双盲（逐题独立随机交换） / AI 自动评审 / Hybrid（AI 预评 + Top-K 人工复核）；金标元评估（Spearman / Kappa）随报告展示
- **指标引擎**：Top-1 / F1 / 精确匹配 / 松弛准确率（判别式）；语义相似度（embedding 双 provider）/ 一致性 / BLEU / ROUGE（生成式）；截断与无效作答转报告警告；预算熔断
- **LLM 出题**：生成 → 五级自动校验（去重/可解性/rubric 完整性/防泄漏/内容安全）→ 人工审核 → 版本化入库；RAG/grounding 忠实性评测
- **Bad Case 体系**：自动挖掘（低分/双败/分歧/边缘情境）→ LLM 归因 + 人工确认 → 基准饱和度监测
- **排行榜形态**：分维度排名 + 综合分 + 胜率矩阵 + bootstrap CI「差异不显著」标注 + 多模型雷达 / 箱线 / 散点 / K-召回率
- **资源管理**：优先级队列 + 并发配额（`MODEL_DUEL_CONCURRENCY`）+ 排队替代 429 + 断点续跑 + 批次整体取消/重跑
- **任务视图 SSE 实时刷新**（ticket 机制，轮询兜底）
- **评测可复现**：环境快照（OS/Python/依赖 + seed/温度/模型/评审配置指纹/数据集版本）+ 评测包导出（zip）
- **代码验真**：默认仅展示 + 语法检查；显式开启后走 Windows 原生隔离（AppContainer + Job Object）逐用例执行

## 快速启动

```powershell
cd webapp
pip install -r requirements.txt
.\run.ps1
```

访问 `http://localhost:8910`。手动启动：`python -m uvicorn backend.main:app --host 127.0.0.1 --port 8910`。

> 服务默认仅监听 `127.0.0.1`。局域网访问：`.\run.ps1 -ListenAddress 0.0.0.0` 并设置 `MODEL_DUEL_TOKEN`（见下）。

## 部署与安全

**单机模式（默认）**：仅监听回环地址，校验 Host 头防 DNS rebinding；无强制认证。

**共享模式（局域网/远程）**

```powershell
$env:MODEL_DUEL_TOKEN = "<强随机串，建议 32+ 位>"   # 鉴权令牌
$env:MODEL_DUEL_RATE_LIMIT = "30"                  # 可选：每 IP 每分钟最大写请求数
.\run.ps1 -ListenAddress 0.0.0.0
```

- 全部 `/api/*` 要求 `Authorization: Bearer <令牌>`（恒定时间比较）；共享模式下 `/docs` 亦需令牌
- 写请求校验 Origin/Referer 同源（CSRF） + 按 IP 限流
- **SSE 进度流**：前端先经已认证的 `POST /api/eval/{job_id}/events/ticket`（或 `POST /api/tasks/events/ticket`）换取**短时（60s）、单次、作用域绑定**的随机 ticket，URL 只携带 ticket 用后即焚；长期管理员令牌不出现于 URL / Referer / 访问日志（认证失败静默 204 不记审计）
- **SSRF 防护**：模型 URL 仅允许公网 http/https（回环/私网/云元数据地址拒绝）；内网场景显式设 `MODEL_DUEL_ALLOW_PRIVATE_UPSTREAM=1`
- **Key 不落盘不变量**：模型配置库 / 辅助配置（gen/judge/embedding）仅存内存，持久化与审计递归脱敏
- 生产建议置于反向代理 + HTTPS 之后（Origin 校验以浏览器视角 Host 为准）
- 审计日志：关键操作 JSONL 追加 `.eval/audit.log`，仅白名单字段、永不包含 Key

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_DUEL_TOKEN` | 空 | 共享模式鉴权令牌 |
| `MODEL_DUEL_RATE_LIMIT` | 30 | 每 IP 每分钟写请求上限 |
| `MODEL_DUEL_CONCURRENCY` | 2 | 调度器并发配额（CPU 池化） |
| `MODEL_DUEL_SSE_TICKET_TTL` | 60 | SSE ticket 有效期（秒） |
| `MODEL_DUEL_ALLOW_PRIVATE_UPSTREAM` | 0 | 放行私网上游（内网代理场景） |
| `MODEL_DUEL_SANDBOX_PYTHON` | 空 | 沙箱预装 Python 运行时绝对路径（自包含） |
| `MODEL_DUEL_EMBEDDING_URL/KEY/NAME` | 空 | 外部 embedding provider 默认配置 |
| `MODEL_DUEL_BGE_MODEL_DIR` | 空 | 本地 BGE 模型目录（懒加载） |

## 前端页面（9 页）

| 页面 | 功能 |
|---|---|
| `/`（index.html） | 配置页：模型 A/B、配置库、评测集、LLM 出题、历史记录 |
| `/tasks.html` | 任务调度：排队/运行视图、优先级、取消、benchmark 批次创建与控制、SSE 实时刷新 |
| `/leaderboard.html` | 排行榜：历史 job 聚合 + 查看 |
| `/perturb.html` | 扰动与偏见测试 |
| `/dashboard.html` | KPI 看板（耗时/Token/CPU） |
| `/badcases.html` | Bad Case 库（筛选/归因确认/导出） |
| `/gen_review.html` | 出题批次审核（批准/驳回/编辑） |
| `/report.html` | 评测报告（指标/图表/元评估/环境快照/导出） |
| `/review.html` | 人工双盲评审（含 Hybrid 复核） |

公共层：`frontend/common.js`（apiFetch/toast/三态/顶栏/轮询）+ `frontend/common.css`（统一布局），9 页统一导航与空/加载/错误态。

## 评测集格式

```json
{
  "name": "评测集名称",
  "description": "可选",
  "tasks": [{
    "id": "T1", "type": "判别式|生成式", "dimension": "八维之一",
    "difficulty": "easy|medium|hard", "tags": [],
    "prompt": "题目", "expected": "判别式必填；生成式可选",
    "rubric_note": "生成式必填", "context": "可选（RAG 忠实性评测载体）",
    "test_cases": [{"input": "...", "expected": "..."}]
  }]
}
```

JSON / CSV / Markdown / TXT 四格式上传（页面可下载模板），`type` 缺省=判别式，旧文件零破坏。

## 自动化测试

```powershell
cd webapp
python -m scripts.sandbox_selfcheck   # 首次运行代码验真相关测试前
python -m pytest tests                # 全量（不含 perf）
python -m pytest tests -m perf        # 性能基准（长文本边界/并发压力/N=10 批次）
python -m pytest tests --cov=backend --cov-report=term-missing
```

分层说明（详见 tests/）：

- **纯函数单测**：parsers/datasets/storage/tasks/metrics/embed/extract/stats/perturb/leaderboard/generator/badcase/scheduler/长期文本基准等
- **集成测试**：全链路状态机（含 AI 评审/hybrid）、batch 聚合与部分失败、批次取消/重跑、断点续跑、配置库 CRUD+脱敏、排队/取消/优先级、SSE ticket（eval + tasks 双路径）、饱和度幂等
- **安全回归**：Key 不落盘、SSE ticket 401/204/剥离、代理日志不含长期令牌、SSRF、审计白名单、CSP
- **浏览器 E2E（本地 playwright）**：`test_xss_playwright.py`（9 页冒烟/顶栏/环境快照卡片/评审流程/任务页），未安装 playwright 自动跳过，CI 不运行
- **性能基准**：`test_perf.py`（`-m perf`，默认排除于 CI 回归）

约定：零真实网络（内存 mock 上游）、存储重定向临时目录、先回归后实现（红→绿）。CI（GitHub Actions）三平台矩阵：Linux/macOS 跑 `-m "not native and not perf"`；Windows 全量（含沙箱）`-m "not perf"`，另跑 `tests/test_perf.py`。

## 项目结构

```
webapp/
├── backend/
│   ├── main.py            # FastAPI 路由 + 调度接线（约 3k 行）
│   ├── schemas.py         # Pydantic 请求模型
│   ├── storage.py         # 历史/评测集/排行榜/扰动/badcase 文件库
│   ├── scheduler.py       # 优先级队列 + 并发配额（纯逻辑）
│   ├── models_registry.py # 模型配置库（Key 仅内存）
│   ├── access.py          # 认证/Host/Origin/限流/SSE ticket
│   ├── audit.py / security.py / ssrf.py / sse_ticket.py / hwmon.py
│   └── engine/
│       ├── tasks.py       # 八维内置题库 + 任务集生成
│       ├── executor.py    # 单模型流式作答（重试/截断/增量落盘）
│       ├── judge.py       # AI 双盲/单臂评审 + 逐题随机交换 + 健康度
│       ├── human_review.py# 人工双盲评审 + 轮次聚合
│       ├── reviewer.py    # 评审方协议（Agent/Human/Hybrid）
│       ├── metrics.py / embed.py / extract.py / stats.py / budget.py
│       ├── generator.py   # LLM 出题 pipeline（五级校验）
│       ├── perturb.py     # 扰动/偏见管线
│       ├── badcase.py     # 错误分类/挖掘/归因
│       ├── leaderboard.py # 排行榜聚合（bootstrap CI）
│       ├── rag_demo.py / longtext.py / gold.py  # 内置资产（启动幂等生成）
│       ├── mock.py        # 演示数据（测试结果案例 / demo_seed）
│       └── isolation/     # 代码沙箱（off / Windows 原生）
├── frontend/              # 9 页面 + common.js/css + echarts.min.js
├── scripts/               # sandbox_selfcheck / scrub_history / demo_seed /
│                          # benchmark_local / gen_longtext_bench
└── tests/                 # 全量回归（见上）
```

## 已知限制

- 纯文本评测（非目标：多模态 / 多轮 Agent）
- 优先级仅对排队中生效（运行中不抢占）
- v1 内存队列：重启后排队任务沉降 error（不自动恢复）
- batch 执行单元不支持断点续跑（resume 面向常规评测任务）
- 评测包仅提供导出（第三方复核），不支持导入
