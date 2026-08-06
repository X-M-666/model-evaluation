# 模型对决评测平台 (Model Duel Evaluation)

输入两个大模型的 API 配置，自动完成「出题 → 双模型作答 → 双盲评审 → 报告展示」的完整评测流程，帮助对比任意两个 LLM 的能力表现。

## 功能特性

- **内置题库**：七大能力维度（知识、推理、代码、语言、指令遵循与对齐、长文本与多模态、效率与稳定性），随机抽题，支持固定种子复现
- **自定义评测集**：上传 JSON / CSV / Markdown / TXT 文件（页面提供各格式模板下载），也可直接粘贴 JSON
- **双盲评审**：评审模型在不知道答案归属的情况下打分，避免偏见；内置复核仲裁
- **重复评测**：重复 N 次自动取平均分 + 标准差，衡量模型稳定性
- **代码验真**：代码题默认仅展示 + 语法检查、**不执行**；显式开启后可走 Windows 原生隔离（AppContainer + Job Object）逐用例执行（详见 SECURITY.md）
- **效率指标**：记录每次调用的延迟、Token 消耗
- **报告页**：ECharts 雷达图、逐题评分表、综合结论、答卷原文，支持导出 JSON

## 快速启动

```powershell
cd webapp
pip install -r requirements.txt
.\run.ps1
```

启动后访问 `http://localhost:8910`。

也可手动启动：

```bash
cd webapp
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8910
```

> 服务默认仅监听 `127.0.0.1`。如需局域网访问，请显式指定 `--host 0.0.0.0`（`.\run.ps1 -ListenAddress 0.0.0.0`），并设置 `MODEL_DUEL_TOKEN` 启用鉴权（见下文「部署与安全」）。

## 部署与安全（issue #8）

平台内置安全加固，默认（单机模式）即为安全配置；两种部署模式：

**单机模式（默认）**
- 仅监听 `127.0.0.1`（`run.ps1` 默认参数），服务端校验 Host 头为回环别名，阻止 DNS rebinding 与局域网直连
- 若设置了 `MODEL_DUEL_TOKEN` 则自动进入共享模式，本机访问也需要令牌

**共享模式（局域网/远程部署）**

```powershell
$env:MODEL_DUEL_TOKEN = "<强随机串，建议 32+ 位>"   # 鉴权令牌
$env:MODEL_DUEL_RATE_LIMIT = "30"                  # 可选：每 IP 每分钟最大写请求数（默认 30）
.\run.ps1 -ListenAddress 0.0.0.0                   # 监听所有网卡
```

- 所有 `/api/*` 接口要求 `Authorization: Bearer <令牌>`（恒定时间比较）；页面顶部可填入令牌，保存在本机浏览器 localStorage
- 共享模式下 `/docs`、`/openapi.json` 等接口文档同样需要令牌访问（单机模式保持开放便于调试）
- 写请求（POST/PUT/DELETE）校验 Origin/Referer 与 Host 同源，阻止跨站请求伪造（CSRF）
- 写请求限流：共享模式下每 IP 每分钟最多 30 次（`MODEL_DUEL_RATE_LIMIT` 可调）
- 评测接口并发上限 2 个任务（防资源耗尽），文件上传上限 5MB、JSON 粘贴上限 2MB、数据集上限 200 题；超限请求在读取阶段即被截断，不会整体读入内存
- SSE 进度流（EventSource 无法携带自定义 header）：前端先通过已认证的 `POST /api/eval/{job_id}/events/ticket` 换取**短时（默认 60 秒）、单次、仅限该任务 events 路由**的随机 ticket，URL 只携带 ticket，用后即焚；长期管理员令牌不出现在 URL、Referer 或访问日志中，反向代理 / WAF / APM 日志不会记录长期令牌（`MODEL_DUEL_SSE_TICKET_TTL` 可调有效期）。ticket 认证失败（如浏览器断线自动重连复用已消费 ticket）静默返回 `204` 且不记审计，避免日志与审计噪声；未携带 ticket 的未认证请求仍返回 `401` 并记录审计
- **SSRF 防护**：模型 URL 仅允许公网 http/https 目标（自动解析全部 IP，拒绝回环/私网/链路本地/云元数据地址，如 `127.0.0.1`、`169.254.169.254`、`192.168.x.x`、`::1`）；内网网关/代理场景需显式设置 `MODEL_DUEL_ALLOW_PRIVATE_UPSTREAM=1` 放行私网目标
- 生产环境**建议**置于反向代理后并启用 HTTPS（Origin 校验以浏览器视角的 Host 为准）；共享模式下**必须**启用 TLS 或置于可信反向代理之后，否则令牌与模型凭据会以明文在局域网传输
- **审计日志**：关键操作（评测启动、评审提交、历史/数据集删除、鉴权失败）以 JSONL 追加至 `.eval/audit.log`，仅记录白名单字段并经递归脱敏，**永不包含 API Key**

## 使用说明

1. 填写模型 A / B 的 API URL（OpenAI 兼容）、API Key、模型名称，可展开高级参数设置 temperature / max_tokens / top_p
2. 选择内置题库（可勾选维度）或自定义评测集（上传文件 / 粘贴 JSON / 从已有评测集选择）
3. 设置重复次数（>1 时自动跑 N 次取平均），点击「开始评测」
4. 实时进度条跟随，完成后自动跳转报告页
5. 页面提供「测试结果案例」按钮，无需 API Key 即可体验完整报告效果

## 自定义评测集格式

统一标准格式（各格式上传后都转换为该结构）：

```json
{
  "name": "评测集名称",
  "description": "可选描述",
  "tasks": [
    {
      "id": "T1",
      "dimension": "知识能力",
      "prompt": "题目内容",
      "expected": "期望答案",
      "rubric_note": "评分标准（可选）"
    }
  ]
}
```

- **JSON**：完整格式（含 test_cases / rubric_note）或简化格式（prompt + expected）
- **CSV**：两列（prompt, expected）或六列（id, dimension, prompt, expected, rubric_note, difficulty）
- **Markdown**：`# 标题` / `## 维度` / `### 题号` / `**题目：**` / `**期望：**` / `**评分标准：**`
- **TXT**：每行 `题目 | 期望答案`（或 TAB 分隔），`# 开头`为维度

页面内可下载各格式示例模板。

## API Key 安全

- API Key 仅保存在后端内存中，**不落盘**；审计日志只记录事件白名单字段（时间/事件/对象/来源），写入前统一递归脱敏，不含任何 Key 明文
- 持久化与对外响应统一经过递归脱敏：`report.json` 与历史/报告接口响应均不包含 `key` 字段
- 历史记录的 config 文件中 Key 一律以 `***` 打码
- 已运行过真实评测的用户：如担心旧历史泄露，请**先轮换相关 API Key**，再执行以下命令清洗历史文件（脱敏重写，`--dry-run` 可预览）：
  ```bash
  cd webapp
  python -m scripts.scrub_history --dry-run
  python -m scripts.scrub_history
  ```
- 评测记录可通过页面「删除」按钮移除

## 任务生命周期（issue #14）

- 状态机：`executing` →（作答完成）→ `reviewing` →（提交评分）→ `completed`；异常落 `error`。
- 删除运行中评测（`DELETE /api/history/{job_id}`）为**同步语义**：先置 `cancelling`、取消后台任务并等待其安全终止（最长约 10 秒兜底），随后才移除内存状态与磁盘目录，返回 `200`；后台流程不会在删除后复活任何文件，上游模型请求随之中断。未知任务返回 `404`，已终态任务直接删除。
- 任务被删除/取消后状态为 `cancelled`，其评审提交接口返回 `409`，避免已删除数据被重新写入。
- 服务关闭时统一取消并回收全部后台任务。

## 代码验真与安全

- 模型输出属于**不可信外部输入**，默认不对其执行：`code_verify_mode=off` 时仅展示代码并做语法检查
- 显式选择 `native-sandbox` 后，代码题会在 **Windows AppContainer（文件/网络硬隔离）+ Job Object（内存/CPU/进程数配额）** 中逐用例执行，超时强杀进程树，不继承宿主环境变量；stdout/stderr 输出另有 1MB 磁盘配额（运行期监视，超限即终止进程，防止撑爆宿主磁盘，`limits.max_output_bytes` 可调）
- **运行时由部署方预装并提供（issue #16）**：应用不会自行下载、解压、安装或更新 Python。设置环境变量 `MODEL_DUEL_SANDBOX_PYTHON` 为部署方预装 `python.exe` 的**绝对路径**；未配置或运行时非法时 `native-sandbox` 明确不可用（fail closed），不会联网下载，也不会从系统 PATH 中寻找替代解释器
- **自包含单目录要求**：指向的必须是自包含运行时（stdlib 位于 `python.exe` 所在目录内，如完整安装根目录或 embeddable 发行包目录）；venv 等跨目录运行时在校验时被拒绝（AppContainer 仅授权该目录，沙箱内无法读取外部 stdlib）
- **信任边界**：运行时的安装、补丁、来源与完整性验证由部署流程负责；应用仅引用配置路径，并校验路径/文件类型/版本/自包含性
- 开启前请运行自检：`cd webapp; python -m scripts.sandbox_selfcheck`
- 完整威胁模型见 [SECURITY.md](SECURITY.md)

### 预装沙箱运行时（Windows）

任选其一，均由部署方完成，应用不参与：

1. **完整安装**：使用官方安装包安装 Python 后，将安装根目录 `python.exe` 的绝对路径写入 `MODEL_DUEL_SANDBOX_PYTHON`。
2. **embeddable 发行包**：手动从 python.org 下载 `python-3.12.x-embed-amd64.zip`，解压到独立目录（如 `C:\arena\runtime\`），将其中 `python.exe` 的绝对路径写入 `MODEL_DUEL_SANDBOX_PYTHON`。下载后请自行核验制品完整性（官方 SHA-256 等）。

版本升级或补丁时，由部署方重新放置运行时并重新执行自检；应用不感知也不自动更新。

## 自动化测试（issue #11）

仓库以回归测试保护「安全、结果正确性、可恢复性」三类核心不变量（详见 `.github/workflows/ci.yml` 与 `webapp/tests/`）。

**本地运行全部测试（推荐）：**

```powershell
cd webapp
python -m scripts.sandbox_selfcheck   # 首次运行代码验真相关测试前执行
python -m pytest tests
```

**带覆盖率报告：**

```powershell
python -m pytest tests --cov=backend --cov-report=term-missing
```

分层说明：

- **纯函数单元测试**：`test_parsers.py` / `test_datasets.py`（四种评测集格式的正常、边界与恶意输入）、`test_storage.py`（配置脱敏落盘、状态推断、损坏 JSON、数据集名消毒）、`test_tasks.py`（内置题库七维度完整性、T/TB 双题、代码/效率题用例形态、任务集生成与数据集转换）、`test_report_builder.py`（报告构建空数据/异常状态）、`test_human_review.py`（X/Y 身份映射）、`test_ssrf.py`、`test_security.py`（Key 不落盘/不外泄）
- **FastAPI 集成测试**：`test_access_control.py` / `test_limits.py`（认证、Host/Origin 校验、限流、并发、上传大小）、`test_recovery.py`（磁盘态任务重启后可评审）、`test_review_validation.py`（评分完整性/唯一性）、`test_report_reveal.py` / `test_report_rounds.py`（reveal 映射与多轮聚合语义）、`test_audit.py`（审计日志白名单字段、递归脱敏、关键操作事件）、`test_proxy_logs.py`（真实 uvicorn 访问日志验证：ticket / 长期 Token 永不进入日志，含认证失败路径）
- **浏览器端到端（本地）**：`test_xss_playwright.py` 需额外安装 `playwright` 与浏览器（`pip install playwright; python -m playwright install chromium`），未安装时自动跳过；CI 不运行此层

约定：

- 所有测试**零真实网络**：模型调用全部走内存 mock，请求经 TestClient/本机端口完成；
  `conftest.py` 的全局网络封锁 fixture 会在未显式 mock 的真实外连（仅回环放行）发生时直接抛错并指出调用位置
- 存储目录由 `conftest.py` 自动重定向到临时目录，**不污染 `.eval/` 与 `webapp/data/`**；失败日志不含敏感字段
- 修复相关 Issue 时，先加入能失败的回归测试，再验证修复（红 → 绿）
- CI（GitHub Actions）：**Linux / macOS / Windows 三平台矩阵**，pull request 与主分支推送时自动执行。
  Linux/macOS 运行公共测试（`-m "not native"`）；Windows 先将 setup-python 解释器路径写入
  `MODEL_DUEL_SANDBOX_PYTHON`，再跑 `scripts.sandbox_selfcheck` 完整自检（校验配置运行时并实测
  逃逸/配额），最后全量执行测试（含 Windows 原生隔离测试）


## 项目结构

```
webapp/
├── backend/
│   ├── main.py            # FastAPI 路由 + 评测调度
│   ├── schemas.py         # Pydantic 模型
│   ├── storage.py         # 历史记录 / 评测集文件库
│   └── engine/
│       ├── tasks.py       # 内置题库（七维度）
│       ├── executor.py    # 双模型并发调用
│       ├── judge.py       # 双盲评审
│       ├── sandbox.py     # 代码验真编排（非安全边界）
│       ├── isolation/     # 代码隔离后端（off / Windows 原生 AppContainer+Job Object）
│       ├── parsers.py     # 评测集解析器注册表（JSON/CSV/MD/TXT）
│       ├── datasets.py    # 评测集校验/归一化
│       └── mock.py        # 模拟数据生成（演示用）
├── frontend/
│   ├── index.html         # 配置页
│   └── report.html        # 报告页
├── requirements.txt
└── run.ps1
```

## 技术栈

- 后端：Python FastAPI + httpx（异步并发调用）
- 前端：原生 HTML/JS + ECharts CDN
- 存储：本地 JSON 文件（`.eval/history/` 评测记录，`webapp/data/datasets/` 评测集）
