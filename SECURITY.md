# 安全模型（代码验真）

## 威胁模型

- **不可信输入**：模型 API 返回的文本（含被提示注入诱导的代码）属于不可信外部输入。
  代码验真会将其中的 Python 代码提取出来并执行，因此执行路径必须视为"执行任意不可信代码"。
- **攻击者目标**：读取宿主敏感文件（配置、源码、历史记录、环境变量中的凭据）、
  在权限范围内修改文件、耗尽宿主资源（内存炸弹 / 死循环 / 进程炸弹）、建立网络连接外传数据。
- **默认姿势**：**默认不执行**（`code_verify_mode=off`，仅展示 + `compile()` 语法检查）。
  任何执行行为都必须由用户在界面上显式选择隔离模式。

## 隔离边界（native-sandbox，Windows）

安全边界全部位于操作系统层，不依赖 Python 运行时黑名单：

| 威胁 | 对策 | 机制 |
|---|---|---|
| 读取宿主文件 | 容器 SID 未授权的路径一律拒绝 | AppContainer（空能力集，系统级强制 ACL） |
| 写宿主文件 | 仅一次性工作目录被显式授权 | AppContainer + `icacls` 授予容器 SID |
| 网络访问 | 默认无网络能力，连接被系统拒绝 | AppContainer 空能力集 |
| 内存炸弹 | 内存配额内终止 | Job Object `JOB_OBJECT_LIMIT_JOB_MEMORY`（默认 256MB） |
| 死循环 | CPU 时间配额内终止 | Job Object `JOB_OBJECT_LIMIT_PROCESS_TIME`（默认 5s） |
| 进程炸弹 | 活动进程数配额内拒绝 | Job Object `JOB_OBJECT_LIMIT_ACTIVE_PROCESS`（默认 8） |
| 输出/磁盘写 | 超限输出在配额内被终止 | 运行期监视线程轮询 stdout/stderr 文件大小（默认 1MB，`max_output_bytes` 可调），超限 `TerminateJobObject` 强杀；返回内容限量读取（`max_output_chars` 4096 字符） |
| 超时/残留进程 | 强杀整棵进程树 | `TerminateJobObject` + `KILL_ON_JOB_CLOSE` |
| 凭据/环境泄露 | 不继承宿主环境变量 | 显式环境白名单（仅 TEMP/TMP/LOCALAPPDATA/SYSTEMROOT/WINDIR/PYTHONDONTWRITEBYTECODE/PYTHONUTF8） |
| 解释器访问系统目录 | 使用部署方指定的自包含运行时 | 部署方配置 `MODEL_DUEL_SANDBOX_PYTHON`（应用不下载/更新运行时，仅授权该目录） |

每个测试用例使用全新一次性工作目录，执行后销毁；每次执行新建 AppContainer 容器进程。

### 并发隔离：固定 SID 池

- 所有并发运行共享同一个工作目录 ACL 会泄漏跨任务数据，因此并发执行
  按固定池（`PROFILE_POOL_SIZE = 4`，≥ 执行器并发上限 `_CODE_SEM = 2`）
  轮换分配互异的 AppContainer profile（`arena.codesb.0..3`），
  池内 profile 每次执行产生互不相同的 SID 与容器 ACL。
- 池满（并发超过上限）时抛出明确错误，不会退化为共享 SID 运行。
- 跨时间复用同一 profile 无风险：SID 仅被用于一次性工作目录的授权，
  工作目录每次执行后即删除。

### 已知残留（与 UWP 应用一致）

- 沙箱代码可读取**世界可读的系统公共文件**（如 `C:\Windows\...\hosts`、系统目录中的
  只读配置文件）。这是 AppContainer 的系统级行为（Windows 商店应用同样如此），
  无法在不引入内核过滤驱动的条件下阻止；边界是**用户数据（文档/项目/配置/凭据）不可读写**。
- `LOCALAPPDATA` 由内核要求必须存在（用于推导 AppContainer 包状态目录，
  缺失将导致 `ERROR_ENVVAR_NOT_FOUND`），实现上指向一次性工作目录而非宿主路径。

## 信任边界与前提

- 本方案面向 **Windows 10/11**（AppContainer 为系统内建能力，标准用户可用，无需管理员）。
- **运行时由部署方提供（issue #16）**：应用不下载、不解压、不安装、不更新 Python。
  未配置 `MODEL_DUEL_SANDBOX_PYTHON`（须为绝对路径、自包含单目录运行时）时，
  `native-sandbox` 明确不可用（fail closed），绝不联网下载或从 PATH 寻找替代解释器；
  路径不存在、相对路径、目录、不可执行文件、不兼容版本（要求 Python ≥ 3.10）与
  stdlib 越界（venv 等）均被明确拒绝，错误信息可操作。
- **信任边界**：运行时的安装、补丁、来源与完整性验证全部由部署流程负责；
  应用仅校验路径/文件类型/版本/自包含性并引用该路径。冒烟校验仅在宿主侧执行
  部署方显式配置的解释器（无害代码），不执行任何下载或缓存内容。
- 若宿主进程已被放入不可嵌套的 Job Object（某些 CI/服务托管场景），
  `AssignProcessToJobObject` 可能失败，此时返回明确错误并回退 `off`，不影响主服务。
- 主服务默认仅监听 `127.0.0.1`；对外暴露接口前请先确认隔离后端可用。

## 部署与审计

- **共享模式必须启用 TLS（或置于可信反向代理之后）**：局域网共享模式下若直接明文传输，
  访问令牌与模型 API Key 可被同网段嗅探；Origin 校验以浏览器视角的 Host 为准，
  反向代理须保持 Host 头一致。单机模式（默认 `127.0.0.1`）无需额外部署。
- **审计日志**：关键操作（评测启动、评审提交、历史/数据集删除、鉴权失败）以 JSONL 追加到
  `.eval/audit.log`。仅记录白名单字段（`ts/event/job_id/target/path/actor`），写入前统一
  递归脱敏，API Key 等敏感内容永不进入日志；审计模块异常静默，不影响主流程。
- **非 Windows 平台**：`native-sandbox` 仅支持 Windows（AppContainer 与 Job Object 为系统
  内建能力），非 Windows 上自动回退 `off`（仅语法检查 + 展示），**不存在**"无隔离直接执行"
  的路径；代码验真在 Linux/macOS 上不可用属预期行为（fail closed），Linux/macOS 隔离后端
  列为后续路线（roadmap），不做静默降级。
- **平台能力上报**：`GET /api/code-runner/status` 返回当前平台、各模式可用性、
  `probe`（组件存在性快检）与 `selfcheck`（真实受限进程自检，`?selfcheck=1` 触发）
  两级健康状态，前端与运维可据此准确判断当前平台支持的功能及限制。
- **CI 矩阵**：Linux / macOS / Windows 三平台均运行公共业务、API、数据集、报告与安全
  回归测试，避免平台分支漂移；Windows 额外执行隔离后端真实冒烟（进程创建、stdin/stdout、
  超时、资源配额、文件与网络边界），平台专属测试统一 `native` 标记，不在错误平台固定失败，
  也不因全部 skip 使功能看似已验证。

## 验收对照

见 `webapp/tests/test_sandbox.py` 与 `webapp/scripts/sandbox_selfcheck.py`，
覆盖：宿主文件读取、目录外写入、网络连接、内存/CPU/进程配额、**输出磁盘配额（stdout/stderr 超限终止）**、
环境变量不继承、并发任务的跨任务工作目录隔离（互异 SID）。
