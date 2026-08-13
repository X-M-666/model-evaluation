# -*- coding: utf-8 -*-
"""Pydantic 请求/响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    url: str = Field(..., description="API base URL，如 https://api.example.com/v1")
    key: str = Field(..., description="API Key")
    name: str = Field(..., description="模型名称，如 gpt-4o")
    temperature: float = Field(0.7, ge=0, le=2, description="温度参数 0-2")
    max_tokens: int = Field(4096, ge=1, le=128000, description="最大生成 token 数")
    top_p: float | None = Field(None, ge=0, le=1, description="Top-P 采样，None=不设置")


class StartRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_a: ModelConfig
    model_b: ModelConfig
    dims: list[str] | None = Field(None, description="评测维度列表，None=全部七维度")
    seed: int | None = Field(None, description="随机种子")
    dataset_name: str | None = Field(None, description="自定义评测集名称，None=用内置题库")
    repeat_n: int = Field(1, ge=1, le=20, description="重复评测次数，取平均")
    num_questions: int | None = Field(None, ge=1, le=8, description="内置题库题目数量（仅内置题库模式生效，3/5/7/8，None=全维度）")
    code_verify_mode: str = Field(
        "off", pattern=r"^(off|native-sandbox)$",
        description="代码验真模式：off=仅展示与语法检查（默认，不执行）；native-sandbox=Windows 原生隔离（AppContainer+Job Object）",
    )
    prompt_strategy: str = Field(
        "cot", pattern=r"^(cot|direct|fewshot)$",
        description="执行侧提示策略（迭代二）：cot=零样本思维链（默认）；direct=直答原文；fewshot=注入示例",
    )
    review: ReviewConfig | None = Field(None, description="AI 评审配置（迭代二，缺省 pure_human）")
    budget: BudgetConfig | None = Field(None, description="预算熔断配置（迭代二，缺省不限制）")
    embedding: EmbeddingConfig | None = Field(None, description="embedding 配置（迭代二，缺省 auto）")


class StartResponse(BaseModel):
    job_id: str


class ReviewScoreItem(BaseModel):
    id: str = Field(..., description="题目 id")
    round: int = Field(1, ge=1, description="轮次（repeat_n>1 时从 1 开始）")
    answer_x: float = Field(..., ge=0, le=10, description="答案X 得分 0-10")
    answer_y: float = Field(..., ge=0, le=10, description="答案Y 得分 0-10")
    note: str = Field("", max_length=2000, description="可选评语")


class ReviewSubmission(BaseModel):
    scores: list[ReviewScoreItem]


class JobStatusResponse(BaseModel):
    job_id: str
    state: str  # pending | executing | judging | completed
    progress: str | None = None
    model_a: str | None = None
    model_b: str | None = None
    created_at: str | None = None


class ReportResponse(BaseModel):
    job_id: str
    config: dict
    tasks: dict
    answers_a: dict
    answers_b: dict
    verdict: dict
    report: dict | None = None


class DatasetInfo(BaseModel):
    name: str
    description: str
    task_count: int
    dimensions: list[str]
    version: str = "v1"
    source: str = "upload"
    type_counts: dict[str, int] = {}
    created_at: str


class ModelRegisterRequest(BaseModel):
    """模型配置库注册请求（迭代一）。Key 仅存进程内存，不落盘。"""
    name: str = Field(..., min_length=1, max_length=200, description="模型配置名称（唯一）")
    url: str = Field(..., max_length=500, description="API base URL，如 https://api.example.com/v1")
    key: str | None = Field(None, description="API Key（可选；仅存内存，重启需补录）")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(4096, ge=1, le=128000)
    top_p: float | None = Field(None, ge=0, le=1)


class ReviewConfig(BaseModel):
    """AI 评审配置（迭代二；迭代三扩展 hybrid）。评审模型独立于被测模型，
    Key 仅存内存不落盘。

    mode:
      - pure_human：纯人工双盲评审（默认，保持现状）
      - pure_agent：作答完成后由评审模型自动评审并生成报告
      - hybrid（迭代三）：Agent 预评全部题目 + 分歧/低分 Top-K 转人工复核，
        人工分覆盖被选题（未复核题沿用 Agent 分）
    k_top_human：hybrid 下转人工复核的题数上限（k==0 时直通 completed，
    等价纯人工结果的 Agent 预评展示；非 hybrid 忽略）。
    fail_open：评审模型全部失败时降级为人工评审（True）或任务置 error（False）。
    """
    mode: str = Field("pure_human", pattern=r"^(pure_human|pure_agent|hybrid)$",
                      description="评审方式：pure_human / pure_agent / hybrid")
    judge: ModelConfig | None = Field(None, description="评审模型配置（pure_agent/hybrid 必填）")
    fail_open: bool = Field(False, description="评审全败时降级人工评审")
    k_top_human: int = Field(5, ge=0, le=20,
                             description="hybrid 转人工复核题数上限（0=不复核直通）")


class GoldScoreItem(BaseModel):
    """金标集中单条打分（迭代三）：某模型在某题上的权威分。"""
    task_id: str = Field(..., min_length=1, max_length=200)
    model_name: str = Field(..., min_length=1, max_length=200)
    score: float = Field(..., ge=0, le=100, description="权威分（0-100）")
    note: str = Field("", max_length=1000)


class GoldSetRequest(BaseModel):
    """金标集（迭代三）：name 为集名（如 demo / qwen3-vl），items 为打分条目。"""
    name: str = Field(..., min_length=1, max_length=100)
    items: list[GoldScoreItem] = Field(default_factory=list, max_length=5000)


class BudgetConfig(BaseModel):
    """预算熔断（迭代二）。预估 token 超限时中止或提醒。

    mode:
      - warn（默认）：启动时仅提醒，运行期超限发一次 budget_warning 事件
      - hard：启动时预估超限直接 400 拒绝
    max_tokens=0 表示不限制（向后兼容）。
    """
    max_tokens: int = Field(0, ge=0, description="预估 token 上限，0=不限制")
    mode: str = Field("warn", pattern=r"^(warn|hard)$",
                      description="warn=提醒不中断 / hard=启动时超限拒绝")


class EmbeddingConfig(BaseModel):
    """embedding provider 配置（迭代二，语义相似度指标用）。Key 仅存内存。

    provider:
      - None / auto（默认）：env MODEL_DUEL_EMBEDDING_* 存在 → external；
        否则本地 BGE 可导入 → local_bge；再否则 offline 字符 n-gram 兜底
      - external：OpenAI 兼容 embedding API（页面可覆盖 env）
      - local_bge：本地 BGE 模型（懒加载 onnxruntime，未安装报错）
      - offline：纯 Python 字符 n-gram 余弦（零依赖，确定性可测）
    """
    provider: str | None = Field(None, description="external / local_bge / offline / auto(None)")
    url: str | None = Field(None, max_length=500, description="embedding API base URL")
    key: str | None = Field(None, description="embedding API Key（仅存内存）")
    name: str | None = Field(None, description="embedding 模型名")


class GenerateRequest(BaseModel):
    """LLM 出题请求（迭代四）。gen_config 必填（出题模型，Key 仅存内存）。

    target_dataset：审核通过后追加的目标数据集（缺省在审核通过时新建）。
    dimension 为空表示由前端/调用方指定（后端必填校验在路由层完成）。
    options: {cot: bool, few_shots: bool, with_context: bool}
    """
    gen_config: ModelConfig | None = Field(None, description="出题模型配置（Key 仅内存）")
    target_dataset: str | None = Field(None, max_length=200, description="审核入库目标数据集")
    task_type: str = Field("判别式", pattern=r"^(判别式|生成式)$")
    dimension: str | None = Field(None, max_length=100, description="目标维度（空=随机）")
    count: int = Field(5, ge=1, le=20, description="生成题数（1-20，默认 5）")
    options: dict = Field(default_factory=dict, description="出题选项（cot/few_shots/with_context）")


class ReviewDecisionRequest(BaseModel):
    """出题审核提交（迭代四）。action=approve 可选带 edits 覆盖字段（白名单）。"""
    action: str = Field(..., pattern=r"^(approve|reject)$")
    edits: dict | None = Field(None, description="人工编辑字段（approve 时生效，字段白名单）")


class PerturbRequest(BaseModel):
    """对抗扰动评测请求（迭代六）。model 为被测模型（Key 仅内存）。

    modes ⊆ {改写, 噪声注入, 属性扰动-性别, 属性扰动-地域, 属性扰动-文化}；
    intensities 缺省用各模式默认梯度；judge 为生成式题单臂评审配置
    （缺省时生成式题得分 N/A，仅输出文本指标与一致性）。
    """
    model_config = {"protected_namespaces": ()}
    model: ModelConfig = Field(..., description="被测模型配置（Key 仅内存）")
    dataset_name: str = Field(..., max_length=200, description="评测集名称（必填）")
    modes: list[str] = Field(default_factory=lambda: ["改写", "噪声注入"],
                             description="扰动模式列表")
    intensities: dict[str, list[float]] | None = Field(
        None, description="各模式强度梯度，缺省用默认梯度")
    seed: int | None = Field(None, description="扰动随机种子（确定性可复现）")
    judge: ModelConfig | None = Field(None, description="单臂评审模型（生成式题打分）")
    prompt_strategy: str = Field("cot", pattern=r"^(cot|direct|fewshot)$",
                                 description="执行侧提示策略")


class LeaderboardRequest(BaseModel):
    """排行榜创建请求（迭代六）：由 N 个已完成 job（同一评测集）聚合。"""
    name: str | None = Field(None, max_length=200, description="排行榜名称（可选）")
    job_ids: list[str] = Field(..., min_length=1, max_length=20,
                               description="参与聚合的已完成 job_id 列表")


class BenchmarkRequest(BaseModel):
    """benchmark 批次请求（迭代七）：1 任务集 × N 模型 × M 轮。

    model_ids 引用模型配置库（Key 取进程内存，未补录 400 提示）；
    每模型一个执行单元，单臂评审（判别式指标分 + 生成式单臂 rubric）。
    """
    model_config = {"protected_namespaces": ()}
    dataset_name: str = Field(..., max_length=200, description="评测集名称（必填）")
    model_ids: list[str] = Field(..., min_length=2, max_length=20,
                                 description="模型配置库 id 列表（N≥2）")
    rounds: int = Field(1, ge=1, le=20, description="每模型重复轮数 M")
    priority: int = Field(0, ge=-10, le=10, description="批次任务优先级（越高越先调度）")
    name: str | None = Field(None, max_length=200, description="批次名称（可选）")
    review: ReviewConfig | None = Field(None, description="评审配置（judge 用于生成式题单臂）")
    prompt_strategy: str = Field("cot", pattern=r"^(cot|direct|fewshot)$")
    code_verify_mode: str = Field("off", pattern=r"^(off|native-sandbox)$")
    budget: BudgetConfig | None = Field(None, description="预算熔断（按 N 模型放大预估）")
    embedding: EmbeddingConfig | None = Field(None, description="embedding 配置")


class PriorityRequest(BaseModel):
    """任务优先级调整请求（迭代七，仅排队中可改）。"""
    priority: int = Field(0, ge=-10, le=10, description="新优先级（-10..10）")
