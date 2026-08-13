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
    """AI 评审配置（迭代二）。评审模型独立于被测模型，Key 仅存内存不落盘。

    mode:
      - pure_human：纯人工双盲评审（默认，保持现状）
      - pure_agent：作答完成后由评审模型自动评审并生成报告
      - hybrid（迭代三）：Agent 预评 + 人工复核
    fail_open：评审模型全部失败时降级为人工评审（True）或任务置 error（False）。
    """
    mode: str = Field("pure_human", pattern=r"^(pure_human|pure_agent)$",
                      description="评审方式：pure_human / pure_agent（hybrid 迭代三）")
    judge: ModelConfig | None = Field(None, description="评审模型配置（pure_agent 必填）")
    fail_open: bool = Field(False, description="评审全败时降级人工评审")


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
