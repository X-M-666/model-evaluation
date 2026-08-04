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


class StartResponse(BaseModel):
    job_id: str


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
    created_at: str
