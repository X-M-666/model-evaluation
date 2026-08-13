# -*- coding: utf-8 -*-
"""统一敏感信息脱敏：持久化与对外响应前必须经过本模块函数，保证 API Key 不落盘、不外泄。"""
from __future__ import annotations

import copy
from typing import Any

SENSITIVE_KEYS = frozenset({"key", "api_key", "apikey", "token", "authorization", "secret"})


def redact_sensitive(obj: Any) -> Any:
    """递归深拷贝并删除嵌套结构中所有命名的敏感字段（如 key/api_key/token）。

    对 dict/list/标量通用且幂等，作为读层/响应层的纵深防御。
    """
    if isinstance(obj, dict):
        return {
            k: redact_sensitive(v)
            for k, v in obj.items()
            if not _is_sensitive_key(k)
        }
    if isinstance(obj, list):
        return [redact_sensitive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_sensitive(v) for v in obj)
    return copy.deepcopy(obj)


def _is_sensitive_key(name: str) -> bool:
    return name.lower() in SENSITIVE_KEYS


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """返回不包含模型/辅助配置 API Key 的配置副本，保留报告展示所需的全部字段。"""
    safe = copy.deepcopy(config)
    for slot in ("model_a", "model_b", "judge", "embedding"):
        model = safe.get(slot)
        if isinstance(model, dict):
            model.pop("key", None)
    review = safe.get("review")
    if isinstance(review, dict):
        judge = review.get("judge")
        if isinstance(judge, dict):
            judge.pop("key", None)
    return safe
