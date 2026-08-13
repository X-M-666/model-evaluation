# -*- coding: utf-8 -*-
"""embedding provider 抽象（迭代二）：语义相似度指标的向量来源。

- external：OpenAI 兼容 embedding API。配置来源：环境变量
  MODEL_DUEL_EMBEDDING_URL/KEY/NAME 默认 + 页面/API 配置可覆盖
  （Key 仅存内存，不落盘）；调用走 SSRF 校验客户端（build_upstream_client）。
- local_bge：本地 BGE 模型（懒加载 onnxruntime，首调才 import；
  未安装时返回明确错误提示，语义指标降级不中断评测）。
- offline：纯 Python 字符 n-gram 余弦（零依赖，缺省离线可用，确定性可测）。

解析优先级（provider=None/auto）：env 存在 → external；
本地 BGE 可导入 → local_bge；否则 offline。
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

ENV_URL = "MODEL_DUEL_EMBEDDING_URL"
ENV_KEY = "MODEL_DUEL_EMBEDDING_KEY"
ENV_NAME = "MODEL_DUEL_EMBEDDING_NAME"

# n-gram 维度（字符级，中文友好）
NGRAM_N = 2


def ngram_vec(text: str, n: int = NGRAM_N) -> dict[str, int]:
    """字符 n-gram 计数向量（去空白；纯 Python，确定性）。"""
    chars = re.sub(r"\s+", "", text or "")
    vec: dict[str, int] = {}
    if not chars:
        return vec
    for i in range(len(chars) - n + 1):
        g = chars[i : i + n]
        vec[g] = vec.get(g, 0) + 1
    return vec


def _dot(a, b) -> float:
    return sum(a.get(k, 0) * v for k, v in b.items())


def _norm_len(v) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def cosine(a: dict | list, b: dict | list) -> float:
    """余弦相似度。支持 dict 计数向量或 list 数值向量；空向量返回 0.0。"""
    if isinstance(a, list) and isinstance(b, list):
        if not a or len(a) != len(b):
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        la = math.sqrt(sum(x * x for x in a))
        lb = math.sqrt(sum(x * x for x in b))
        return num / (la * lb) if la and lb else 0.0
    da, db = dict(a or {}), dict(b or {})
    if not da or not db:
        return 0.0
    return _dot(da, db) / (_norm_len(da) * _norm_len(db))


def _bge_importable() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_provider(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """解析实际生效的 provider。

    Returns:
        {"kind": "external"|"local_bge"|"offline", "cfg": dict, "error": str | None}
    """
    cfg = config or {}
    provider = (cfg.get("provider") or "auto").strip().lower()

    if provider == "external":
        return _resolve_external(cfg)
    if provider == "local_bge":
        if not _bge_importable():
            return {"kind": "local_bge", "cfg": cfg,
                    "error": "本地 BGE 需要 onnxruntime，未安装；请切换 external/offline"}
        return {"kind": "local_bge", "cfg": cfg}
    if provider == "offline":
        return {"kind": "offline", "cfg": cfg}

    # auto：env → local_bge → offline
    env_cfg = {
        "url": os.environ.get(ENV_URL, "").strip(),
        "key": os.environ.get(ENV_KEY, "").strip() or None,
        "name": os.environ.get(ENV_NAME, "").strip() or None,
    }
    if env_cfg["url"]:
        return _resolve_external({**cfg, **{k: v for k, v in env_cfg.items() if v}})
    if _bge_importable():
        return {"kind": "local_bge", "cfg": cfg}
    return {"kind": "offline", "cfg": cfg}


def _resolve_external(cfg: dict[str, Any]) -> dict[str, Any]:
    if not (cfg.get("url") or "").strip():
        return {"kind": "external", "cfg": cfg,
                "error": "external provider 缺少 URL（配置或 MODEL_DUEL_EMBEDDING_URL）"}
    return {"kind": "external", "cfg": cfg}


async def embed_texts(
    kind: str,
    cfg: dict[str, Any],
    client: Any,
    texts: list[str],
) -> list[list[float]] | None:
    """external 时调用 {url}/embeddings 返回向量列表；失败返回 None（降级 n-gram）。

    仅 external 走网络（复用调用方已建的 SSRF 校验客户端）；其余 kind 返回 None。
    """
    if kind != "external":
        return None
    url = (cfg.get("url") or "").rstrip("/") + "/embeddings"
    payload: dict[str, Any] = {
        "model": cfg.get("name") or "text-embedding-3-small",
        "input": texts,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.get("key"):
        headers["Authorization"] = f"Bearer {cfg['key']}"
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=60)
        body = resp.json()
        if resp.status_code >= 400:
            return None
        return [item["embedding"] for item in body["data"]]
    except Exception:
        return None


def local_bge_embed(texts: list[str]) -> list[list[float]] | None:
    """本地 BGE 懒加载嵌入（首调才 import onnxruntime/模型，不中断评测）。

    onnxruntime 未安装 / 模型缺失 / 运行失败 → None（调用方降级 n-gram 兜底）。
    """
    try:
        from backend.engine.bge_local import embed_local

        return embed_local(texts)
    except Exception:
        return None
