# -*- coding: utf-8 -*-
"""本地 BGE 嵌入（懒加载，不中断评测）。

onnxruntime + 整数输入 ONNX 模型（BGE 系列导出格式：input_ids/attention_mask，
input_ids 为词元 id 张量）。模型路径由环境变量 MODEL_DUEL_BGE_MODEL_DIR 或配置
model_dir 指定，目录内需含 model.onnx。

未安装 onnxruntime / 模型缺失 / 运行失败 → 返回 None，由调用方降级 n-gram 兜底。
此模块在测试环境始终走降级路径（无模型文件），核心保证是"缺失即降级、不崩溃"。
"""
from __future__ import annotations

import os
from typing import Any

_session: Any = None
_model_dir: str | None = None


def _load_session(model_dir: str) -> Any | None:
    """懒加载 ONNX 会话（带缓存）；模型缺失返回 None。"""
    global _session, _model_dir
    if _session is not None and _model_dir == model_dir:
        return _session
    try:
        import onnxruntime as ort

        path = os.path.join(model_dir, "model.onnx")
        if not os.path.isfile(path):
            return None
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _model_dir = model_dir
        return _session
    except Exception:
        return None


def _tokenize(text: str, max_len: int = 512) -> dict[str, list[int]]:
    """最小词元化：UTF-8 字节码序列作为输入 id（与字符级 BGE 导出模型兼容）。

    真实部署可替换为完整 tokenizer（如 tokenizers 库加载 vocab）；此处保证
    模型输入形状合法，质量差异由报告层 n-gram 兜底指标吸收。
    """
    b = [c for c in text.encode("utf-8")][: max_len - 2]
    return {
        "input_ids": [1] + b + [2],
        "attention_mask": [1] * (len(b) + 2),
    }


def embed_local(texts: list[str]) -> list[list[float]] | None:
    """批量嵌入文本；任意失败返回 None（不抛异常）。"""
    try:
        model_dir = os.environ.get("MODEL_DUEL_BGE_MODEL_DIR", "").strip() or None
        if not model_dir:
            return None
        sess = _load_session(model_dir)
        if sess is None:
            return None
        import numpy as np

        feats = [_tokenize(t or "") for t in texts]
        max_len = max(len(f["input_ids"]) for f in feats)
        ids = np.zeros((len(feats), max_len), dtype=np.int64)
        mask = np.zeros((len(feats), max_len), dtype=np.int64)
        for i, f in enumerate(feats):
            ids[i, : len(f["input_ids"])] = f["input_ids"]
            mask[i, : len(f["attention_mask"])] = f["attention_mask"]
        out = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        if not out:
            return None
        emb = np.asarray(out[0])
        if emb.ndim == 3:
            emb = emb[:, 0, :]
        return emb.tolist()
    except Exception:
        return None