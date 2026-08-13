# -*- coding: utf-8 -*-
"""模型配置库（迭代一）：注册常用模型配置，评测表单一键填入。

- 持久化：webapp/data/models/<safe_id>.json，仅存脱敏 key_masked（"***"）；
- API Key 仅存进程内存表，重启后清空（前端提示补录，不落盘）；
- 注册时校验 upstream URL（SSRF 防护，复用 validate_upstream_url）。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.ssrf import validate_upstream_url, UpstreamUrlError

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# 进程内存 Key 表：{id: key}，重启清空
_MODEL_KEYS: dict[str, str] = {}

# 与 storage._safe_dataset_name 相同的文件名消毒规则
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

MAX_MODEL_NAME_LEN = 200
MAX_MODEL_URL_LEN = 500


class ModelRegistryError(ValueError):
    """模型配置库错误（重名/非法 URL/未找到），由 API 层渲染为 400/404。"""


def _safe_model_id(name: str) -> str:
    safe = _INVALID_FS_CHARS.sub("_", str(name)).strip().rstrip(".").strip()
    return safe or "model"


def _ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(model_id: str) -> Path:
    return MODELS_DIR / f"{_safe_model_id(model_id)}.json"


def register(
    name: str,
    url: str,
    key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float | None = None,
) -> dict[str, Any]:
    """注册模型配置。URL 必须通过 SSRF 校验；Key 仅存内存。

    Returns:
        脱敏后的模型信息（含 has_key）。
    """
    name = (name or "").strip()
    url = (url or "").strip()
    if not name:
        raise ModelRegistryError("模型名称不能为空")
    if len(name) > MAX_MODEL_NAME_LEN:
        raise ModelRegistryError(f"模型名称长度超过上限 {MAX_MODEL_NAME_LEN}")
    if not url:
        raise ModelRegistryError("API URL 不能为空")
    if len(url) > MAX_MODEL_URL_LEN:
        raise ModelRegistryError(f"API URL 长度超过上限 {MAX_MODEL_URL_LEN}")
    try:
        validate_upstream_url(url)
    except UpstreamUrlError as e:
        raise ModelRegistryError(f"目标地址校验失败: {e}")

    model_id = _safe_model_id(name)
    p = _model_path(model_id)
    if p.exists():
        raise ModelRegistryError(f"模型配置已存在：{name}")

    _ensure_models_dir()
    info = {
        "id": model_id,
        "name": name,
        "url": url,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "key_masked": "***",
        "has_key": bool(key),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    if key:
        _MODEL_KEYS[model_id] = key
    return {**info, "has_key": model_id in _MODEL_KEYS}


def list_models() -> list[dict[str, Any]]:
    """列出全部模型配置（脱敏）。"""
    if not MODELS_DIR.exists():
        return []
    result = []
    for p in sorted(MODELS_DIR.glob("*.json")):
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            info.setdefault("id", p.stem)
            info.setdefault("name", p.stem)
            info["has_key"] = info.get("id") in _MODEL_KEYS
            info["key_masked"] = "***"
            result.append(info)
        except Exception:
            continue
    return result


def get_model(model_id: str) -> dict[str, Any] | None:
    """按 id 取模型配置（脱敏；has_key 反映内存中是否有 Key）。"""
    p = _model_path(model_id)
    if not p.exists():
        return None
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    info.setdefault("id", p.stem)
    info["has_key"] = info.get("id") in _MODEL_KEYS
    info["key_masked"] = "***"
    return info


def get_key(model_id: str) -> str | None:
    """取内存中的 API Key（未注册/已补录为空的返回 None）。"""
    if not _model_path(model_id).exists():
        return None
    return _MODEL_KEYS.get(_safe_model_id(model_id))


def delete_model(model_id: str) -> bool:
    """删除模型配置（同时清空内存 Key）。"""
    p = _model_path(model_id)
    if not p.exists():
        return False
    _MODEL_KEYS.pop(_safe_model_id(model_id), None)
    try:
        p.unlink()
        return True
    except OSError:
        return False


def clear_memory_keys() -> None:
    """清空内存 Key 表（服务重启时调用；测试隔离用）。"""
    _MODEL_KEYS.clear()
