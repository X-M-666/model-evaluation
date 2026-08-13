# -*- coding: utf-8 -*-
"""模型配置库单元与 API 集成测试（迭代一）。

保护不变量：
- 配置元信息落盘、API Key 仅存进程内存（重启需补录）
- 注册/删除/列举幂等语义；重名与 SSRF URL 拒绝
- has_key 反映内存 Key 状态；delete 清 Key 并落盘移除
- 审计事件 model_registered / model_deleted
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import models_registry
from backend.models_registry import (
    ModelRegistryError,
    clear_memory_keys,
    delete_model,
    get_key,
    get_model,
    list_models,
    register,
)

VALID_URL = "https://8.8.8.8/v1"
PRIVATE_URL = "http://10.0.0.1/v1"


@pytest.fixture
def client():
    return TestClient(main_module.app)


def _fresh_registry():
    for m in list(list_models()):
        delete_model(m["id"])
    clear_memory_keys()


# ---------------- 单元：注册 / 列举 / 删除 ----------------

def test_register_with_key_masked_on_disk():
    _fresh_registry()
    reg = register(name="模型甲", url=VALID_URL, key="sk-123", temperature=0.3)
    assert reg["has_key"] is True
    assert get_key(reg["id"]) == "sk-123"
    raw = (models_registry.MODELS_DIR / f"{reg['id']}.json").read_text(encoding="utf-8")
    assert "sk-123" not in raw
    assert json.loads(raw)["key_masked"] == "***"
    assert json.loads(raw)["name"] == "模型甲"
    assert json.loads(raw)["temperature"] == 0.3


def test_register_without_key():
    _fresh_registry()
    reg = register(name="模型乙", url=VALID_URL)
    assert reg["has_key"] is False
    assert get_key(reg["id"]) is None


def test_register_duplicate_name_rejected():
    _fresh_registry()
    register(name="同名", url=VALID_URL)
    with pytest.raises(ModelRegistryError, match="已存在"):
        register(name="同名", url=VALID_URL)


def test_register_public_url_ssrf_rejected():
    _fresh_registry()
    with pytest.raises(ModelRegistryError, match="目标地址校验失败"):
        register(name="外网", url=PRIVATE_URL)


def test_register_blank_name_rejected():
    _fresh_registry()
    with pytest.raises(ModelRegistryError):
        register(name="   ", url=VALID_URL)


def test_delete_removes_disk_and_key():
    _fresh_registry()
    reg = register(name="待删", url=VALID_URL, key="k1")
    assert (models_registry.MODELS_DIR / f"{reg['id']}.json").exists()
    delete_model(reg["id"])
    assert not (models_registry.MODELS_DIR / f"{reg['id']}.json").exists()
    assert get_key(reg["id"]) is None
    assert get_model(reg["id"]) is None


def test_clear_memory_keys_only():
    _fresh_registry()
    mid = register(name="清Key", url=VALID_URL, key="keep-on-disk")["id"]
    assert get_key(mid) == "keep-on-disk"
    clear_memory_keys()
    m = get_model(mid)
    assert m is not None
    assert m["has_key"] is False
    assert get_key(mid) is None


def test_list_models_never_exposes_key():
    _fresh_registry()
    register(name="脱敏", url=VALID_URL, key="sk-top-secret")
    for m in list_models():
        assert "key" not in m
        assert "sk-" not in json.dumps(m, ensure_ascii=False)


def test_list_has_key_reflects_memory_only():
    _fresh_registry()
    mid = register(name="重启语义", url=VALID_URL, key="temporary")["id"]
    assert list_models()[0]["has_key"] is True
    clear_memory_keys()
    assert get_model(mid)["has_key"] is False
    assert list_models()[0]["has_key"] is False


def test_safe_model_id_sanitizes():
    _fresh_registry()
    reg = register(name="a/b..\\c", url=VALID_URL)
    assert "/" not in reg["id"] and "\\" not in reg["id"]
    p = (models_registry.MODELS_DIR / f"{reg['id']}.json").resolve()
    assert p.is_relative_to(models_registry.MODELS_DIR.resolve())


# ---------------- API 集成 ----------------

def test_api_models_crud(client):
    _fresh_registry()
    r = client.post("/api/models", json={"name": "API模型", "url": VALID_URL, "key": "api-key-1"})
    assert r.status_code == 200, r.text
    mid = r.json()["model"]["id"]
    assert r.json()["model"]["has_key"] is True
    assert "api-key-1" not in json.dumps(r.json(), ensure_ascii=False)

    r = client.get("/api/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["models"]]
    assert mid in ids

    r = client.get(f"/api/models/{mid}")
    assert r.status_code == 200
    assert r.json()["has_key"] is True
    # 单条读取供「填入」流程使用：附带进程内存中的 Key（列表接口不含）
    assert r.json()["key"] == "api-key-1"
    r = client.get("/api/models")
    assert "api-key-1" not in json.dumps(r.json(), ensure_ascii=False)

    r = client.delete(f"/api/models/{mid}")
    assert r.status_code == 200
    r = client.get("/api/models")
    assert mid not in [m["id"] for m in r.json()["models"]]


def test_api_models_duplicate_400(client):
    _fresh_registry()
    body = {"name": "重复模型", "url": VALID_URL}
    assert client.post("/api/models", json=body).status_code == 200
    r = client.post("/api/models", json=body)
    assert r.status_code == 400
    assert "已存在" in r.json()["detail"]


def test_api_models_ssrf_400(client):
    _fresh_registry()
    r = client.post("/api/models", json={"name": "外网", "url": PRIVATE_URL})
    assert r.status_code == 400
    assert "目标地址" in r.json()["detail"]


def test_api_models_delete_missing_404(client):
    _fresh_registry()
    assert client.delete("/api/models/nope").status_code == 404


def test_api_stats_saturation_empty(client):
    r = client.get("/api/stats/saturation")
    assert r.status_code == 200
    data = r.json()
    assert data["jobs"] == []
    # 迭代五：trend 段（空态）与旧 jobs 字段共存
    assert data["trend"]["available"] is False
    assert data["trend"]["datasets"] == {}


def test_models_api_audited(client):
    _fresh_registry()
    from backend import audit
    r = client.post("/api/models", json={"name": "审计模型", "url": VALID_URL})
    assert r.status_code == 200, r.text
    mid = r.json()["model"]["id"]
    client.delete(f"/api/models/{mid}")
    events = {e["event"]: e for e in audit.read_events()}
    assert "model_registered" in events and "model_deleted" in events
    assert events["model_deleted"]["target"] == mid
