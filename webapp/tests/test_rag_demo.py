# -*- coding: utf-8 -*-
"""D2 内置 RAG 演示集（迭代四）：无条件可用的带 context 评测集。

- ensure_demo_rag_dataset：数据集目录无 rag_demo 时写入（source=builtin_demo）；
- 幂等：已存在同名数据集（含用户上传覆盖）时跳过，不回退；
- 三道生成式题均携带 context（参考文档），可通过 validator；
- 与 lifespan 接入：TestClient 生命周期触发种子。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage
from backend.engine.rag_demo import DEMO_NAME, ensure_demo_rag_dataset
from backend.engine.datasets import validate_standard_dataset


@pytest.fixture(autouse=True)
def _isolate_datasets(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DATASETS_DIR", tmp_path / "datasets")


def test_seed_when_absent():
    ensure_demo_rag_dataset()
    ds = storage.load_dataset(DEMO_NAME)
    assert ds is not None
    assert ds["source"] == "builtin_demo"
    assert ds["version"] == "v1"
    assert len(ds["tasks"]) == 3


def test_seed_tasks_valid():
    ensure_demo_rag_dataset()
    ds = storage.load_dataset(DEMO_NAME)
    validated = validate_standard_dataset(ds)
    assert len(validated["tasks"]) == 3
    for t in validated["tasks"]:
        assert t["type"] == "生成式"
        assert t["rubric_note"]
        assert len(t["context"]) >= 300  # 参考文档 300~500 字
        assert t["dimension"] == "知识能力"


def test_seed_idempotent_after_user_override():
    ensure_demo_rag_dataset()
    storage.save_dataset(DEMO_NAME, {
        "name": DEMO_NAME, "description": "用户覆盖", "source": "upload",
        "tasks": [{"id": "T1", "dimension": "知识能力", "prompt": "自定义题",
                   "expected": "x", "rubric_note": "r", "type": "判别式"}],
    })
    ensure_demo_rag_dataset()  # 再次调用不覆盖
    ds = storage.load_dataset(DEMO_NAME)
    assert ds["source"] == "upload"
    assert ds["tasks"][0]["prompt"] == "自定义题"


def test_lifespan_seeds_via_client():
    with TestClient(main_module.app) as client:
        resp = client.get("/api/datasets")
        assert resp.status_code == 200
    assert any(d["name"] == DEMO_NAME for d in resp.json()["datasets"])


def test_rag_demo_answerable_prompts_reference_doc():
    """题目要求仅依据参考文档作答：验证题面与 context 语义连贯（含文档关键词）。"""
    ensure_demo_rag_dataset()
    ds = storage.load_dataset(DEMO_NAME)
    for t in ds["tasks"]:
        assert "参考文档" in t["prompt"]