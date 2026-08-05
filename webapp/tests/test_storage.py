# -*- coding: utf-8 -*-
"""存储层单元测试（issue #11）。

覆盖：配置脱敏落盘、文件状态推断 _job_state、损坏 JSON 容错、
数据集名称消毒 _safe_dataset_name、数据集与任务删除/列举。
（目录隔离由 conftest 的 module 级 fixture 提供。）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend import storage

JOB_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")


def _write(job_id: str, name: str, content: str) -> Path:
    p = storage.BASE_DIR / job_id / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------- save_config 脱敏落盘 ----------------

def test_save_config_masks_key_keeps_display_fields():
    job_id = "cfg1"
    storage.save_config(job_id, {
        "model_a": {"name": "A", "url": "https://a/v1", "key": "secret-a",
                    "temperature": 0.3, "max_tokens": 2048, "top_p": 0.9},
        "model_b": {"name": "B", "url": "https://b/v1", "key": "secret-b"},
        "dims": ["知识能力"], "seed": 42, "dataset_name": "ds",
        "repeat_n": 3, "code_verify_mode": "native-sandbox",
    })
    raw = (storage.BASE_DIR / job_id / "config.json").read_text(encoding="utf-8")
    assert "secret-a" not in raw and "secret-b" not in raw
    cfg = json.loads(raw)
    assert "key" not in cfg["model_a"] and "key" not in cfg["model_b"]
    assert cfg["model_a"]["key_masked"] == "***"
    assert cfg["model_b"]["key_masked"] == "***"
    assert cfg["model_a_name"] == "A" and cfg["model_b_name"] == "B"
    assert cfg["model_a"]["url"] == "https://a/v1"
    assert cfg["model_a"]["temperature"] == 0.3
    assert cfg["model_a"]["max_tokens"] == 2048
    assert cfg["model_a"]["top_p"] == 0.9
    assert cfg["seed"] == 42 and cfg["dataset_name"] == "ds"
    assert cfg["repeat_n"] == 3 and cfg["code_verify_mode"] == "native-sandbox"


def test_save_config_defaults():
    storage.save_config("cfg2", {"model_a": {}, "model_b": {}})
    cfg = json.loads((storage.BASE_DIR / "cfg2" / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_a"]["name"] == "?"
    assert cfg["model_a"]["temperature"] == 0.7
    assert cfg["model_a"]["max_tokens"] == 4096
    assert cfg["repeat_n"] == 1


# ---------------- _job_state 状态推断 ----------------

def test_state_pending_empty_dir():
    d = storage.BASE_DIR / "st-empty"
    d.mkdir(parents=True, exist_ok=True)
    assert storage._job_state(d) == "pending"


def test_state_executing_tasks_only():
    _write("st-exe", "tasks.json", "{}")
    assert storage._job_state(storage.BASE_DIR / "st-exe") == "executing"


def test_state_executing_answers_a_only():
    """仅有单侧答卷不算可评审：按当前实现落入 pending/executing 区间（锁定）。"""
    _write("st-a", "answers-a.json", "{}")
    assert storage._job_state(storage.BASE_DIR / "st-a") == "pending"


def test_state_reviewing_both_answers():
    _write("st-rev", "answers-a.json", "{}")
    _write("st-rev", "answers-b.json", "{}")
    assert storage._job_state(storage.BASE_DIR / "st-rev") == "reviewing"


def test_state_judging_with_verdict():
    _write("st-judge", "answers-a.json", "{}")
    _write("st-judge", "answers-b.json", "{}")
    _write("st-judge", "verdict.json", "{}")
    assert storage._job_state(storage.BASE_DIR / "st-judge") == "judging"


def test_state_completed_with_report():
    _write("st-done", "answers-a.json", "{}")
    _write("st-done", "answers-b.json", "{}")
    _write("st-done", "verdict.json", "{}")
    _write("st-done", "report.json", "{}")
    assert storage._job_state(storage.BASE_DIR / "st-done") == "completed"


def test_state_error_takes_priority_even_with_report():
    _write("st-err", "report.json", "{}")
    _write("st-err", "error.json", '{"error":"boom"}')
    assert storage._job_state(storage.BASE_DIR / "st-err") == "error"


# ---------------- 损坏 JSON 容错 ----------------

def test_load_reveal_missing_returns_none():
    assert storage.load_reveal("noreveal") is None


def test_load_reveal_corrupt_returns_none():
    _write("cor-reveal", "reveal.json", "{broken")
    assert storage.load_reveal("cor-reveal") is None


def test_load_review_corrupt_returns_none():
    _write("cor-review", "review.json", "not json")
    assert storage.load_review("cor-review") is None


def test_load_reveal_roundtrip():
    storage.save_reveal("rev-ok", {"rounds": [{"answer_x": "a", "answer_y": "b"}]})
    assert storage.load_reveal("rev-ok")["rounds"][0]["answer_x"] == "a"


def test_get_job_files_corrupt_entry_returns_none():
    _write("cor-files", "verdict.json", "{broken")
    _write("cor-files", "report.json", "{}")
    files = storage.get_job_files("cor-files")
    assert files["verdict.json"] is None
    assert files["report.json"] == {}


def test_get_job_files_includes_round_files():
    _write("rf", "answers-a-r1.json", "{}")
    _write("rf", "answers-b-r2.json", "{}")
    files = storage.get_job_files("rf")
    assert "answers-a-r1.json" in files
    assert "answers-b-r2.json" in files


def test_get_job_files_missing_job_returns_none():
    assert storage.get_job_files("no-such-job") is None


# ---------------- _safe_dataset_name 消毒 ----------------

def test_safe_name_replaces_illegal_fs_chars():
    safe = storage._safe_dataset_name('a<b>c:d"e/f\\g|h?i*j')
    assert safe == "a_b_c_d_e_f_g_h_i_j"


def test_safe_name_strips_trailing_dots_and_spaces():
    assert storage._safe_dataset_name(" 名字. ") == "名字"
    assert storage._safe_dataset_name("..") == "dataset"


def test_safe_name_empty_falls_back():
    assert storage._safe_dataset_name("") == "dataset"
    assert storage._safe_dataset_name("...") == "dataset"
    assert storage._safe_dataset_name(None) == "None"


def test_safe_name_blocks_path_traversal():
    """路径分隔符被替换，消毒后是普通文件名，落盘位置始终在数据集目录内。"""
    safe = storage._safe_dataset_name("..\\..\\etc")
    assert "\\" not in safe and "/" not in safe
    assert safe == ".._.._etc"
    p = storage.save_dataset("..\\..\\etc", {"name": "x"})
    assert storage.DATASETS_DIR in p.parents
    assert p.name == f"{safe}.json"


def test_safe_name_control_chars_replaced():
    assert storage._safe_dataset_name("a\x00b\x1fc") == "a_b_c"


# ---------------- 数据集增删查 ----------------

def test_dataset_save_load_roundtrip():
    data = {"name": "D", "description": "desc", "tasks": [{"prompt": "p"}]}
    storage.save_dataset("D", data)
    loaded = storage.load_dataset("D")
    assert loaded["name"] == "D"
    assert loaded["tasks"][0]["prompt"] == "p"


def test_dataset_load_missing_returns_none():
    assert storage.load_dataset("不存在") is None


def test_dataset_delete_ok_then_missing():
    storage.save_dataset("删", {"name": "删"})
    assert storage.delete_dataset("删") is True
    assert storage.load_dataset("删") is None
    assert storage.delete_dataset("删") is False


def test_dataset_list_summaries():
    storage.save_dataset("A", {"name": "A", "description": "d",
                               "tasks": [{"dimension": "知识"}, {"dimension": "代码"}]})
    storage.save_dataset("B", {"name": "B", "tasks": []})
    summaries = {s["name"]: s for s in storage.list_datasets()}
    assert summaries["A"]["task_count"] == 2
    assert set(summaries["A"]["dimensions"]) == {"知识", "代码"}
    assert summaries["B"]["task_count"] == 0


def test_dataset_list_skips_corrupt_files():
    storage.save_dataset("好", {"name": "好", "tasks": [{"dimension": "知识"}]})
    (storage.DATASETS_DIR / f"{storage._safe_dataset_name('坏')}.json").write_text(
        "{broken", encoding="utf-8")
    names = [s["name"] for s in storage.list_datasets()]
    assert "好" in names and "坏" not in names


# ---------------- 任务删除 ----------------

def test_delete_job_missing_returns_false():
    assert storage.delete_job("不存在") is False


def test_delete_job_removes_directory():
    _write("del-me", "config.json", "{}")
    _write("del-me", "report.json", "{}")
    assert (storage.BASE_DIR / "del-me").exists()
    assert storage.delete_job("del-me") is True
    assert not (storage.BASE_DIR / "del-me").exists()


def test_create_job_id_unique_format():
    ids = {storage.create_job_id() for _ in range(50)}
    assert len(ids) == 50
    for jid in ids:
        assert JOB_ID_RE.match(jid)


def test_save_round_verdicts_persisted():
    storage.save_round_verdicts("rv", [{"round": 1, "x": 1}, {"round": 2, "x": 2}])
    p = storage.BASE_DIR / "rv" / "round-verdicts.json"
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8"))[1]["round"] == 2
