# -*- coding: utf-8 -*-
"""存储层单元测试（issue #11）。

覆盖：配置脱敏落盘、文件状态推断 _job_state、损坏 JSON 容错、
数据集名称消毒 _safe_dataset_name、数据集与任务删除/列举。
（目录隔离由 conftest 的 module 级 fixture 提供。）
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from backend import storage

JOB_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")


def _jid(tag: str) -> str:
    """生成符合 JOB_ID_RE 的确定性 job_id（issue #17 后存储层仅接受系统格式）。"""
    return "20260101_120000_" + hashlib.md5(tag.encode()).hexdigest()[:6]


def _write(job_id: str, name: str, content: str) -> Path:
    p = storage.BASE_DIR / job_id / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------- save_config 脱敏落盘 ----------------

def test_save_config_masks_key_keeps_display_fields():
    job_id = _jid("cfg1")
    storage.save_config(job_id, {
        "model_a": {"name": "A", "url": "https://a/v1", "key": "secret-a",
                    "temperature": 0.3, "max_tokens": 2048, "top_p": 0.9},
        "model_b": {"name": "B", "url": "https://b/v1", "key": "secret-b"},
        "dims": ["知识能力"], "seed": 42, "dataset_name": "ds",
        "repeat_n": 3, "code_verify_mode": "native-sandbox",
    })
    raw = (storage.BASE_DIR / _jid("cfg1") / "config.json").read_text(encoding="utf-8")
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
    storage.save_config(_jid("cfg2"), {"model_a": {}, "model_b": {}})
    cfg = json.loads((storage.BASE_DIR / _jid("cfg2") / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_a"]["name"] == "?"
    assert cfg["model_a"]["temperature"] == 0.7
    assert cfg["model_a"]["max_tokens"] == 4096
    assert cfg["repeat_n"] == 1


# ---------------- _job_state 状态推断 ----------------

def test_state_pending_empty_dir():
    d = storage.BASE_DIR / _jid("st-empty")
    d.mkdir(parents=True, exist_ok=True)
    assert storage._job_state(d) == "pending"


def test_state_executing_tasks_only():
    _write(_jid("st-exe"), "tasks.json", "{}")
    assert storage._job_state(storage.BASE_DIR / _jid("st-exe")) == "executing"


def test_state_executing_answers_a_only():
    """仅有单侧答卷不算可评审：按当前实现落入 pending/executing 区间（锁定）。"""
    _write(_jid("st-a"), "answers-a.json", "{}")
    assert storage._job_state(storage.BASE_DIR / _jid("st-a")) == "pending"


def test_state_queued_config_only():
    """迭代七：仅 config.json（tasks.json 延后落盘）→ queued。"""
    _write(_jid("st-q"), "config.json", "{}")
    assert storage._job_state(storage.BASE_DIR / _jid("st-q")) == "queued"


def test_state_queued_then_executing():
    """落盘 tasks.json 后 queued → executing（排队 → 派发语义）。"""
    jid = _jid("st-q2")
    _write(jid, "config.json", "{}")
    assert storage._job_state(storage.BASE_DIR / jid) == "queued"
    _write(jid, "tasks.json", "{}")
    assert storage._job_state(storage.BASE_DIR / jid) == "executing"


def test_state_reviewing_both_answers():
    _write(_jid("st-rev"), "answers-a.json", "{}")
    _write(_jid("st-rev"), "answers-b.json", "{}")
    assert storage._job_state(storage.BASE_DIR / _jid("st-rev")) == "reviewing"


def test_state_judging_with_verdict():
    _write(_jid("st-judge"), "answers-a.json", "{}")
    _write(_jid("st-judge"), "answers-b.json", "{}")
    _write(_jid("st-judge"), "verdict.json", "{}")
    assert storage._job_state(storage.BASE_DIR / _jid("st-judge")) == "judging"


def test_state_completed_with_report():
    _write(_jid("st-done"), "answers-a.json", "{}")
    _write(_jid("st-done"), "answers-b.json", "{}")
    _write(_jid("st-done"), "verdict.json", "{}")
    _write(_jid("st-done"), "report.json", "{}")
    assert storage._job_state(storage.BASE_DIR / _jid("st-done")) == "completed"


def test_state_error_takes_priority_even_with_report():
    _write(_jid("st-err"), "report.json", "{}")
    _write(_jid("st-err"), "error.json", '{"error":"boom"}')
    assert storage._job_state(storage.BASE_DIR / _jid("st-err")) == "error"


# ---------------- 损坏 JSON 容错 ----------------

def test_load_reveal_missing_returns_none():
    assert storage.load_reveal(_jid("noreveal")) is None


def test_load_reveal_corrupt_returns_none():
    _write(_jid("cor-reveal"), "reveal.json", "{broken")
    assert storage.load_reveal(_jid("cor-reveal")) is None


def test_load_review_corrupt_returns_none():
    _write(_jid("cor-review"), "review.json", "not json")
    assert storage.load_review(_jid("cor-review")) is None


def test_load_reveal_roundtrip():
    storage.save_reveal(_jid("rev-ok"), {"rounds": [{"answer_x": "a", "answer_y": "b"}]})
    assert storage.load_reveal(_jid("rev-ok"))["rounds"][0]["answer_x"] == "a"


def test_get_job_files_corrupt_entry_returns_none():
    _write(_jid("cor-files"), "verdict.json", "{broken")
    _write(_jid("cor-files"), "report.json", "{}")
    files = storage.get_job_files(_jid("cor-files"))
    assert files["verdict.json"] is None
    assert files["report.json"] == {}


def test_get_job_files_includes_round_files():
    _write(_jid("rf"), "answers-a-r1.json", "{}")
    _write(_jid("rf"), "answers-b-r2.json", "{}")
    files = storage.get_job_files(_jid("rf"))
    assert "answers-a-r1.json" in files
    assert "answers-b-r2.json" in files


def test_get_job_files_missing_job_returns_none():
    assert storage.get_job_files(_jid("no-such-job")) is None


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
    assert storage.delete_job(_jid("missing")) is False


def test_delete_job_removes_directory():
    _write(_jid("del-me"), "config.json", "{}")
    _write(_jid("del-me"), "report.json", "{}")
    assert (storage.BASE_DIR / _jid("del-me")).exists()
    assert storage.delete_job(_jid("del-me")) is True
    assert not (storage.BASE_DIR / _jid("del-me")).exists()


def test_create_job_id_unique_format():
    ids = {storage.create_job_id() for _ in range(50)}
    assert len(ids) == 50
    for jid in ids:
        assert JOB_ID_RE.match(jid)


def test_save_round_verdicts_persisted():
    storage.save_round_verdicts(_jid("rv"), [{"round": 1, "x": 1}, {"round": 2, "x": 2}])
    p = storage.BASE_DIR / _jid("rv") / "round-verdicts.json"
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8"))[1]["round"] == 2


# ---------------- job_id 校验与路径穿越防护（issue #17 / R3-001） ----------------

def test_is_valid_job_id_rejects_traversal_and_junk():
    for bad in ("..", ".", "../x", "a/b", "..\\..\\etc", "C:\\x",
                "abc", "2026-01-01_120000_abcdef",
                "20260101_120000_xyz123", "20260101_120000_ab",
                "20260101_120000_abcdefg", "20260101_120000_abcdef ",
                "", None):
        assert storage.is_valid_job_id(bad) is False, repr(bad)
    assert storage.is_valid_job_id(_jid("ok")) is True


def test_job_path_rejects_out_of_base():
    for bad in ("..", ".", "../x", "a/b", "..\\..\\etc", "C:\\x"):
        with pytest.raises(ValueError):
            storage._job_path(bad)


def test_delete_job_dotdot_never_escapes_base():
    """``..``/``.``/越界写法删除必须被拒绝：父目录哨兵与兄弟 job 不受影响。"""
    parent = storage.BASE_DIR.parent
    audit_log = parent / "audit.log"
    sentinel = parent / "sentinel.txt"
    audit_log.write_text("audit-line\n", encoding="utf-8")
    sentinel.write_text("keep", encoding="utf-8")
    storage.BASE_DIR.mkdir(parents=True, exist_ok=True)
    victim = _jid("victim")
    _write(victim, "config.json", "{}")

    for bad in ("..", ".", "../", "..\\..", "a/b", "C:\\x"):
        assert storage.delete_job(bad) is False, repr(bad)

    assert audit_log.read_text(encoding="utf-8") == "audit-line\n"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (storage.BASE_DIR / victim).exists()


def test_read_apis_traversal_return_none():
    assert storage.get_job_files("..") is None
    assert storage.get_job_status("..") is None
    assert storage.load_reveal("..") is None
    assert storage.load_review("..") is None


def test_read_apis_never_create_directory():
    """读操作不得隐式创建目录（_job_path 只取路径不 mkdir）。"""
    jid = _jid("read-only")
    assert storage.get_job_files(jid) is None
    assert not (storage.BASE_DIR / jid).exists()

# ---------------- 迭代一：数据集元信息 / 跨 job 汇总 ----------------

def test_dataset_version_source_roundtrip():
    data = {"name": "V", "tasks": [{"prompt": "p"}], "version": "v2", "source": "csv"}
    storage.save_dataset("V", data)
    loaded = storage.load_dataset("V")
    assert loaded["version"] == "v2"
    assert loaded["source"] == "csv"
    assert loaded["created_at"]
    raw = json.loads((storage.DATASETS_DIR / "V.json").read_text(encoding="utf-8"))
    assert raw["version"] == "v2" and raw["source"] == "csv"


def test_dataset_legacy_file_gets_default_version():
    (storage.DATASETS_DIR / "老.json").write_text(
        json.dumps({"name": "老", "tasks": [{"prompt": "p"}]}, ensure_ascii=False),
        encoding="utf-8")
    loaded = storage.load_dataset("老")
    assert loaded["version"] == "v1"
    assert loaded["source"] == "upload"


def test_dataset_list_type_counts():
    storage.save_dataset("TC", {"name": "TC", "tasks": [
        {"prompt": "p1", "type": "判别式"},
        {"prompt": "p2", "type": "生成式"},
        {"prompt": "p3"},
    ]})
    s = {x["name"]: x for x in storage.list_datasets()}["TC"]
    assert s["version"] == "v1"
    assert s["source"] == "upload"
    assert s["type_counts"] == {"判别式": 2, "生成式": 1}


def test_saturation_update_idempotent_by_job():
    assert storage.update_saturation(_jid("sat"), [{"id": "T1", "winner": "answer_x"}]) is True
    assert storage.update_saturation(_jid("sat"), [{"id": "T1", "winner": "answer_x"}]) is False
    data = storage.get_saturation()
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["job_id"] == _jid("sat")
    assert data["jobs"][0]["entries"][0]["winner"] == "answer_x"
    assert "updated_at" in data["jobs"][0]


def test_saturation_invalid_job_id_rejected():
    assert storage.update_saturation("../bad", []) is False


def test_saturation_corrupt_recovers_empty():
    storage.SATURATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    storage.SATURATION_FILE.write_text("{broken", encoding="utf-8")
    assert storage.get_saturation() == {"jobs": []}


# ---------------- 迭代七：benchmark 批次 ----------------

def _bid(tag: str) -> str:
    return "batch_" + _jid(tag)


def test_batch_crud():
    storage.save_batch(_bid("b1"), {"batch_id": _bid("b1"), "state": "running",
                                    "jobs": ["j1"], "models": ["m1"]})
    data = storage.load_batch(_bid("b1"))
    assert data["state"] == "running"
    assert storage.load_batch(_bid("nope")) is None
    summary = {x["batch_id"]: x for x in storage.list_batches()}
    assert summary[_bid("b1")]["n_jobs"] == 1


def test_batch_id_validation():
    with pytest.raises(ValueError):
        storage.save_batch("../bad", {})
    assert storage.is_valid_batch_id(_bid("ok")) is True
    assert storage.is_valid_batch_id("batch_bad") is False


def test_batch_corrupt_recovers_none():
    storage._ensure_batches_dir()
    (storage.BATCHES_DIR / f"{_bid('c')}.json").write_text("{broken", encoding="utf-8")
    assert storage.load_batch(_bid("c")) is None


# ---------------- 迭代七：断点续跑增量答案 ----------------

def test_answers_inc_idempotent_merge():
    jid = _jid("inc1")
    storage.save_answers_inc(jid, "T1", {"id": "T1", "raw_answer": "a1"})
    storage.save_answers_inc(jid, "T2", {"id": "T2", "raw_answer": "a2"})
    storage.save_answers_inc(jid, "T1", {"id": "T1", "raw_answer": "a1v2"})  # 幂等覆盖
    data = storage.load_answers_inc(jid)
    assert set(data) == {"T1", "T2"}
    assert data["T1"]["raw_answer"] == "a1v2"
    assert storage.partial_answers_count(jid) == 2


def test_answers_inc_missing_and_corrupt():
    assert storage.load_answers_inc(_jid("inc-none")) == {}
    jid = _jid("inc-bad")
    storage.save_answers_inc(jid, "T1", {})
    (storage.BASE_DIR / jid / "answers-inc.json").write_text("{broken", encoding="utf-8")
    assert storage.load_answers_inc(jid) == {}
    assert storage.partial_answers_count(jid) == 0
