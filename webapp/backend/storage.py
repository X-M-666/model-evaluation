# -*- coding: utf-8 -*-
"""历史记录本地文件库：任务集/答卷/报告持久化到 .eval/history/<jobId>/，
不存储 API Key，仅在内存中保留 Key。"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".eval" / "history"


def _job_dir(job_id: str) -> Path:
    d = BASE_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def save_config(job_id: str, config: dict) -> None:
    """保存任务配置（API Key 打码，不落盘）。保留嵌套结构与 dataset_name/repeat_n。"""
    def _mask_model(m):
        return {
            "name": m.get("name", "?"),
            "url": m.get("url", ""),
            "key_masked": "***",
            "temperature": m.get("temperature", 0.7),
            "max_tokens": m.get("max_tokens", 4096),
            "top_p": m.get("top_p"),
        }
    safe = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_a": _mask_model(config.get("model_a", {})),
        "model_b": _mask_model(config.get("model_b", {})),
        "model_a_name": config.get("model_a", {}).get("name", "?"),
        "model_b_name": config.get("model_b", {}).get("name", "?"),
        "dims": config.get("dims"),
        "seed": config.get("seed"),
        "dataset_name": config.get("dataset_name"),
        "repeat_n": config.get("repeat_n", 1),
        "model_a_key_masked": "***",
        "model_b_key_masked": "***",
    }
    (_job_dir(job_id) / "config.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


def save_task_set(job_id: str, task_set: dict) -> Path:
    p = _job_dir(job_id) / "tasks.json"
    p.write_text(json.dumps(task_set, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def save_answers(job_id: str, label: str, answers: dict) -> Path:
    p = _job_dir(job_id) / f"answers-{label}.json"
    p.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def save_verdict(job_id: str, verdict: dict) -> Path:
    p = _job_dir(job_id) / "verdict.json"
    p.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def save_error(job_id: str, message: str) -> Path:
    p = _job_dir(job_id) / "error.json"
    p.write_text(json.dumps({"error": message}, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def save_report(job_id: str, report: dict) -> Path:
    p = _job_dir(job_id) / "report.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _job_state(d: Path) -> str:
    files = [f.name for f in d.iterdir()]
    if "error.json" in files:
        return "error"
    if "report.json" in files:
        return "completed"
    if "answers-a.json" in files and "answers-b.json" in files:
        return "judging"
    if "tasks.json" in files:
        return "executing"
    return "pending"


def get_job_status(job_id: str) -> dict[str, Any] | None:
    cfg_path = _job_dir(job_id) / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    d = _job_dir(job_id)
    files = [f.name for f in d.iterdir()]
    verdict = None
    if "verdict.json" in files:
        verdict = json.loads((d / "verdict.json").read_text(encoding="utf-8"))
    return {
        "job_id": job_id,
        "config": cfg,
        "state": _job_state(d),
        "verdict": verdict,
        "files": files,
    }


def list_jobs() -> list[dict[str, Any]]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in sorted(BASE_DIR.iterdir(), reverse=True):
        if d.is_dir():
            cfg_path = d / "config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                jobs.append({
                    "job_id": d.name,
                    "state": _job_state(d),
                    "model_a": cfg.get("model_a_name", "?"),
                    "model_b": cfg.get("model_b_name", "?"),
                    "created_at": cfg.get("created_at", ""),
                })
    return jobs


def delete_job(job_id: str) -> bool:
    """删除指定任务的所有文件。"""
    d = BASE_DIR / job_id
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def get_job_files(job_id: str) -> dict[str, Any] | None:
    d = BASE_DIR / job_id
    if not d.exists():
        return None
    result = {}
    for name in ["tasks.json", "answers-a.json", "answers-b.json", "verdict.json", "report.json", "config.json"]:
        p = d / name
        if p.exists():
            try:
                result[name] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result[name] = None
    return result


# ---- 数据集管理 ----

DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"


def _ensure_datasets_dir():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(name: str, data: dict) -> Path:
    """保存评测集到 data/datasets/{safe_name}.json（永久保存）。"""
    _ensure_datasets_dir()
    safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    p = DATASETS_DIR / f"{safe_name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_dataset(name: str) -> dict | None:
    """读取评测集。"""
    safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    p = DATASETS_DIR / f"{safe_name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_datasets() -> list[dict[str, Any]]:
    """列出所有已上传的评测集摘要。"""
    _ensure_datasets_dir()
    result = []
    for p in sorted(DATASETS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            tasks = data.get("tasks", [])
            dims = list({t.get("dimension", "自定义") for t in tasks})
            result.append({
                "name": data.get("name", p.stem),
                "description": data.get("description", ""),
                "task_count": len(tasks),
                "dimensions": dims,
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    return result


def delete_dataset(name: str) -> bool:
    """删除评测集。"""
    safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    p = DATASETS_DIR / f"{safe_name}.json"
    if not p.exists():
        return False
    p.unlink()
    return True
