# -*- coding: utf-8 -*-
"""历史记录本地文件库：任务集/答卷/报告持久化到 .eval/history/<jobId>/，
不存储 API Key，仅在内存中保留 Key。"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.security import redact_sensitive

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".eval" / "history"

# 系统生成 job_id 的唯一合法格式（create_job_id 输出），API 与存储统一入口校验
JOB_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")


def is_valid_job_id(job_id: str) -> bool:
    """job_id 必须符合 create_job_id() 生成的格式（issue #17，防路径穿越）。"""
    return isinstance(job_id, str) and bool(JOB_ID_RE.fullmatch(job_id))


def validate_job_id(job_id: str) -> str:
    if not is_valid_job_id(job_id):
        raise ValueError(f"非法 job_id: {job_id!r}")
    return job_id


def _job_path(job_id: str) -> Path:
    """取 job 目录路径（只解析不创建，读操作专用）。

    双层防护：格式校验 + resolve() 后必须是 BASE_DIR 的直接子目录，
    拒绝 ``..``、``.``、绝对路径等任何越界写法。
    """
    validate_job_id(job_id)
    d = (BASE_DIR / job_id).resolve()
    if d.parent != BASE_DIR.resolve():
        raise ValueError(f"job_id 越界: {job_id!r}")
    return d


def _job_dir(job_id: str) -> Path:
    """取 job 目录路径并创建（写操作专用），职责与 _job_path 分离。"""
    d = _job_path(job_id)
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
        "code_verify_mode": config.get("code_verify_mode", "off"),
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


def save_reveal(job_id: str, reveal: dict) -> Path:
    """持久化双盲身份映射（答案X/答案Y 对应哪份答卷），重启不丢。"""
    p = _job_dir(job_id) / "reveal.json"
    p.write_text(json.dumps(reveal, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_reveal(job_id: str) -> dict | None:
    try:
        p = _job_path(job_id) / "reveal.json"
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_review(job_id: str, review: dict) -> Path:
    """保存用户已提交的人工评审（草稿/结果）。"""
    p = _job_dir(job_id) / "review.json"
    p.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def save_round_verdicts(job_id: str, round_verdicts: list[dict]) -> Path:
    """持久化每轮 verdict（含原始 X/Y 分数与该轮 reveal 映射），保证结果可审计可重算。"""
    p = _job_dir(job_id) / "round-verdicts.json"
    p.write_text(json.dumps(round_verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_review(job_id: str) -> dict | None:
    try:
        p = _job_path(job_id) / "review.json"
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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
        if "verdict.json" not in files:
            return "reviewing"
        return "judging"
    if "tasks.json" in files:
        return "executing"
    return "pending"


def get_job_status(job_id: str) -> dict[str, Any] | None:
    try:
        cfg_path = _job_path(job_id) / "config.json"
    except ValueError:
        return None
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    d = _job_path(job_id)
    files = [f.name for f in d.iterdir()]
    verdict = None
    if "verdict.json" in files:
        verdict = json.loads((d / "verdict.json").read_text(encoding="utf-8"))
    return redact_sensitive({
        "job_id": job_id,
        "config": cfg,
        "state": _job_state(d),
        "verdict": verdict,
        "files": files,
    })


def list_jobs() -> list[dict[str, Any]]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in sorted(BASE_DIR.iterdir(), reverse=True):
        if d.is_dir():
            cfg_path = d / "config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                jobs.append(redact_sensitive({
                    "job_id": d.name,
                    "state": _job_state(d),
                    "model_a": cfg.get("model_a_name", "?"),
                    "model_b": cfg.get("model_b_name", "?"),
                    "created_at": cfg.get("created_at", ""),
                }))
    return jobs


def delete_job(job_id: str) -> bool:
    """删除指定任务的所有文件。

    仅允许删除 BASE_DIR 的直接子目录：非法格式或越界路径一律拒绝
    （issue #17，杜绝 job_id 路径穿越删除 .eval 等上级目录）。
    """
    try:
        d = _job_path(job_id)
    except ValueError:
        return False
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def get_job_files(job_id: str) -> dict[str, Any] | None:
    try:
        d = _job_path(job_id)
    except ValueError:
        return None
    if not d.exists():
        return None
    result = {}
    names = ["tasks.json", "answers-a.json", "answers-b.json", "verdict.json",
             "report.json", "config.json", "round-verdicts.json"]
    names += sorted(p.name for p in d.glob("answers-*-r*.json"))
    for name in names:
        p = d / name
        if p.exists():
            try:
                result[name] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result[name] = None
    return redact_sensitive(result)


# ---- 数据集管理 ----

DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"

# 跨 job 历史汇总（迭代一：接口与幂等写入，真实接入在后续迭代）
STATS_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".eval" / "stats"
SATURATION_FILE = STATS_DIR / "saturation.json"


def update_saturation(job_id: str, entries: list[dict]) -> bool:
    """向跨 job 汇总表追加一轮评测的逐题结果（按 job_id 幂等：重复调用跳过）。

    entries 为 [{id, dimension, type, answer_x, answer_y, winner}]。
    返回 True=首次写入成功；False=已存在（幂等跳过）或写入失败。
    """
    if not is_valid_job_id(job_id):
        return False
    try:
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        data = get_saturation()
        for job in data.get("jobs", []):
            if job.get("job_id") == job_id:
                return False
        data.setdefault("jobs", []).append({
            "job_id": job_id,
            "entries": entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        SATURATION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def get_saturation() -> dict:
    """读取跨 job 汇总表；文件缺失或损坏时静默返回空结构（不抛异常）。"""
    if not SATURATION_FILE.exists():
        return {"jobs": []}
    try:
        data = json.loads(SATURATION_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
            return {"jobs": []}
        return data
    except (json.JSONDecodeError, OSError):
        return {"jobs": []}


# Windows 文件名非法字符（含路径分隔、控制字符），全部替换为下划线
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_dataset_name(name: str) -> str:
    """将数据集名消毒为安全文件名（保留展示用 name 原文，仅文件层面替换）。"""
    safe = _INVALID_FS_CHARS.sub("_", str(name)).strip().rstrip(".").strip()
    return safe or "dataset"


def _ensure_datasets_dir():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(name: str, data: dict) -> Path:
    """保存评测集到 data/datasets/{safe_name}.json（永久保存）。

    迭代一：附加版本与来源元信息（version/source/created_at），
    老文件读取时缺省补 version="v1"（零破坏）。
    """
    _ensure_datasets_dir()
    p = DATASETS_DIR / f"{_safe_dataset_name(name)}.json"
    meta = {
        "version": data.get("version", "v1"),
        "source": data.get("source", "upload"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data = {**meta, **data}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_dataset(name: str) -> dict | None:
    """读取评测集（老文件缺 version/source 时补缺省，保持零破坏）。"""
    p = DATASETS_DIR / f"{_safe_dataset_name(name)}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("version", "v1")
    data.setdefault("source", "upload")
    return data


def _type_counts(tasks: list[dict]) -> dict[str, int]:
    """统计评测集任务类型分布（判别式/生成式）。"""
    counts: dict[str, int] = {}
    for t in tasks:
        ttype = t.get("type", "判别式")
        counts[ttype] = counts.get(ttype, 0) + 1
    return counts


def list_datasets() -> list[dict[str, Any]]:
    """列出所有已上传的评测集摘要（含版本与类型分布）。"""
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
                "version": data.get("version", "v1"),
                "source": data.get("source", "upload"),
                "type_counts": _type_counts(tasks),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    return result


def delete_dataset(name: str) -> bool:
    """删除评测集。"""
    p = DATASETS_DIR / f"{_safe_dataset_name(name)}.json"
    if not p.exists():
        return False
    p.unlink()
    return True
