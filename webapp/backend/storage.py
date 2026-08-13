# -*- coding: utf-8 -*-
"""历史记录本地文件库：任务集/答卷/报告持久化到 .eval/history/<jobId>/，
不存储 API Key，仅在内存中保留 Key。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import shutil
import sys
import uuid
import zipfile
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


def collect_env_snapshot() -> dict[str, Any]:
    """采集运行环境快照（迭代四）：OS/Python/CPU/关键运行时信息，无任何密钥。"""
    import importlib.metadata as _md

    def _pkg(name: str) -> str | None:
        try:
            return _md.version(name)
        except Exception:
            return None

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor() or "",
            "python_impl": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "cwd": os.getcwd(),
        "packages": {
            "fastapi": _pkg("fastapi"),
            "uvicorn": _pkg("uvicorn"),
            "httpx": _pkg("httpx"),
            "pydantic": _pkg("pydantic"),
            "numpy": _pkg("numpy"),
            "pandas": _pkg("pandas"),
        },
    }


def save_env_snapshot(job_id: str) -> Path:
    """写环境快照 <job>/env.json（迭代四：评测可复现性元数据，无密钥）。"""
    p = _job_dir(job_id) / "env.json"
    p.write_text(json.dumps(collect_env_snapshot(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def load_env_snapshot(job_id: str) -> dict[str, Any] | None:
    """读取环境快照；不存在/损坏时返回 None（零破坏）。"""
    try:
        p = _job_path(job_id) / "env.json"
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_config(job_id: str, config: dict) -> None:
    """保存任务配置（API Key 打码，不落盘）。保留嵌套结构与 dataset_name/repeat_n。

    迭代二：judge/embedding 辅助配置同样掩码（Key 不落盘不变量），
    并落盘 prompt_strategy 供历史报告自描述执行方式。
    """
    def _mask_model(m):
        if not isinstance(m, dict):
            return m
        return {
            "name": m.get("name", "?"),
            "url": m.get("url", ""),
            "key_masked": "***",
            "temperature": m.get("temperature", 0.7),
            "max_tokens": m.get("max_tokens", 4096),
            "top_p": m.get("top_p"),
        }
    judge_cfg = None
    if isinstance(config.get("review"), dict):
        judge_cfg = config["review"].get("judge")
    elif config.get("judge"):
        judge_cfg = config.get("judge")
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
        "prompt_strategy": config.get("prompt_strategy", "cot"),
        "model_a_key_masked": "***",
        "model_b_key_masked": "***",
        "judge": _mask_model(judge_cfg) if judge_cfg else None,
        "judge_key_masked": "***",
        "embedding": _mask_model(config.get("embedding")) if config.get("embedding") else None,
        "embedding_key_masked": "***",
        "review_mode": config.get("review", {}).get("mode", "pure_human") if isinstance(config.get("review"), dict) else "pure_human",
        "fail_open": bool(config["review"].get("fail_open")) if isinstance(config.get("review"), dict) else False,
        "review_k_top_human": int(config["review"].get("k_top_human") or 0) if isinstance(config.get("review"), dict) else 0,
        "review_degraded": bool(config["review"].get("degraded")) if isinstance(config.get("review"), dict) else False,
        "budget": config.get("budget") if isinstance(config.get("budget"), dict) else None,
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


def load_round_verdicts(job_id: str) -> list[dict] | None:
    """读取逐轮 verdict（迭代三：hybrid 重启恢复降级/M2 复用）。"""
    try:
        p = _job_path(job_id) / "round-verdicts.json"
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


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


# 导出白名单（迭代四）：仅这些文件进入评测包，杜绝目录/未知文件外泄
EXPORT_NAMES = [
    "tasks.json", "answers-a.json", "answers-b.json", "verdict.json",
    "report.json", "config.json", "round-verdicts.json",
    "hybrid-review.json", "env.json",
]


def build_export_zip(job_id: str) -> bytes:
    """打包评测包（迭代四）：白名单文件 + MANIFEST.json（逐文件 sha256）。

    校验 job_id 格式与目录越界（与 _job_path 同防护）；文件不存在则跳过。
    返回 zip 字节（内存组装，不落盘）。
    """
    d = _job_path(job_id)
    names = list(EXPORT_NAMES)
    names += sorted(p.name for p in d.glob("answers-*-r*.json"))
    buf = io.BytesIO()
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "total": 0,
    }
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            p = d / name
            if not p.exists():
                continue
            data = p.read_bytes()
            zf.writestr(name, data)
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
            manifest["total"] += 1
        zf.writestr("MANIFEST.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2))
    return buf.getvalue()


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
             "report.json", "config.json", "round-verdicts.json",
             "hybrid-review.json", "env.json"]
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

# 金标集目录（迭代三）：<name>.json，结构 {name, items, source, created_at}
GOLD_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"

# 出题待审核批次池（迭代四）：<gen_id>.json，结构见 save_generation_batch
GENERATED_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

# 跨 job 历史汇总（迭代一：接口与幂等写入，迭代五：main.py 接线 + dataset 分组）
STATS_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".eval" / "stats"
SATURATION_FILE = STATS_DIR / "saturation.json"

# Bad Case 库（迭代五）：<case_id>.json，结构见 save_badcase
BADCASES_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".eval" / "badcases"


def update_saturation(job_id: str, entries: list[dict], dataset: str | None = None) -> bool:
    """向跨 job 汇总表追加一轮评测的逐题结果（按 job_id 幂等：重复调用跳过）。

    entries 为 [{id, dimension, type, answer_x, answer_y, winner}]。
    dataset：评测集名（迭代五，供饱和度趋势按数据集分组；缺省不记录）。
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
        record: dict[str, Any] = {
            "job_id": job_id,
            "entries": entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if dataset:
            record["dataset"] = dataset
        data.setdefault("jobs", []).append(record)
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


# ---- Bad Case 库（迭代五） ----

# case_id = bc_<job_id>_<task_id 消毒>，job_id 段与 create_job_id 同格式
BAD_CASE_ID_RE = re.compile(r"^bc_\d{8}_\d{6}_[0-9a-f]{6}_[0-9A-Za-z_]+$")


def is_valid_badcase_id(case_id: str) -> bool:
    """case_id 必须符合 save_badcase 生成格式（防路径穿越）。"""
    return isinstance(case_id, str) and bool(BAD_CASE_ID_RE.fullmatch(case_id))


def make_badcase_id(job_id: str, task_id: str) -> str:
    """由 job_id + task_id 生成 case_id（task_id 消毒为安全文件名段）。"""
    return f"bc_{job_id}_{_safe_dataset_name(str(task_id))}"


def save_badcase(case: dict) -> Path:
    """写入一条 bad case 记录（按 case_id 覆盖更新）。"""
    case_id = case.get("case_id", "")
    if not is_valid_badcase_id(case_id):
        raise ValueError(f"非法 case_id: {case_id!r}")
    BADCASES_DIR.mkdir(parents=True, exist_ok=True)
    p = BADCASES_DIR / f"{case_id}.json"
    p.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_badcase(case_id: str) -> dict | None:
    """读取单条 bad case；非法 id / 不存在 / 损坏返回 None。"""
    if not is_valid_badcase_id(case_id):
        return None
    p = BADCASES_DIR / f"{case_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_badcases(job_id: str | None = None) -> list[dict]:
    """列出 bad case 摘要（新→旧）；job_id 非空时仅返回该 job 的记录。"""
    BADCASES_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(BADCASES_DIR.glob("bc_*.json"), reverse=True):
        try:
            case = json.loads(p.read_text(encoding="utf-8"))
            if job_id and case.get("job_id") != job_id:
                continue
            attribution = case.get("attribution") or {}
            result.append({
                "case_id": case.get("case_id", p.stem),
                "job_id": case.get("job_id", ""),
                "task_id": case.get("task_id", ""),
                # 归因后以 attribution.label 为权威分类（LLM/人工确认均可更新）
                "category": attribution.get("label") or case.get("category", "未归类"),
                "sources": case.get("sources", []),
                "model": case.get("model", "both"),
                "score": case.get("score", {}),
                "confirmed": bool(attribution.get("confirmed")),
                "attribution_by": attribution.get("by", "auto"),
                "suggestion": attribution.get("suggestion", ""),
                "created_at": case.get("created_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return result


def delete_badcase(case_id: str) -> bool:
    """删除一条 bad case。"""
    if not is_valid_badcase_id(case_id):
        return False
    p = BADCASES_DIR / f"{case_id}.json"
    if not p.exists():
        return False
    p.unlink()
    return True


def update_badcase_attribution(case_id: str, attribution: dict) -> dict | None:
    """更新归因字段（人工确认/改标/驳回），返回更新后的记录；不存在返回 None。"""
    case = load_badcase(case_id)
    if case is None:
        return None
    merged = dict(case.get("attribution") or {})
    merged.update({k: v for k, v in attribution.items() if v is not None})
    case["attribution"] = merged
    save_badcase(case)
    return case


def export_badcases_json(job_id: str | None = None) -> str:
    """导出 bad case 清单（含修订建议），序列化为 JSON 字符串。"""
    cases = []
    for p in sorted(BADCASES_DIR.glob("bc_*.json")):
        try:
            case = json.loads(p.read_text(encoding="utf-8"))
            if job_id and case.get("job_id") != job_id:
                continue
            cases.append(case)
        except (json.JSONDecodeError, OSError):
            continue
    return json.dumps({"job_id": job_id, "total": len(cases), "cases": cases},
                      ensure_ascii=False, indent=2)


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


# ---- 金标集（迭代三） ----

def _ensure_gold_dir():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)


def save_gold(name: str, data: dict) -> Path:
    """保存金标集到 data/gold/{safe_name}.json；同名覆盖（manual 覆盖 demo）。"""
    _ensure_gold_dir()
    p = GOLD_DIR / f"{_safe_dataset_name(name)}.json"
    record = {
        "name": str(name),
        "items": data.get("items", []),
        "source": data.get("source", "manual"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_gold(name: str) -> dict | None:
    """读取金标集；文件缺失/损坏返回 None（损坏不自动覆盖，保留现场）。"""
    p = GOLD_DIR / f"{_safe_dataset_name(name)}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("source", "manual")
    return data


def list_gold() -> list[dict[str, Any]]:
    """列出全部金标集摘要（含 source，供前端标注 demo/manual）。"""
    _ensure_gold_dir()
    result = []
    for p in sorted(GOLD_DIR.glob("*.json")):
        data = load_gold(p.stem)
        if data is None:
            continue
        result.append({
            "name": data.get("name", p.stem),
            "source": data.get("source", "manual"),
            "item_count": len(data.get("items", [])),
            "created_at": data.get("created_at", ""),
        })
    return result


def delete_gold(name: str) -> bool:
    """删除金标集。"""
    p = GOLD_DIR / f"{_safe_dataset_name(name)}.json"
    if not p.exists():
        return False
    p.unlink()
    return True


# ---- hybrid 复核（迭代三） ----

def save_hybrid_review(job_id: str, data: dict) -> Path:
    """持久化 hybrid 复核集/已提交复核（重启恢复 M2）。"""
    p = _job_dir(job_id) / "hybrid-review.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_hybrid_review(job_id: str) -> dict | None:
    try:
        p = _job_path(job_id) / "hybrid-review.json"
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---- 出题待审核批次池（迭代四） ----

# 系统生成 gen_id 的唯一合法格式（main 生成：gen_ + create_job_id）
GEN_ID_RE = re.compile(r"^gen_\d{8}_\d{6}_[0-9a-f]{6}$")


def is_valid_gen_id(gen_id: str) -> bool:
    """gen_id 必须符合系统生成格式（防路径穿越）。"""
    return isinstance(gen_id, str) and bool(GEN_ID_RE.fullmatch(gen_id))


def _ensure_generated_dir():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def save_generation_batch(gen_id: str, data: dict) -> Path:
    """保存出题批次（含 spec/items/状态）。spec 不含 gen_config URL/Key 明文。"""
    if not is_valid_gen_id(gen_id):
        raise ValueError(f"非法 gen_id: {gen_id!r}")
    _ensure_generated_dir()
    p = GENERATED_DIR / f"{gen_id}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_generation_batch(gen_id: str) -> dict | None:
    """读取出题批次；缺失/损坏返回 None。"""
    if not is_valid_gen_id(gen_id):
        return None
    p = GENERATED_DIR / f"{gen_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def list_generation_batches() -> list[dict]:
    """列出全部出题批次摘要（含题目状态分布）。"""
    _ensure_generated_dir()
    result = []
    for p in sorted(GENERATED_DIR.glob("gen_*.json")):
        data = load_generation_batch(p.stem)
        if data is None:
            continue
        items = data.get("items", []) if isinstance(data.get("items"), list) else []
        stats = {"total": len(items), "pending": 0, "approved": 0, "rejected": 0}
        for it in items:
            st = it.get("status")
            if st in stats:
                stats[st] += 1
        result.append({
            "gen_id": p.stem,
            "state": data.get("state", "unknown"),
            "created_at": data.get("created_at", ""),
            "task_type": (data.get("spec") or {}).get("task_type", ""),
            "dimension": (data.get("spec") or {}).get("dimension", ""),
            "target_dataset": (data.get("spec") or {}).get("target_dataset"),
            "gen_name": (data.get("spec") or {}).get("gen_name", ""),
            "items": stats,
        })
    return result


def bump_dataset_version(version: str | None) -> str:
    """评测集版本递增：v1 → v2 ……；非 v{n} 形从 v1 起算。"""
    m = re.match(r"^v(\d+)$", str(version or "").strip())
    n = int(m.group(1)) if m else 0
    return f"v{n + 1}"


# ---- 扰动评测（迭代六）----

# 系统生成 perturb_id 的唯一合法格式（main 生成：prb_ + create_job_id）
PERTURB_ID_RE = re.compile(r"^prb_\d{8}_\d{6}_[0-9a-f]{6}$")

# 扰动评测产物（运行数据，同 badcases 约定）
PERTURB_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".eval" / "perturb"


def is_valid_perturb_id(perturb_id: str) -> bool:
    return isinstance(perturb_id, str) and bool(PERTURB_ID_RE.fullmatch(perturb_id))


def _ensure_perturb_dir():
    PERTURB_DIR.mkdir(parents=True, exist_ok=True)


def save_perturb(perturb_id: str, data: dict) -> Path:
    """保存扰动评测结果。data 含 model 掩码字段，不含 Key 明文。"""
    if not is_valid_perturb_id(perturb_id):
        raise ValueError(f"非法 perturb_id: {perturb_id!r}")
    _ensure_perturb_dir()
    p = PERTURB_DIR / f"{perturb_id}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_perturb(perturb_id: str) -> dict | None:
    if not is_valid_perturb_id(perturb_id):
        return None
    p = PERTURB_DIR / f"{perturb_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def list_perturbs() -> list[dict]:
    """列出全部扰动评测摘要。"""
    _ensure_perturb_dir()
    result = []
    for p in sorted(PERTURB_DIR.glob("prb_*.json")):
        data = load_perturb(p.stem)
        if data is None:
            continue
        result.append({
            "perturb_id": p.stem,
            "state": data.get("state", "unknown"),
            "created_at": data.get("created_at", ""),
            "model": data.get("model_name", ""),
            "dataset": data.get("dataset", ""),
            "modes": data.get("modes", []),
            "progress": data.get("progress", ""),
        })
    return result


# ---- 排行榜（迭代六）----

# 系统生成 lb_id 的唯一合法格式（main 生成：lb_ + create_job_id）
LB_ID_RE = re.compile(r"^lb_\d{8}_\d{6}_[0-9a-f]{6}$")

# 排行榜（用户数据，同 datasets 约定）
LEADERBOARD_DIR = Path(__file__).resolve().parent.parent / "data" / "leaderboards"


def is_valid_lb_id(lb_id: str) -> bool:
    return isinstance(lb_id, str) and bool(LB_ID_RE.fullmatch(lb_id))


def _ensure_leaderboard_dir():
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)


def save_leaderboard(lb_id: str, data: dict) -> Path:
    if not is_valid_lb_id(lb_id):
        raise ValueError(f"非法 lb_id: {lb_id!r}")
    _ensure_leaderboard_dir()
    p = LEADERBOARD_DIR / f"{lb_id}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_leaderboard(lb_id: str) -> dict | None:
    if not is_valid_lb_id(lb_id):
        return None
    p = LEADERBOARD_DIR / f"{lb_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def list_leaderboards() -> list[dict]:
    _ensure_leaderboard_dir()
    result = []
    for p in sorted(LEADERBOARD_DIR.glob("lb_*.json")):
        data = load_leaderboard(p.stem)
        if data is None:
            continue
        result.append({
            "lb_id": p.stem,
            "name": data.get("name", ""),
            "created_at": data.get("created_at", ""),
            "models": data.get("models", []),
            "dataset": data.get("dataset", ""),
            "n_jobs": len(data.get("jobs", [])),
        })
    return result
