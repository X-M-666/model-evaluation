# -*- coding: utf-8 -*-
"""指标注册表与计算（迭代二）：按任务类型路由到纯函数指标。

- 判别式：top1 / exact_match / f1（字符级）/ relaxed_accuracy（数值容差）
- 生成式：semantic_sim / rubric_score / consistency / bleu / rouge_l
- 代码维度：文本指标一律 N/A（以 code_verify 客观执行为权威）

全部指标为纯函数：给定 task 与答卷条目（含语义向量）即确定性输出，
报告层无网络调用。BLEU/ROUGE 纯 Python 实现（零新依赖）。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

from backend.engine.extract import extract_per_case
from backend.engine.embed import cosine, ngram_vec
from backend.engine.tasks import CODE_DIMENSION

METRICS: dict[str, Callable[..., Any]] = {}

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"[A-Da-d]")


def _metric(name: str):
    """注册表装饰器：向 METRICS 注册指标实现（迭代 0 扩展点落地）。"""

    def deco(fn: Callable) -> Callable:
        METRICS[name] = fn
        return fn

    return deco


def _norm(s: str) -> str:
    return re.sub(r"[\s，。、；：:()（）\[\]【】\"''`~\-—]+", "", (s or "")).lower()


def _try_num(s: str) -> float | None:
    m = _FLOAT_RE.search(str(s or ""))
    return float(m.group()) if m else None


def _char_f1(pred: str, ref: str) -> float:
    """字符级 F1（中文友好，去空白，字符多集重叠）。"""
    pc = [c for c in (pred or "") if not c.isspace()]
    rc = [c for c in (ref or "") if not c.isspace()]
    if not pc or not rc:
        return 0.0
    inter = sum((Counter(pc) & Counter(rc)).values())
    if inter == 0:
        return 0.0
    p, r = inter / len(pc), inter / len(rc)
    return round(2 * p * r / (p + r), 4)


def _tokenize(s: str) -> list[str]:
    """按词切分（中文连续字符视为词，兼容标点）。"""
    return re.findall(r"[\w]+", s or "")


@_metric("bleu")
def bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    """简化 BLEU（1-4 gram + 长度惩罚），纯 Python 零依赖。"""
    p, r = _tokenize(pred), _tokenize(ref)
    if not p:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(p) < n:
            break
        pred_ngrams = Counter(zip(*[p[i:] for i in range(n)]))
        ref_ngrams = Counter(zip(*[r[i:] for i in range(n)]))
        total = sum(pred_ngrams.values())
        clipped = sum(min(c, ref_ngrams.get(g, 0)) for g, c in pred_ngrams.items())
        precisions.append(clipped / total if total else 0.0)
    if not precisions:
        return 0.0
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(len(p), 1))
    if any(pr == 0 for pr in precisions):
        return 0.0
    return round(bp * math.exp(sum(math.log(pr) for pr in precisions) / len(precisions)), 4)


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = dp[j + 1]
            if x == y:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j], cur)
            prev = cur
    return dp[-1]


@_metric("rouge_l")
def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L：最长公共子序列 F1（词级）。"""
    p, r = _tokenize(pred), _tokenize(ref)
    if not p or not r:
        return 0.0
    l = _lcs_len(p, r)
    precision, recall = l / len(p), l / len(r)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _case_expecteds(task: dict[str, Any]) -> list[str]:
    cases = task.get("test_cases") or []
    if cases:
        return [str(c.get("expected", "")) for c in cases]
    e = task.get("expected", "")
    return [e] if e else []


def _case_match(extracted: str | None, expected: str) -> bool:
    """子题比对：字母 → 字母精确；数值 → isclose 容差；否则归一化包含匹配。"""
    if not expected:
        return False
    ex = (extracted or "").strip()
    if not ex:
        return False
    if _LETTER_RE.fullmatch(expected) and _LETTER_RE.fullmatch(ex):
        return ex.upper() == expected.upper()
    if _LETTER_RE.fullmatch(ex):
        m = _LETTER_RE.search(expected)
        if m:
            return ex.upper() == m.group(0).upper()
    num_e, num_x = _try_num(expected), _try_num(ex)
    if num_e is not None and num_x is not None:
        return math.isclose(num_x, num_e, rel_tol=0.01, abs_tol=0.5)
    return _norm(ex) == _norm(expected) or expected.strip().lower() in ex.lower()


def _main_entry(entries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """取首个成功条目；失败返回 (None, 原因)。"""
    ok = [e for e in entries if (e.get("api_info") or {}).get("status") == "ok"]
    if not ok:
        api = (entries[0] or {}).get("api_info", {}) if entries else {}
        return None, api.get("error") or "调用失败，无可用回答"
    main = ok[0]
    if (main.get("api_info") or {}).get("truncated"):
        return main, "回答被截断（finish_reason=length）"
    return main, None


@_metric("top1")
def metric_top1(matched: list[bool]) -> float | None:
    """判别式 top1：正确子题占比（多子题平均）。"""
    return round(sum(matched) / len(matched), 4) if matched else None


@_metric("exact_match")
def metric_exact_match(matched: list[bool]) -> float:
    """判别式 exact_match：全对为 1.0，否则 0.0。"""
    return 1.0 if matched and all(matched) else 0.0


@_metric("f1")
def metric_f1(pairs: list[tuple[str | None, str]]) -> float | None:
    """判别式 f1：子题字符级 F1 平均。"""
    f1s = [_char_f1(x, e) for x, e in pairs if e and x]
    return round(sum(f1s) / len(f1s), 4) if f1s else None


@_metric("relaxed_accuracy")
def metric_relaxed_accuracy(pairs: list[tuple[str | None, str]]) -> float | None:
    """判别式 relaxed_accuracy：仅数值子题按 isclose 容差判对。"""
    numeric = [1.0 if _case_match(x, e) else 0.0 for x, e in pairs if _try_num(e) is not None]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


@_metric("semantic_sim")
def metric_semantic_sim(raw: str, expected: str, sem: dict[str, Any] | None) -> float | None:
    """生成式 semantic_sim：优先预计算向量（执行期采集），否则离线 n-gram 余弦。"""
    sem = sem or {}
    vec_ans = sem.get("vector") if isinstance(sem, dict) else None
    vec_ref = sem.get("ref_vector") if isinstance(sem, dict) else None
    if vec_ans is not None and vec_ref is not None:
        return round(cosine(vec_ans, vec_ref), 4)
    if not raw and not expected:
        return 0.0
    return round(cosine(ngram_vec(raw), ngram_vec(expected)), 4)


@_metric("rubric_score")
def metric_rubric_score(verdict_score: float | None) -> float | None:
    """生成式 rubric_score：评审评分透传（无评审时为 None）。"""
    return round(float(verdict_score), 2) if verdict_score is not None else None


@_metric("consistency")
def metric_consistency(runs: list[str]) -> float | None:
    """生成式 consistency：多次运行两两 n-gram 余弦均值（稳定性维度）。"""
    if len(runs) < 2:
        return None
    sims = [cosine(ngram_vec(a), ngram_vec(b)) for a, b in zip(runs, runs[1:])]
    return round(sum(sims) / len(sims), 4) if sims else None


def _discriminative(
    task: dict[str, Any],
    main: dict[str, Any],
) -> dict[str, Any]:
    raw = main.get("raw_answer", "")
    expected_list = [e for e in _case_expecteds(task) if e]
    if not expected_list:
        return {"skipped": True, "reason": "no_expected"}
    extracted = extract_per_case(task, raw)
    pairs = list(zip(extracted, expected_list))
    matched = [_case_match(x, e) for x, e in pairs if e]

    return {
        "top1": metric_top1(matched),
        "exact_match": metric_exact_match(matched),
        "f1": metric_f1(pairs),
        "relaxed_accuracy": metric_relaxed_accuracy(pairs),
    }


def _generative(
    task: dict[str, Any],
    entries: list[dict[str, Any]],
    main: dict[str, Any],
    verdict_score: float | None,
) -> dict[str, Any]:
    raw = main.get("raw_answer", "")
    expected = (task.get("expected") or "").strip() or None

    out: dict[str, Any] = {"rubric_score": metric_rubric_score(verdict_score)}

    if expected is None:
        out["semantic_sim"] = None
        out["bleu"] = None
        out["rouge_l"] = None
    else:
        out["semantic_sim"] = metric_semantic_sim(raw, expected, main.get("semantic"))
        out["bleu"] = bleu_score(raw, expected)
        out["rouge_l"] = rouge_l(raw, expected)

    runs = [
        e.get("raw_answer", "")
        for e in entries
        if (e.get("api_info") or {}).get("status") == "ok" and e.get("raw_answer")
    ]
    out["consistency"] = metric_consistency(runs)
    return out


def compute_task_metrics(
    task: dict[str, Any],
    entries: list[dict[str, Any]],
    verdict_score: float | None = None,
) -> dict[str, Any]:
    """计算单任务单侧指标。entries 为该题该侧全部运行条目。

    失败/截断条目：跳过指标并返回 {skipped, reason}（报告层转为 warning）。
    代码维度：以 code_verify 为权威，文本指标 N/A。
    """
    entries = [e for e in entries or [] if isinstance(e, dict)]
    if not entries:
        return {"skipped": True, "reason": "no_answer"}
    main, skip_reason = _main_entry(entries)
    if skip_reason is not None:
        return {"skipped": True, "reason": skip_reason}

    if task.get("dimension") == CODE_DIMENSION:
        cv = main.get("code_verify") or {}
        if cv.get("status") != "run":
            return {"skipped": True, "reason": "code_not_run"}
        return {
            "code_verify": {"passed": cv.get("passed", 0), "total": cv.get("total", 0)},
            "top1": None, "exact_match": None, "f1": None, "relaxed_accuracy": None,
            "semantic_sim": None, "consistency": None, "bleu": None, "rouge_l": None,
        }

    if task.get("type") == "生成式":
        return _generative(task, entries, main, verdict_score)
    return _discriminative(task, main)
