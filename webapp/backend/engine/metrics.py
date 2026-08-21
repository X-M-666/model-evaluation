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
import statistics
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


# RAG/上下文忠实性（迭代四）：答案 vs 参考文档的 n-gram 支持度阈值
GROUNDING_SUPPORT_THRESHOLD = 0.35


def token_efficiency(
    score: float,
    prompt: int = 0,
    completion: int = 0,
) -> dict[str, Any]:
    """每 1K token 得分（迭代十二：效率归一化，论文 token 效率落地）。

    Returns:
        {"score_per_1k_tokens": score / max(tokens/1000, 1e-6),
         "cost_per_score": tokens / max(score, 1e-6)（token/分）,
         "tokens": total}
        无可得分（score<=0 或 tokens<=0）时 score_per_1k_tokens=None。
    """
    tokens = max(int(prompt or 0), 0) + max(int(completion or 0), 0)
    if tokens <= 0 or score <= 0:
        return {"score_per_1k_tokens": None, "cost_per_score": None, "tokens": tokens}
    eff = score / max(tokens / 1000.0, 1e-6)
    return {
        "score_per_1k_tokens": round(eff, 4),
        "cost_per_score": round(tokens / max(score, 1e-6), 4),
        "tokens": tokens,
    }


def normalize_efficiency(effs: list[float | None]) -> list[float | None]:
    """归一化效率得分：按全局最大值缩放至 0-1（1.0=当前最高效率）。

    跨双方及（如有）历史最大值归一，供「选型决策」对比：
    避免只看绝对分数、兼顾成本差异（论文核心）。全 None 时输出全 None。
    """
    vals = [e for e in effs if e is not None and e > 0]
    if not vals:
        return [None] * len(effs)
    gmax = max(vals)
    return [round(e / gmax, 4) if e is not None else None for e in effs]


# 边际收益判定窗口与阈值（迭代十二：成本-性能前沿线拐点，论文边际成本落地）
FRONTIER_WINDOW = 2
FRONTIER_RATIO = 0.3


def efficiency_frontier(
    points: list[tuple[float, float]],
) -> dict[str, Any]:
    """成本–性能前沿线 + 边际收益拐点（纯函数，零依赖）。

    points: [(本题 tokens, 本题得分), ...]（顺序不限，内部按 tokens 升序）。
    输出累计 token/累计得分序列，并按「最近 N 题每 1K token 增益 < 全局
    中位数 30%」判定拐点（边际收益递减起始点）。

    Returns: {"cum_tokens", "cum_score", "knee_index", "knee_tokens",
              "knee_score", "note"}；有效边际不足时 knee_* 为 None。
    """
    pts = sorted(points, key=lambda p: float(p[0]))
    cum_tokens: list[float] = []
    cum_score: list[float] = []
    ct = cs = 0.0
    for t, s in pts:
        ct += float(t or 0)
        cs += float(s or 0)
        cum_tokens.append(round(ct, 2))
        cum_score.append(round(cs, 2))
    n = len(pts)
    marginals: list[float | None] = [None] * n
    for i in range(1, n):
        j = max(0, i - FRONTIER_WINDOW)
        dt = cum_tokens[i] - cum_tokens[j]
        ds = cum_score[i] - cum_score[j]
        marginals[i] = ds / max(dt / 1000.0, 1e-6) if dt > 0 and ds > 0 else None
    valid = [m for m in marginals if m is not None]
    knee_index: int | None = None
    if len(valid) >= 2:
        median_m = statistics.median(valid)
        for i, m in enumerate(marginals):
            if m is not None and m < FRONTIER_RATIO * median_m:
                knee_index = i
                break
    return {
        "cum_tokens": cum_tokens,
        "cum_score": cum_score,
        "knee_index": knee_index,
        "knee_tokens": cum_tokens[knee_index] if knee_index is not None else None,
        "knee_score": cum_score[knee_index] if knee_index is not None else None,
        "note": "拐点=最近 N 题每 1K token 增益显著放缓处（边际收益递减起始点）",
    }


# 判别力档位（迭代十二：基线判别力洞察）见 tasks.discriminative_band

# 答案冗余度阈值（迭代十二 N1，论文冗余度 R）：>0.5 判同质化、≤0.3 判分化
REDUNDANCY_SAME_THRESHOLD = 0.5
REDUNDANCY_DISTINCT_THRESHOLD = 0.3
# 数值漂移阈值（迭代十二 N2，论文错误分类法）：相对真值偏差 >5% 判漂移
NUMERIC_DRIFT_RATIO = 0.05


def answer_redundancy(
    raw_a: str,
    raw_b: str,
    sem_a: dict[str, Any] | None = None,
    sem_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """双模型答案冗余度/分化度（论文 N1）：输出嵌入余弦相似度。

    优先复用执行期采集的 semantic vector；缺失时 n-gram 余弦兜底。
    Returns: {"cosine": 0-1, "band": "same-ish"|"distinct"|"ok", "source"}
    """
    va = (sem_a or {}).get("vector") if isinstance(sem_a, dict) else None
    vb = (sem_b or {}).get("vector") if isinstance(sem_b, dict) else None
    if va is not None and vb is not None:
        value = float(cosine(va, vb))
        source = "embedding"
    else:
        if not (raw_a or "").strip() or not (raw_b or "").strip():
            return {"cosine": None, "band": None, "source": "no_answer"}
        value = float(cosine(ngram_vec(raw_a), ngram_vec(raw_b)))
        source = "ngram"
    value = round(value, 4)
    if value > REDUNDANCY_SAME_THRESHOLD:
        band = "same-ish"
    elif value <= REDUNDANCY_DISTINCT_THRESHOLD:
        band = "distinct"
    else:
        band = "ok"
    return {"cosine": value, "band": band, "source": source}


def numeric_drift(
    runs: list[str],
    expected: str,
) -> dict[str, Any]:
    """数值漂移检测（论文 N2）：计算类误差累积致结果相对真值偏差 >5%。

    从各次运行答案中抽取数值与期望比对（期望无数值 → 仅做轮次间一致性）；
    轮次间数值不一致亦判漂移。无数值可提取 → {drift: null}。
    Returns: {"drift": bool|None, "max_delta_ratio": float|None,
              "flag": "numeric_drift"|None, "values": [...]}
    """
    runs = [r for r in (runs or []) if r and str(r).strip()]
    if not runs:
        return {"drift": None, "max_delta_ratio": None, "flag": None, "values": []}
    values = [v for v in (_try_num(r) for r in runs) if v is not None]
    exp = _try_num(expected)
    if not values:
        return {"drift": None, "max_delta_ratio": None, "flag": None, "values": []}
    drift = False
    max_ratio: float | None = None
    ratios: list[float] = []
    if exp is not None:
        for v in values:
            ratio = abs(v - exp) / max(abs(exp), 1e-9)
            ratios.append(round(ratio, 4))
            if ratio > NUMERIC_DRIFT_RATIO:
                drift = True
        max_ratio = max(ratios)
    elif len(values) >= 2 and max(values) - min(values) > 1e-9:
        base = max(abs(v) for v in values) or 1e-9
        max_ratio = round((max(values) - min(values)) / base, 4)
        drift = max_ratio > NUMERIC_DRIFT_RATIO
    return {
        "drift": drift,
        "max_delta_ratio": max_ratio,
        "flag": "numeric_drift" if drift else None,
        "values": values,
    }

# coherence（迭代十二）：推导一致性轻量代理（零依赖零网络）
_CONCLUSION_MARKERS = ("因此", "所以", "综上", "由此可见", "结论是", "即", "故")
_NEGATION_WORDS = ("不", "没", "非", "无", "否", "错", "假", "不可能")
_CONTRADICTION_KEYS = ("是", "等于", "为", "可以", "能", "属于", "需要")


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句（纯规则，零依赖）。"""
    parts = re.split(r"[。！？!?\n；;]", text or "")
    return [p.strip() for p in parts if p.strip()]


def _has_negation(s: str) -> bool:
    return any(w in s for w in _NEGATION_WORDS)


def _conclusion_part(sentences: list[str]) -> tuple[list[str], int]:
    """提取结论句：命中连接词（因此/所以/结论是…）起，否则取末句。"""
    for i, s in enumerate(sentences):
        if any(m in s for m in _CONCLUSION_MARKERS):
            return sentences[i:], i
    if sentences:
        return [sentences[-1]], len(sentences) - 1
    return [], -1


def _contradiction_signal(premises: list[str], conclusion: str) -> bool:
    """轻量自我矛盾检测：同一谓词断言在前提与结论中极性相反。

    要求结论与某前提共享同一谓词键（是/等于/为/可以/能/属于/需要），
    且两者对同一键的否定极性相反（前提肯定+结论否定，或反之），
    才判为逻辑矛盾信号（例如“该产品可以食用” vs “因此该产品不可以食用”）。
    无共享谓词或极性一致时不预警，避免单句/无关句误报。
    """
    if not conclusion:
        return False
    c_neg = _has_negation(conclusion)
    c_keys = [k for k in _CONTRADICTION_KEYS if k in conclusion]
    if not c_keys:
        return False
    for p in premises:
        p_neg = _has_negation(p)
        shared = [k for k in c_keys if k in p]
        if shared and p_neg != c_neg:
            return True
    return False


@_metric("coherence")
def coherence(raw: str, prompt: str) -> dict[str, Any]:
    """生成式答案「前提-结论」推导自洽性（0-1，迭代十二）。

    轻量 n-gram/规则实现：切句 → 以「因此/所以/结论是」等连接词定位结论句
    （无连接词取末句）→ 前提与结论 n-gram 余弦相关性；检测明显自我矛盾
    （同实体断言为真又为假）。矛盾信号时得分贴近 0 并附 flag 供报告高亮。
    """
    sentences = _split_sentences(raw)
    if not sentences:
        return {"score": 0.0, "flag": None}
    conclusions, conc_pos = _conclusion_part(sentences)
    premises = sentences[:conc_pos] if conc_pos > 0 else []
    conclusion = "".join(conclusions) if conclusions else (sentences[-1] if sentences else "")
    premise_text = "".join(premises) if premises else (sentences[0] if sentences else "")
    if not premise_text or not conclusion:
        return {"score": 0.0, "flag": None}
    corr = cosine(ngram_vec(premise_text), ngram_vec(conclusion))
    if _contradiction_signal(premises or sentences, conclusion):
        return {"score": round(min(corr, 0.15), 4), "flag": "logical_contradiction"}
    rel = metric_answer_relevancy(raw, prompt)
    score = 0.7 * corr + 0.3 * rel
    return {"score": round(min(score, 1.0), 4), "flag": None}


@_metric("grounding_faithfulness")
def metric_grounding_faithfulness(answer: str, context: str) -> float:
    """忠实度：答案与参考文档的 n-gram 余弦（纯 n-gram、零依赖）。

    答案大量援引/复述文档片段 → 接近 1；与文档无关的自由发挥 → 接近 0。
    """
    ans, ctx = (answer or "").strip(), (context or "").strip()
    if not ans or not ctx:
        return 0.0
    return round(cosine(ngram_vec(ans), ngram_vec(ctx)), 4)


@_metric("answer_relevancy")
def metric_answer_relevancy(answer: str, prompt: str) -> float:
    """相关性：答案与题面 prompt 的 n-gram 余弦（纯 n-gram、零依赖）。

    答非所问（复述文档但未回应问题）时相关度低，与忠实度互补。
    """
    ans, q = (answer or "").strip(), (prompt or "").strip()
    if not ans or not q:
        return 0.0
    return round(cosine(ngram_vec(ans), ngram_vec(q)), 4)


def _discriminative(
    task: dict[str, Any],
    main: dict[str, Any],
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = main.get("raw_answer", "")
    expected_list = [e for e in _case_expecteds(task) if e]
    if not expected_list:
        return {"skipped": True, "reason": "no_expected"}
    extracted = extract_per_case(task, raw)
    pairs = list(zip(extracted, expected_list))
    matched = [_case_match(x, e) for x, e in pairs if e]

    out: dict[str, Any] = {
        "top1": metric_top1(matched),
        "exact_match": metric_exact_match(matched),
        "f1": metric_f1(pairs),
        "relaxed_accuracy": metric_relaxed_accuracy(pairs),
    }
    # 迭代十二（N2）：数值漂移监测（>5% 判漂移）；期望无数值则做轮次间一致性
    numeric_exps = [e for e in expected_list if _try_num(e) is not None]
    if numeric_exps:
        runs = [
            e.get("raw_answer", "")
            for e in (entries or [])
            if (e.get("api_info") or {}).get("status") == "ok" and e.get("raw_answer")
        ]
        out["numeric_drift"] = numeric_drift(runs, numeric_exps[0])
    return out


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
    out["coherence"] = coherence(raw, task.get("prompt", ""))
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
    return _discriminative(task, main, entries)
