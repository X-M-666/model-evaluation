# -*- coding: utf-8 -*-
"""模拟评测数据生成器：每次点击随机生成不同题目、不同答案、不同评分，无需真实 API。"""
from __future__ import annotations

import random
import time
from backend.engine.tasks import build_task_set
from backend.storage import create_job_id, save_config, save_task_set, save_answers, save_verdict, save_report

# 模型名池（每次随机抽两个）
MODEL_POOL = [
    "GPT-4o", "GPT-4o Mini", "Claude Sonnet 4", "Claude Haiku 3.5",
    "Gemini 2.5 Pro", "Gemini 2.5 Flash", "Llama 3.1 70B", "Llama 3.1 405B",
    "Qwen 3 235B", "DeepSeek V3", "Grok 3", "Mistral Large 2",
]
MODEL_URLS = {
    "GPT-4o": "https://api.openai.com/v1",
    "GPT-4o Mini": "https://api.openai.com/v1",
    "Claude Sonnet 4": "https://api.anthropic.com/v1",
    "Claude Haiku 3.5": "https://api.anthropic.com/v1",
    "Gemini 2.5 Pro": "https://generativelanguage.googleapis.com/v1",
    "Gemini 2.5 Flash": "https://generativelanguage.googleapis.com/v1",
    "Llama 3.1 70B": "https://api.together.xyz/v1",
    "Llama 3.1 405B": "https://api.together.xyz/v1",
    "Qwen 3 235B": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "DeepSeek V3": "https://api.deepseek.com/v1",
    "Grok 3": "https://api.x.ai/v1",
    "Mistral Large 2": "https://api.mistral.ai/v1",
}

# ---- 每个维度的多套答案模板 ----
ANSWER_TEMPLATES = {
    "知识能力": [
        "Q1-A 木星\nQ2-A ATP\nQ3-A 苏必利尔湖",
        "Q1是A木星，太阳系体积最大的行星。Q2是A ATP，细胞的能量货币。Q3是A苏必利尔湖，世界面积最大的淡水湖。",
        "1.A 木星  2.A ATP  3.A 苏必利尔湖",
        "Q1-A 木星（体积最大的气态巨行星）\nQ2-A ATP（三磷酸腺苷）\nQ3-A 苏必利尔湖（面积最大的淡水湖）",
    ],
    "推理能力": [
        "四个角必为偶数{2,4,6,8}，对角和为10。8种摆放得到8624、6248、2486、4862、8426、4268、2684、6842，总和44440。",
        "通过枚举所有满足条件的幻方（含旋转与镜像共8种），四角数字为{2,4,6,8}的偶数组合，对角和固定为10。所有可能的四位数：8624、6248、2486、4862、8426、4268、2684、6842。求和过程：8624+6248+2486+4862+8426+4268+2684+6842=44440。",
        "分析：正中为5，四角必为偶数。对角互补和为10。旋转4种×镜像2种=8种摆放。\n四位数：8624, 6248, 2486, 4862, 8426, 4268, 2684, 6842\n总和=44440",
        "四角数字只能取自{2,4,6,8}，共8种旋转/镜像排列：8624、6248、2486、4862、8426、4268、2684、6842，总和为44440。",
    ],
    "代码能力": [
        "```python\ndef merge(intervals):\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for s, e in intervals[1:]:\n        if s <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], e)\n        else:\n            merged.append([s, e])\n    return merged\n# O(n log n)\n```",
        "```python\ndef merge(intervals):\n    intervals.sort()\n    res = []\n    for interval in intervals:\n        if not res or res[-1][1] < interval[0]:\n            res.append(interval)\n        else:\n            res[-1][1] = max(res[-1][1], interval[1])\n    return res\n# 时间复杂度 O(n log n)\n```",
        "```python\ndef merge(intervals):\n    if not intervals:\n        return []\n    intervals = sorted(intervals, key=lambda x: x[0])\n    result = [intervals[0]]\n    for start, end in intervals[1:]:\n        if start <= result[-1][1]:\n            result[-1][1] = max(result[-1][1], end)\n        else:\n            result.append([start, end])\n    return result\n# Time: O(n log n), Space: O(n)\n```",
    ],
    "语言能力": [
        "Q1-B 约 1%\nQ2: 蒸腾拉力提供根系吸水与向上运输的动力；蒸腾散热可降低叶面温度防止灼伤；蒸腾促进根部对矿质元素的吸收运输。\nQ3: 植物蒸腾作用是指水分以气态从叶片散失的过程，约99%水分通过蒸腾散失，主要意义为促进水分运输、降低叶温、促进矿质吸收。",
        "Q1: B\nQ2: 第一，蒸腾拉力是水分和无机盐向上运输的主要动力；第二，蒸腾散热可降低叶面温度避免灼伤；第三，蒸腾促进根对矿质元素的吸收与运输。\nQ3: 植物蒸腾作用散失约99%的水分，其主要意义在于促进水分与矿质运输、降低叶温，是植物重要的生理过程。",
        "Q1-B\nQ2三点意义：①蒸腾拉力驱动水分和无机盐向上运输 ②散热降低叶面温度防止灼伤 ③促进根部矿质元素吸收运输\nQ3概括：蒸腾作用是植物以气态散失水分的过程，约99%水分经此散失，对水分运输、降温和矿质吸收有重要意义。",
    ],
    "指令遵循与对齐": [
        '{"week":"W34","summary":"团队完成订单模块重构并修复支付超时bug，发布v2.3版本","tasks":[{"title":"订单模块重构","status":"done"},{"title":"支付超时bug修复","status":"done"},{"title":"退款流程改造","status":"pending"}],"risk":{"level":"high","description":"第三方支付渠道接口即将变更"},"next_week":{"task":"退款流程改造","priority":1}}',
        '{"week":"W34","summary":"完成订单重构与支付bug修复并发布v2.3","tasks":[{"title":"订单模块重构","status":"已完成"},{"title":"支付超时修复","status":"已完成"},{"title":"退款改造","status":"进行中"}],"risk":{"level":"中","description":"第三方支付接口即将变更"},"next_week":{"task":"退款流程改造","priority":1}}',
        '{"week":"W34","summary":"完成订单重构与超时修复，发布v2.3版本","tasks":[{"title":"订单模块重构","status":"done"},{"title":"支付超时修复","status":"done"},{"title":"退款流程改造","status":"pending"}],"risk":{"level":"high","description":"第三方支付渠道接口即将变更"},"next_week":{"task":"退款流程改造","priority":1}}',
    ],
    "长文本与多模态": [
        "Q1: 周明远（理事长）、林晓雯（农技员）、陈子昂（电商运营）、苏芷若（品牌设计）。\nQ2: 销售额2,360万元；净利润480万元；2026年目标3,000万元。\nQ3: 2021年星澜市示范农民专业合作社；2023年5月省级生态农场示范点；2025年10月最受欢迎农产品金奖。\nQ4: 青禾农场2014年成立，2017年引入稻鱼共生，2023年获评省级生态农场，2025年底社员86户销售额2360万。",
        "Q1：周明远（理事长）、林晓雯（农技员）、陈子昂（电商运营）、苏芷若（品牌设计），共4人。\nQ2：全年销售额2360万元，净利润480万元，2026年目标突破3000万元。\nQ3：2021年获星澜市示范合作社称号；2023年5月获评省级生态农场示范点；2025年10月青禾香米获最受欢迎农产品金奖。\nQ4：2014年成立主营水稻，2017年引入稻鱼共生模式，2019年建成智慧大棚，2025年底社员86户、面积1200亩、销售额2360万。",
        "Q1: 周明远、林晓雯、陈子昂、苏芷若。Q2: 销售额2360万元，净利润480万元，2026年目标3000万元。Q3: 2021年星澜市示范合作社，2023年5月省级生态农场，2025年10月最受欢迎农产品金奖。Q4: 2014年成立种植水稻，2017年引入稻鱼共生，2023年获评省级示范，2025年底社员86户销售额2360万。",
    ],
    "效率与稳定性": [
        "```python\nimport sys\nfrom collections import Counter\ndef top5(text):\n    counts = Counter(c for c in text if '\\u4e00' <= c <= '\\u9fff')\n    sorted_chars = sorted(counts.items(), key=lambda x: (-x[1], x[0]))\n    for ch, cnt in sorted_chars[:5]:\n        print(ch, cnt)\n# O(n) 单遍扫描\n```",
        "```python\nfrom collections import Counter\ndef top5(text):\n    cn = [c for c in text if '\\u4e00' <= c <= '\\u9fff']\n    freq = Counter(cn)\n    for ch, cnt in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:5]:\n        print(f'{ch} {cnt}')\n# O(n) time\n```",
        "```python\nimport sys\nfrom collections import Counter\ndef solve():\n    text = sys.stdin.read()\n    freq = Counter()\n    for ch in text:\n        if '\\u4e00' <= ch <= '\\u9fff':\n            freq[ch] += 1\n    for ch, cnt in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:5]:\n        print(ch, cnt)\nif freq: solve()\nelse: print('EMPTY')\n# O(n) single pass\n```",
    ],
}

# 评审依据模板（按胜方类型）
BASIS_TEMPLATES = {
    "tie": [
        "双方回答均正确完整，表述略有差异但质量等价，均为满分。",
        "两模型输出功能等价，细节处理相当，无明显优劣。",
        "双方答案均覆盖全部要点，准确度一致，判为平局。",
        "两份回答质量相当，各有特色但整体水平相同。",
    ],
    "answer_x": [
        "X回答更完整准确，覆盖所有要点；Y回答有轻微遗漏或表述不够精确。",
        "X的推理过程更清晰，结论准确；Y结论正确但过程略简略。",
        "X在细节处理上更到位，格式更规范；Y基本正确但有小瑕疵。",
        "X的回答信息量更大且准确，Y的回答稍有不足。",
    ],
    "answer_y": [
        "Y回答更完整准确，覆盖所有要点；X回答有轻微遗漏或表述不够精确。",
        "Y的推理过程更清晰，结论准确；X结论正确但过程略简略。",
        "Y在细节处理上更到位，格式更规范；X基本正确但有小瑕疵。",
        "Y的回答信息量更大且准确，X的回答稍有不足。",
    ],
}


def _pick_model(exclude: str = "") -> str:
    """从模型池中随机选一个不重复的模型名。"""
    pool = [m for m in MODEL_POOL if m != exclude]
    return random.choice(pool)


def _random_api_info(base_latency: int = 500) -> dict:
    return {
        "status": "ok",
        "attempts": 1,
        "truncated": False,
        "error": None,
        "latency_ms": base_latency + random.randint(-200, 300),
        "prompt_tokens": random.randint(200, 600),
        "completion_tokens": random.randint(50, 350),
        "repeat_index": 1,
    }


def _random_scores(tasks: list[dict]) -> list[dict]:
    """为每道题生成随机评分。"""
    scores = []
    for t in tasks:
        dim = t["dimension"]
        is_code = dim == "代码能力"
        # 代码题更容易平局，其他题随机
        if is_code:
            x = random.choice([9, 9.5, 10, 10])
            y = random.choice([9, 9.5, 10, 10])
        else:
            x = round(random.uniform(6, 10), 1)
            y = round(random.uniform(6, 10), 1)

        if x == y:
            winner = "tie"
        elif x > y:
            winner = "answer_x"
        else:
            winner = "answer_y"

        basis = random.choice(BASIS_TEMPLATES[winner])
        arbiter_note = ""
        if abs(x - y) <= 0.5 and winner != "tie":
            arbiter_note = f"分差仅{abs(x-y)}分，维持原判。"

        scores.append({
            "id": t["id"],
            "dimension": dim,
            "answer_x": x,
            "answer_y": y,
            "winner": winner,
            "basis": basis,
            "arbiter_note": arbiter_note,
        })
    return scores


def generate_mock_job(seed: int | None = None) -> str:
    """生成一条完整的模拟评测记录，返回 job_id。"""
    if seed is None:
        seed = int(time.time()) % 100000

    job_id = create_job_id()
    model_a_name = _pick_model()
    model_b_name = _pick_model(exclude=model_a_name)
    model_a_url = MODEL_URLS.get(model_a_name, "https://api.example.com/v1")
    model_b_url = MODEL_URLS.get(model_b_name, "https://api.example.com/v1")

    # 1. 保存配置
    save_config(job_id, {
        "model_a": {"name": model_a_name, "url": model_a_url, "key": "mock"},
        "model_b": {"name": model_b_name, "url": model_b_url, "key": "mock"},
        "dims": None,
        "seed": seed,
    })

    # 2. 生成任务集（随机种子决定抽哪些题）
    task_set = build_task_set(seed=seed)
    save_task_set(job_id, task_set)

    # 3. 构造模拟答卷
    answers_a = {
        "model": model_a_name,
        "api": {"name": model_a_name, "url": model_a_url},
        "note": "模拟数据（无需真实 API）",
        "answers": [],
    }
    answers_b = {
        "model": model_b_name,
        "api": {"name": model_b_name, "url": model_b_url},
        "note": "模拟数据（无需真实 API）",
        "answers": [],
    }

    for t in task_set["tasks"]:
        tid = t["id"]
        dim = t["dimension"]
        templates = ANSWER_TEMPLATES.get(dim, ["(模拟回答)"])

        # 每个模型随机选一套答案模板
        ans_a_text = random.choice(templates)
        ans_b_text = random.choice(templates)
        # 保证两个模型答案不同（如果池子里有多套）
        if len(templates) > 1 and ans_a_text == ans_b_text:
            ans_b_text = random.choice([t for t in templates if t != ans_a_text] or templates)

        is_code = dim == "代码能力"
        base_latency = random.randint(300, 900)

        for ans_dict, raw in [(answers_a, ans_a_text), (answers_b, ans_b_text)]:
            entry = {
                "id": tid,
                "raw_answer": raw,
                "api_info": _random_api_info(base_latency),
            }
            if is_code:
                passed = random.choice([4, 5, 5, 5])  # 大多数全过
                entry["code_verify"] = {"passed": passed, "total": 5}
            ans_dict["answers"].append(entry)

    save_answers(job_id, "a", answers_a)
    save_answers(job_id, "b", answers_b)

    # 4. 构造 verdict
    scores = _random_scores(task_set["tasks"])
    dim_totals = {}
    for s in scores:
        dim = s["dimension"]
        if dim not in dim_totals:
            dim_totals[dim] = {"x": 0, "y": 0}
        dim_totals[dim]["x"] += s["answer_x"]
        dim_totals[dim]["y"] += s["answer_y"]

    total_x = round(sum(d["x"] for d in dim_totals.values()), 1)
    total_y = round(sum(d["y"] for d in dim_totals.values()), 1)
    if total_x > total_y:
        winner_model = model_a_name
    elif total_y > total_x:
        winner_model = model_b_name
    else:
        winner_model = "平局"

    verdict = {
        "meta": {"total": 7, "valid": 7, "invalid": 0, "tie_arbitrated": 0},
        "scores": scores,
        "per_dimension": dim_totals,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": {"answer_x": f"{model_a_name} (模拟)", "answer_y": f"{model_b_name} (模拟)"},
        "conclusion": f"{winner_model} 以 {max(total_x,total_y)}:{min(total_x,total_y)} 胜出（模拟数据）",
        "winner_model": winner_model,
    }
    save_verdict(job_id, verdict)
    save_report(job_id, {
        "config": {"model_a": {"name": model_a_name, "url": model_a_url},
                   "model_b": {"name": model_b_name, "url": model_b_url},
                   "dims": None, "seed": seed, "repeat_n": 1},
        "tasks": task_set,
        "answers_a": answers_a,
        "answers_b": answers_b,
        "verdict": verdict,
    })

    return job_id
