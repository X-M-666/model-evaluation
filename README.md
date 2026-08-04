# 模型对决评测平台 (Model Duel Evaluation)

输入两个大模型的 API 配置，自动完成「出题 → 双模型作答 → 双盲评审 → 报告展示」的完整评测流程，帮助对比任意两个 LLM 的能力表现。

## 功能特性

- **内置题库**：七大能力维度（知识、推理、代码、语言、指令遵循与对齐、长文本与多模态、效率与稳定性），随机抽题，支持固定种子复现
- **自定义评测集**：上传 JSON / CSV / Markdown / TXT 文件（页面提供各格式模板下载），也可直接粘贴 JSON
- **双盲评审**：评审模型在不知道答案归属的情况下打分，避免偏见；内置复核仲裁
- **重复评测**：重复 N 次自动取平均分 + 标准差，衡量模型稳定性
- **代码验真**：代码题在安全沙箱中运行测试用例验证（沙箱禁止危险操作）
- **效率指标**：记录每次调用的延迟、Token 消耗
- **报告页**：ECharts 雷达图、逐题评分表、综合结论、答卷原文，支持导出 JSON

## 快速启动

```powershell
cd webapp
pip install -r requirements.txt
.\run.ps1
```

启动后访问 `http://localhost:8910`。

也可手动启动：

```bash
cd webapp
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8910
```

## 使用说明

1. 填写模型 A / B 的 API URL（OpenAI 兼容）、API Key、模型名称，可展开高级参数设置 temperature / max_tokens / top_p
2. 选择内置题库（可勾选维度）或自定义评测集（上传文件 / 粘贴 JSON / 从已有评测集选择）
3. 设置重复次数（>1 时自动跑 N 次取平均），点击「开始评测」
4. 实时进度条跟随，完成后自动跳转报告页
5. 页面提供「测试结果案例」按钮，无需 API Key 即可体验完整报告效果

## 自定义评测集格式

统一标准格式（各格式上传后都转换为该结构）：

```json
{
  "name": "评测集名称",
  "description": "可选描述",
  "tasks": [
    {
      "id": "T1",
      "dimension": "知识能力",
      "prompt": "题目内容",
      "expected": "期望答案",
      "rubric_note": "评分标准（可选）"
    }
  ]
}
```

- **JSON**：完整格式（含 test_cases / rubric_note）或简化格式（prompt + expected）
- **CSV**：两列（prompt, expected）或六列（id, dimension, prompt, expected, rubric_note, difficulty）
- **Markdown**：`# 标题` / `## 维度` / `### 题号` / `**题目：**` / `**期望：**` / `**评分标准：**`
- **TXT**：每行 `题目 | 期望答案`（或 TAB 分隔），`# 开头`为维度

页面内可下载各格式示例模板。

## API Key 安全

- API Key 仅保存在后端内存中，**不落盘、不写入日志**
- 历史记录的 config 文件中 Key 一律以 `***` 打码
- 评测记录可通过页面「删除」按钮移除

## 项目结构

```
webapp/
├── backend/
│   ├── main.py            # FastAPI 路由 + 评测调度
│   ├── schemas.py         # Pydantic 模型
│   ├── storage.py         # 历史记录 / 评测集文件库
│   └── engine/
│       ├── tasks.py       # 内置题库（七维度）
│       ├── executor.py    # 双模型并发调用
│       ├── judge.py       # 双盲评审
│       ├── sandbox.py     # 代码安全沙箱
│       ├── parsers.py     # 评测集解析器注册表（JSON/CSV/MD/TXT）
│       ├── datasets.py    # 评测集校验/归一化
│       └── mock.py        # 模拟数据生成（演示用）
├── frontend/
│   ├── index.html         # 配置页
│   └── report.html        # 报告页
├── requirements.txt
└── run.ps1
```

## 技术栈

- 后端：Python FastAPI + httpx（异步并发调用）
- 前端：原生 HTML/JS + ECharts CDN
- 存储：本地 JSON 文件（`.eval/history/` 评测记录，`webapp/data/datasets/` 评测集）
