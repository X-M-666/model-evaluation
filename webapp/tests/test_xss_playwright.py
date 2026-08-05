# -*- coding: utf-8 -*-
"""Issue #7：浏览器级 XSS 防护验证（Playwright，无 playwright 时自动跳过）。

覆盖四个验收面：
1. 含脚本载荷的数据集上传（name/dimension/description/prompt 全部注入）→
   列表渲染不执行载荷、不产生注入节点
2. 删除按钮在 CSP 下走事件委托仍可用（confirm 弹出 → DELETE 生效）
3. 报告页 ECharts tooltip（richText 渲染）悬停不产生 HTML 注入节点
4. 三个页面在 CSP 下无 JS 异常、无 CSP 违规日志（核心功能不回归）

服务以同进程线程方式启动（端口随机），storage.BASE_DIR 重定向到临时目录，
避免污染真实 .eval/history。
"""
from __future__ import annotations

import json
import threading
import time

import pytest
import uvicorn

from backend import storage
from backend.engine.mock import prepare_mock_job
from backend.main import app, _jobs

playwright = pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright, expect

# ---- 注入载荷：覆盖 HTML 元素、JS 引号逃逸、SVG onload、script 标签四类 ----
# 载荷以 alert 形式检测执行（dialog 事件），CSP 下内联脚本/事件处理器全部失效
XSS_NAME = '<img src=x onerror="alert(\'XSS_IMG\')">'
XSS_DIM = "';alert('XSS_QUOTE');//"
XSS_DESC = '<svg onload="alert(\'XSS_SVG\')">'
XSS_PROMPT = '<script>alert(\'XSS_SCRIPT\')</script>'
XSS_QID = '<img src=x onerror="alert(\'XSS_QID\')">'

BASE: str | None = None
SERVER: uvicorn.Server | None = None


@pytest.fixture(scope="module", autouse=True)
def _server_and_browser(tmp_path_factory):
    """同进程起 uvicorn（端口随机）+ 存储隔离到临时目录 + 浏览器实例。"""
    global BASE, SERVER
    orig_base = storage.BASE_DIR
    orig_ds = storage.DATASETS_DIR
    storage.BASE_DIR = tmp_path_factory.mktemp("history")
    storage.DATASETS_DIR = tmp_path_factory.mktemp("datasets")
    for jid in list(_jobs):
        _jobs.pop(jid)

    SERVER = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=SERVER.run, daemon=True)
    thread.start()
    while not SERVER.started:
        time.sleep(0.01)
    port = SERVER.servers[0].sockets[0].getsockname()[1]
    BASE = f"http://127.0.0.1:{port}"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()

    SERVER.should_exit = True
    thread.join(timeout=10)
    storage.BASE_DIR = orig_base
    storage.DATASETS_DIR = orig_ds


@pytest.fixture(autouse=True)
def _page(_server_and_browser):
    page = _server_and_browser.new_page()
    yield page
    page.close()


def _upload_dataset(page, name: str, dim: str = "知识能力", desc: str = "",
                    prompt: str = "1+1?", qid: str = "T1") -> None:
    payload = json.dumps({
        "name": name,
        "description": desc,
        "tasks": [{"id": qid, "dimension": dim, "prompt": prompt, "expected": "2"}],
    }, ensure_ascii=False)
    r = page.request.post(
        f"{BASE}/api/datasets/upload-json",
        data=json.dumps({"content": payload}),
        headers={"Content-Type": "application/json"},
    )
    assert r.ok, r.text


def test_payload_dataset_renders_without_execution(_page):
    dialogs = []
    _page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    _upload_dataset(_page, XSS_NAME, dim=XSS_DIM, desc=XSS_DESC, prompt=XSS_PROMPT)
    _page.goto(f"{BASE}/")
    _page.check('input[name="ds-mode"][value="custom"]')
    _page.wait_for_selector(".ds-item")

    assert not dialogs, f"载荷在页面执行了: {dialogs}"
    # 注入内容不得产生任何 HTML 元素
    assert _page.locator('.ds-item img, .ds-item svg, .ds-item script').count() == 0
    # 维度文本应原样展示（含引号载荷），只是纯文本
    assert "XSS_QUOTE" in _page.locator(".ds-item").first.text_content()
    # 功能不回归：dims 复选框正常渲染（内置七维度）
    assert _page.locator("#dims-box input[type=checkbox]").count() == 7


def test_dataset_delete_via_delegation_under_csp(_page):
    """自包含：先上传两个数据集，删除一个，目标消失且其余保留（不依赖模块内测试顺序）。"""
    _upload_dataset(_page, "保留数据集", dim="代码能力")
    _upload_dataset(_page, "删除测试集")
    _page.goto(f"{BASE}/")
    _page.check('input[name="ds-mode"][value="custom"]')
    _page.wait_for_selector(".ds-item .btn-danger")

    dialogs = []
    _page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    _page.locator(".ds-item", has_text="删除测试集").locator(".btn-danger").click()

    expect(_page.locator(".ds-item", has_text="删除测试集")).to_have_count(0, timeout=5000)
    assert _page.locator(".ds-item", has_text="保留数据集").count() >= 1
    assert any("删除" in m for m in dialogs), "事件委托应触发 confirm"


def test_report_tooltip_does_not_render_html(_page):
    data = prepare_mock_job(seed=2025)
    job_id = data["job_id"]

    # 篡改任务集：把第一题 id 换成注入载荷（进入图表类目/tooltip 路径）
    ts_path = storage.BASE_DIR / job_id / "tasks.json"
    ts = json.loads(ts_path.read_text(encoding="utf-8"))
    ts["tasks"][0]["id"] = XSS_QID
    ts_path.write_text(json.dumps(ts, ensure_ascii=False), encoding="utf-8")

    scores = [
        {"id": t["id"], "round": 1, "answer_x": 8, "answer_y": 7, "note": ""}
        for t in ts["tasks"]
    ]
    r = _page.request.post(
        f"{BASE}/api/eval/{job_id}/review",
        data=json.dumps({"scores": scores}),
        headers={"Content-Type": "application/json"},
    )
    assert r.ok, r.text

    dialogs = []
    _page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    _page.goto(f"{BASE}/report.html?job={job_id}")
    _page.wait_for_selector("#chartScore canvas")
    # 悬停触发 tooltip（axis 触发），若走 HTML 渲染则载荷会进入 tooltip DOM 并执行
    box = _page.locator("#chartScore canvas").bounding_box()
    _page.mouse.move(box["x"] + box["width"] * 0.15, box["y"] + box["height"] * 0.5)
    _page.wait_for_timeout(600)

    assert not dialogs, f"tooltip 载荷执行了: {dialogs}"
    assert _page.locator('img[src="x"], svg[onload]').count() == 0
    # 报告主体正常渲染（功能不回归）
    assert _page.locator("#winnerCard").text_content().strip()


def test_pages_have_no_js_or_csp_errors(_page):
    page_errors, csp_violations = [], []
    _page.on("pageerror", lambda e: page_errors.append(str(e)))
    _page.on("console", lambda m: csp_violations.append(m.text)
             if "Content-Security-Policy" in m.text else None)

    for path in ["/", "/review.html", "/report.html"]:
        _page.goto(f"{BASE}{path}")
        _page.wait_for_timeout(400)

    assert not page_errors, f"页面 JS 异常: {page_errors}"
    assert not csp_violations, f"CSP 违规: {csp_violations}"


def test_review_page_full_flow_renders_and_submits(_page):
    """mock 答案（无 code_verify 字段）评审页可完整渲染并提交（回归 cv=null 崩溃）。"""
    data = prepare_mock_job(seed=2026)
    job_id = data["job_id"]

    _page.goto(f"{BASE}/review.html?job={job_id}")
    _page.wait_for_selector(".q-card", timeout=10000)
    assert _page.locator(".q-card").count() == len(data["task_set"]["tasks"])
    assert _page.locator(".answer-box").count() >= 2

    for e in _page.locator('input[data-field="x"]').all():
        e.fill("5")
    for e in _page.locator('input[data-field="y"]').all():
        e.fill("5")
    _page.locator("#submitBtn").click()
    _page.wait_for_url("**/report.html?job=*", timeout=15000)
    assert "report.html" in _page.url
