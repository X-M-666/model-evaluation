# -*- coding: utf-8 -*-
"""迭代十一：文本对战页 + 任务页向导 E2E（Playwright，无 playwright 时自动跳过）。

覆盖：
1. battle.html 渲染：设置面板/顶栏入口，无 JS 异常
2. 对战全流程：抽题（route 拦截）→ 双栏流式（拦截 SSE）→ 5 档投票 →
   手动「下一题」才进入下一道并作答 → 上一题回看走缓存（不重复请求）→
   结果统计弹窗（总体统计 + 逐题评分）
3. 盲评模式：代号显示 + 真实身份隐藏 + 每题左右随机（Math.random 固定方向，
   断言 A/B 输出按方向落位）→ 全部评完才可「揭晓模型身份」→ 揭晓后显示真实名
4. 会话持久化：开始对局 → 投票 → reload 后对局恢复（当前题/票数/缓存/模型配置），
   缓存命中不重复请求；流式被打断恢复后自动重新作答；重新对战清空会话
5. tasks.html 新建任务向导：4 步流转 + 提交（拦截 /api/benchmark）

服务同进程启动（端口随机），存储隔离到临时目录。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
import uvicorn

from backend import storage
from backend.main import app, _jobs

playwright = pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright

BASE: str | None = None
SERVER: uvicorn.Server | None = None

FAKE_QUESTIONS = {
    "questions": [
        {"question_id": "Q1", "category": "数学能力", "prompt": "1+1等于几？", "context": ""},
        {"question_id": "Q2", "category": "知识能力", "prompt": "介绍一下长城", "context": ""},
    ],
    "total": 2,
}

FAKE_SSE = (
    "data: {\"side\": \"a\", \"delta\": \"答案A1\"}\n\n"
    "data: {\"side\": \"b\", \"delta\": \"答案B1\"}\n\n"
    "data: {\"side\": \"a\", \"delta\": \"答案A2\"}\n\n"
    "data: {\"side\": \"b\", \"done\": true}\n\n"
    "data: {\"side\": \"a\", \"done\": true}\n\n"
)


@pytest.fixture(scope="module", autouse=True)
def _server_and_browser(tmp_path_factory):
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


def _fill_models(page):
    page.fill("#mA-url", "https://8.8.8.8/v1")
    page.fill("#mA-key", "k")
    page.fill("#mA-name", "模型甲")
    page.fill("#mB-url", "https://8.8.8.8/v1")
    page.fill("#mB-key", "k")
    page.fill("#mB-name", "模型乙")


def test_battle_page_renders_no_js_errors(_page):
    errors = []
    _page.on("pageerror", lambda e: errors.append(str(e)))
    _page.goto(f"{BASE}/battle.html")
    _page.wait_for_selector("#startBtn")
    assert _page.locator("#startBtn").is_visible()
    assert _page.locator("#battle-count").input_value() == "10"
    assert _page.locator(".topnav a[href='/battle.html']").count() == 1
    assert not errors, f"battle.html JS 异常: {errors}"


def test_battle_full_flow_and_statistics(_page):
    errors = []
    _page.on("pageerror", lambda e: errors.append(str(e)))
    _page.route("**/api/battle/questions", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps(FAKE_QUESTIONS)))
    stream_calls = {"n": 0}

    def on_stream(route):
        stream_calls["n"] += 1
        route.fulfill(status=200, content_type="text/event-stream", body=FAKE_SSE)

    _page.route("**/api/battle/stream", on_stream)

    _page.goto(f"{BASE}/battle.html")
    _page.wait_for_selector("#startBtn")
    _fill_models(_page)
    _page.click("#startBtn")

    # 对战区出现 + 第 1 题自动流式渲染
    _page.wait_for_selector("#battleArea", state="visible")
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.includes("答案A")')
    _page.wait_for_function(
        '() => document.getElementById("outB").textContent.includes("答案B")')
    assert _page.locator("#qTotal").text_content() == "2"
    assert "1+1" in _page.locator("#qText").text_content()
    assert stream_calls["n"] == 1

    # 投票 A更好 → 停在第 1 题（不自动跳转）
    _page.click('.vote-bar .vote-btn[data-v="0"]')
    _page.wait_for_timeout(200)
    assert "1+1" in _page.locator("#qText").text_content()
    assert _page.locator("#nextBtn").is_enabled()
    assert stream_calls["n"] == 1

    # 点「下一题」→ 进入第 2 题并自动流式作答
    _page.click("#nextBtn")
    _page.wait_for_function(
        '() => document.getElementById("qText").textContent.includes("长城")')
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.includes("答案A")')
    assert stream_calls["n"] == 2

    # 上一题回看：走缓存渲染，不重复请求
    _page.click("#prevBtn")
    _page.wait_for_timeout(200)
    assert "1+1" in _page.locator("#qText").text_content()
    assert "答案A1" in _page.locator("#outA").text_content()
    assert stream_calls["n"] == 2

    # 回第 2 题（缓存）→ 投平手 → 停在最后一题
    _page.click("#nextBtn")
    _page.wait_for_timeout(200)
    assert stream_calls["n"] == 2
    _page.click('.vote-bar .vote-btn[data-v="2"]')
    _page.wait_for_timeout(300)
    assert _page.locator("#nextBtn").is_disabled()

    # 结果统计弹窗：总体统计 + 逐题
    _page.click("#statBtn")
    _page.wait_for_selector("#statModal.open")
    body = _page.locator("#statBody").text_content()
    assert "总题数" in body and "已评分" in body and "2" in body
    assert "模型A获胜" in body and "平局" in body
    assert "A更好" in body and "平手" in body
    _page.click("#statClose")
    _page.wait_for_selector("#statModal.open", state="hidden")

    assert not errors, f"battle 全流程 JS 异常: {errors}"


def test_battle_blind_mode_and_reveal(_page):
    errors = []
    _page.on("pageerror", lambda e: errors.append(str(e)))
    # 固定 Math.random > 0.5 → 每题真实模型 A 在右栏（b-left）
    _page.add_init_script("Math.random = () => 0.9;")
    _page.route("**/api/battle/questions", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps(FAKE_QUESTIONS)))
    _page.route("**/api/battle/stream", lambda route: route.fulfill(
        status=200, content_type="text/event-stream", body=FAKE_SSE))
    _page.goto(f"{BASE}/battle.html")
    _page.wait_for_selector("#startBtn")
    _fill_models(_page)
    _page.check("#anon-mode")
    _page.click("#startBtn")
    _page.wait_for_selector("#battleArea", state="visible")
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.length > 0')

    # 盲评：代号显示 + 真实名不出现；b-left → 真实 A 输出在右栏
    assert _page.locator("#nameA").text_content() == "模型 A"
    assert _page.locator("#nameB").text_content() == "模型 B"
    assert _page.locator("#anonA").text_content() == "（匿名）"
    _page.wait_for_function(
        '() => document.getElementById("outB").textContent.includes("答案A1")')
    assert "答案B1" in _page.locator("#outA").text_content()

    # 第 1 题投票 → 不跳转
    _page.click('.vote-bar .vote-btn[data-v="0"]')
    _page.wait_for_timeout(200)
    assert "1+1" in _page.locator("#qText").text_content()

    # 未评完：揭晓按钮禁用
    _page.click("#statBtn")
    _page.wait_for_selector("#statModal.open")
    _page.wait_for_selector("#revealBox", state="visible")
    assert _page.locator("#revealBtn").is_disabled()
    _page.click("#statClose")
    _page.wait_for_selector("#statModal.open", state="hidden")

    # 下一题 → 第 2 题作答（同样 b-left）
    _page.click("#nextBtn")
    _page.wait_for_function(
        '() => document.getElementById("qText").textContent.includes("长城")')
    _page.wait_for_function(
        '() => document.getElementById("outB").textContent.includes("答案A1")')

    # 第 2 题投票 → 全部评完 → 揭晓
    _page.click('.vote-bar .vote-btn[data-v="2"]')
    _page.wait_for_timeout(300)
    _page.click("#statBtn")
    _page.wait_for_selector("#statModal.open")
    _page.wait_for_selector("#revealBox", state="visible")
    assert not _page.locator("#revealBtn").is_disabled()
    _page.click("#revealBtn")
    _page.wait_for_function(
        '() => document.getElementById("revealInfo").textContent.includes("模型甲")')
    assert "模型乙" in _page.locator("#revealInfo").text_content()
    # 侧栏更新为真实名
    assert _page.locator("#nameA").text_content() == "模型甲"
    assert _page.locator("#nameB").text_content() == "模型乙"

    assert not errors, f"盲评 JS 异常: {errors}"


def test_battle_session_persists_across_reload(_page):
    _page.route("**/api/battle/questions", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps(FAKE_QUESTIONS)))
    stream_calls = {"n": 0}

    def on_stream(route):
        stream_calls["n"] += 1
        route.fulfill(status=200, content_type="text/event-stream", body=FAKE_SSE)

    _page.route("**/api/battle/stream", on_stream)

    _page.goto(f"{BASE}/battle.html")
    _page.wait_for_selector("#startBtn")
    _fill_models(_page)
    _page.click("#startBtn")
    _page.wait_for_selector("#battleArea", state="visible")
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.includes("答案A1")')
    assert stream_calls["n"] == 1
    _page.click('.vote-bar .vote-btn[data-v="0"]')
    _page.wait_for_timeout(200)
    assert "已评分 1 / 2" in _page.locator("#vProgress").text_content()

    # 跳转/刷新 → 对局恢复
    _page.reload()
    _page.wait_for_selector("#battleArea", state="visible")
    _page.wait_for_timeout(500)
    assert "1+1" in _page.locator("#qText").text_content()
    assert "已评分 1 / 2" in _page.locator("#vProgress").text_content()
    assert "答案A1" in _page.locator("#outA").text_content()
    assert _page.locator("#mA-name").input_value() == "模型甲"
    assert stream_calls["n"] == 1  # 缓存命中，不重复请求
    assert _page.evaluate("() => sessionStorage.getItem('battle_session') !== null")

    # 恢复后仍可继续：下一题自动作答
    _page.click("#nextBtn")
    _page.wait_for_function(
        '() => document.getElementById("qText").textContent.includes("长城")')
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.includes("答案A1")')
    assert stream_calls["n"] == 2

    # 重新对战 → 回到第 1 题，会话写入新对局（votes 清零）
    _page.click("#restartBtn")
    _page.wait_for_function(
        '() => document.getElementById("qText").textContent.includes("1+1")')
    _page.wait_for_timeout(300)
    assert "已评分 0 / 2" in _page.locator("#vProgress").text_content()
    assert _page.evaluate(
        "() => Object.keys(JSON.parse(sessionStorage.getItem('battle_session')).votes).length === 0")


def test_battle_session_interrupted_stream_restarts(_page):
    _page.route("**/api/battle/questions", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps(FAKE_QUESTIONS)))
    stream_calls = {"n": 0}

    def on_stream(route):
        stream_calls["n"] += 1
        if stream_calls["n"] == 1:
            return  # 首次请求挂起（不 fulfill），模拟流式未完成时跳转打断
        route.fulfill(status=200, content_type="text/event-stream", body=FAKE_SSE)

    _page.route("**/api/battle/stream", on_stream)

    _page.goto(f"{BASE}/battle.html")
    _page.wait_for_selector("#startBtn")
    _fill_models(_page)
    _page.click("#startBtn")
    _page.wait_for_selector("#battleArea", state="visible")
    _page.wait_for_timeout(300)
    # 首题流式挂起中打断（fetch 被 abort，不写缓存）
    _page.reload()
    _page.wait_for_selector("#battleArea", state="visible")
    # 恢复后该题无缓存 → 自动重新流式并完成
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.includes("答案A1")')
    assert stream_calls["n"] >= 2  # 打断后自动重新作答


def test_battle_exit_resets_to_initial(_page):
    _page.route("**/api/battle/questions", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps(FAKE_QUESTIONS)))
    _page.route("**/api/battle/stream", lambda route: route.fulfill(
        status=200, content_type="text/event-stream", body=FAKE_SSE))

    _page.goto(f"{BASE}/battle.html")
    _page.wait_for_selector("#startBtn")
    _fill_models(_page)
    _page.click("#startBtn")
    _page.wait_for_selector("#battleArea", state="visible")
    _page.wait_for_function(
        '() => document.getElementById("outA").textContent.includes("答案A1")')
    assert _page.locator("#exitBtn").is_visible()
    _page.click('.vote-bar .vote-btn[data-v="0"]')
    _page.wait_for_timeout(200)
    assert "已评分 1 / 2" in _page.locator("#vProgress").text_content()

    # 退出对决：确认弹窗 → 回到初始设置
    _page.on("dialog", lambda d: d.accept())
    _page.click("#exitBtn")
    _page.wait_for_selector("#battleArea", state="hidden")
    assert _page.locator("#startBtn").is_visible()
    assert _page.locator("#restartBtn").is_hidden()
    assert _page.locator("#exitBtn").is_hidden()
    assert _page.evaluate("() => sessionStorage.getItem('battle_session') === null")

    # 刷新后保持初始状态（对局不恢复）
    _page.reload()
    _page.wait_for_selector("#startBtn")
    assert _page.locator("#battleArea").is_hidden()


def test_tasks_wizard_creates_task(_page):
    submitted = []

    def on_benchmark(route):
        if route.request.method == "POST" and route.request.post_data:
            submitted.append(json.loads(route.request.post_data))
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"ok": True, "batch_id": "batch_e2e_1",
                                       "jobs": ["j1", "j2"], "models": ["E2E甲", "E2E乙"],
                                       "state": "running"}))

    _page.route("**/api/benchmark", on_benchmark)

    _page.goto(f"{BASE}/tasks.html")
    _page.wait_for_selector("#newTaskBtn")
    assert _page.locator(".topnav a[href='/battle.html']").count() == 1

    # 打开向导 → 第 1 步填 2 个模型
    _page.click("#newTaskBtn")
    _page.wait_for_selector("#wizardMask.open")
    _page.wait_for_selector(".wizard-step.active #wz-models")
    _page.fill('#wz-models .model-row[data-idx="0"] input[data-f="url"]', "https://8.8.8.8/v1")
    _page.fill('#wz-models .model-row[data-idx="0"] input[data-f="key"]', "k")
    _page.fill('#wz-models .model-row[data-idx="0"] input[data-f="name"]', "E2E甲")
    _page.fill('#wz-models .model-row[data-idx="1"] input[data-f="url"]', "https://8.8.8.8/v1")
    _page.fill('#wz-models .model-row[data-idx="1"] input[data-f="key"]', "k")
    _page.fill('#wz-models .model-row[data-idx="1"] input[data-f="name"]', "E2E乙")
    _page.click("#wz-next")

    # 第 2 步：AI 评审开关（不填评审 → 下一步）+ 扰动评测开启
    _page.wait_for_selector('#ws-1.active')
    _page.check("#wz-perturb-on")
    _page.wait_for_selector("#wz-perturb-modes", state="visible")
    _page.click("#wz-next")
    # 第 3 步：内置题库（默认）
    _page.wait_for_selector('#ws-2.active')
    assert _page.locator("#wz-dims input").count() == 8
    _page.click("#wz-next")
    # 第 4 步：名称 + 提交
    _page.wait_for_selector('#ws-3.active')
    _page.fill("#wz-name", "E2E 任务")
    _page.click("#wz-submit")
    _page.wait_for_timeout(500)

    assert len(submitted) == 1
    body = submitted[0]
    assert len(body["models"]) == 2
    assert body["models"][0]["name"] == "E2E甲"
    assert body["models"][0]["max_tokens"] == 16384
    assert body["name"] == "E2E 任务"
    assert body["rounds"] == 1
    assert body["code_verify_mode"] == "off"
    # 扰动评测：默认勾选 改写+噪声注入
    assert body.get("perturb_modes") == ["改写", "噪声注入"]
