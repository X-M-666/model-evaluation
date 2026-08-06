# -*- coding: utf-8 -*-
# 模型对决评测平台 - 一键启动脚本（PowerShell）
param(
    [string]$ListenAddress = "127.0.0.1"
)
$ErrorActionPreference = "Stop"
$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $DIR

Write-Host "`n⚔️  模型对决评测平台 v0.1" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor DarkGray

# 检查 Python
try { $py = python --version 2>&1 } catch {
    Write-Host "❌ 未找到 Python，请先安装 Python 3.10+" -ForegroundColor Red
    exit 1
}

# 检查依赖
Write-Host "检查依赖..." -ForegroundColor DarkGray
python -c "import fastapi, uvicorn, httpx, pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "安装依赖..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# 默认仅本机可访问（127.0.0.1）；如需局域网访问：.\run.ps1 -ListenAddress 0.0.0.0
$PORT = if ($env:PORT) { $env:PORT } else { "8910" }

# 安全提示（issue #8）：非回环监听时提醒设置访问令牌
$TOKEN = $env:MODEL_DUEL_TOKEN
if ($ListenAddress -ne "127.0.0.1" -and $ListenAddress -ne "localhost") {
    Write-Host "`n⚠️  正在监听 $ListenAddress，其他设备可访问本平台！" -ForegroundColor Yellow
    if (-not $TOKEN) {
        Write-Host "⚠️  未设置 MODEL_DUEL_TOKEN：局域网内任何人都可发起评测（无鉴权）。" -ForegroundColor Yellow
        Write-Host "    强烈建议设置令牌：`$env:MODEL_DUEL_TOKEN = '<强随机串>'" -ForegroundColor Yellow
    } else {
        Write-Host "✅ 已检测到 MODEL_DUEL_TOKEN，共享模式鉴权已启用。" -ForegroundColor Green
    }
} elseif ($TOKEN) {
    Write-Host "🔒 共享模式：已启用访问令牌（MODEL_DUEL_TOKEN）鉴权。" -ForegroundColor Cyan
}

Write-Host "`n🚀 启动 http://$ListenAddress`:$PORT" -ForegroundColor Green
Write-Host "   按 Ctrl+C 停止`n" -ForegroundColor DarkGray

# 启动 uvicorn
python -m uvicorn backend.main:app --host $ListenAddress --port $PORT --reload

Pop-Location
