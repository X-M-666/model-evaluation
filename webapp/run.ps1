# -*- coding: utf-8 -*-
"""模型对决评测平台 - 一键启动脚本（PowerShell）"""
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

# 默认监听所有网卡（局域网可访问）；如需仅本机改为 $HOST = "127.0.0.1"
$HOST = "0.0.0.0"
$PORT = if ($env:PORT) { $env:PORT } else { "8910" }

Write-Host "`n🚀 启动 http://$HOST`:$PORT" -ForegroundColor Green
Write-Host "   按 Ctrl+C 停止`n" -ForegroundColor DarkGray

# 启动 uvicorn
python -m uvicorn backend.main:app --host $HOST --port $PORT --reload

Pop-Location
