# 复审：SSE 长期 Token 不进 URL 修复的残余问题与核对（issue #13）

## 背景

issue #13（R2-004，严重度：高）要求共享模式下长期管理员 Token 不得经 SSE URL 传递。
修复采用「短时单次 ticket」方案（建议方向 2）：前端先经已认证的
`POST /api/eval/{job_id}/events/ticket` 换取随机 ticket，再以原生 EventSource
携带 `?ticket=...` 建立连接。本反馈为对该修复实现的第三轮复审核对。

## 现状与核对

修复实现：

- `backend/sse_ticket.py`：短时（默认 60s，`MODEL_DUEL_SSE_TICKET_TTL` 可调）、
  单次（`threading.Lock` 原子 consume 即用即焚）、作用域限定（与签发 job 绑定）、
  容量上限 10000 条并惰性清理过期条目。
- `backend/access.py:119-133`：共享模式下 `/events` 路由接受 Bearer header 或
  单次 ticket；**长期 Token 经 URL（`?token=`）一律 401**；认证通过后从
  `scope["query_string"]` 剥离 ticket 参数，避免进入 uvicorn 访问日志。
- `frontend/index.html:456-509`：ticket 签发 → EventSource 连接 → 断线重连重新签发。
- 测试：`tests/test_sse_ticket.py`（单次/过期/作用域不匹配/并发单用/容量淘汰）+
  `tests/test_access_control.py`（URL 带 Token 拒绝、伪造/过期/重放/跨 job ticket 401、
  query_string 剥离单测）。

验收建议逐条核对：

| 验收建议 | 结果 |
|---|---|
| 长期 Token 不出现在 URL/Referer/访问日志 | ✅ 前端只传 ticket；`?token=` 401；审计日志白名单不含查询串 |
| SSE 临时凭据短时、单次、仅限指定资源 | ✅ 60s TTL / 原子单次 / job 作用域 + `/events` 路径限定 |
| 未认证、过期、重复使用、作用域不匹配均拒绝 | ✅ 均有集成测试覆盖 |
| 反向代理部署场景有自动化/集成测试验证代理日志不记长期 Token | ⚠️ 部分满足（见残余 3） |
| 日志、异常与监控事件统一对认证材料脱敏 | ⚠️ 部分满足（见残余 1） |
| 不因改用 Cookie 引入 CSRF | ✅ 未采用 Cookie 方案，无 CSRF 面 |

## 残余风险

1. **认证失败的 401 请求其 ticket 参数不剥离**（低危）
   `access.py:126-133`：剥离逻辑位于 `if not ok: return 401` **之后**，仅对认证
   通过且带 ticket 的请求生效。伪造/过期/重放 ticket 的 401 请求，其 URL 查询串
   原样进入 uvicorn 访问日志与前置代理日志。ticket 已失效故风险低，但「日志统一
   对认证材料脱敏」的验收建议未完全达成。
   位置：`webapp/backend/access.py:126-133`

2. **EventSource 自动重连会携带已用 ticket 触发一次 401**（低危）
   断线后浏览器按规范自动重连同一 URL（ticket 已被首次连接消费）→ 401 →
   `onerror` 触发后前端才重新签发并新建 EventSource。产生一次无效请求与 401
   日志噪声；若 TTL 极短（如自定义 <1s）可能放大为连续重试。
   位置：`webapp/frontend/index.html:482,502-509`

3. **反向代理日志验证无自动化测试**（低危）
   测试全部经 TestClient 直连应用，未覆盖「代理先看到 URL」的真实链路；代理
   /WAF 日志中的 ticket（短时单次，非长期凭据）暴露面仅靠文档声明。
   位置：`README.md:55`

4. **ticket 表内存残留**（低危）
   `consume` 不删除已用条目，仅由下次 `issue` 惰性清理或容量淘汰回收；已用
   ticket 在 TTL 内驻留内存。有 10000 上限兜底，无 DoS 面（签发端点受认证与
   写限流保护）。
   位置：`webapp/backend/sse_ticket.py:46-63`

## 建议方向

1. 将 `/events` 的 ticket 参数剥离无条件前置：无论认证成败，进入路由前统一从
   `scope["query_string"]` 移除 `ticket`（`access.py` 中把剥离逻辑移到 401 分支之前）。
2. 前端改用 `fetch` + `ReadableStream` 流式读取替代原生 EventSource（issue #13
   建议方向 3），彻底消除自动重连携带已用 ticket 的竞态与 401 噪声；或保持
   EventSource 时在 `onerror` 中忽略带 ticket 的自动重连（EventSource 无法完全
   禁用自动重连，需配合服务端对重复消费返回 204/429 静默）。
3. 补充代理链路验证：在 CI 中增加可选集成步骤（本地 nginx 反代 + 断言访问日志
   不含长期 Token；ticket 允许出现），或至少在部署文档中明确残余面。
4. `consume` 成功后即删除条目（`_tickets.pop`），与过期清理共用路径，减少残留。

## 验收建议

- 401 与 200 的 `/events` 请求在 uvicorn 访问日志中均不出现 `ticket` 查询参数。
- 断线重连流程不产生「携带已用 ticket 的自动重连请求」；或该请求被服务端静默处理。
- 代理访问日志的自动化验证有明确交付（CI 步骤或部署文档声明）。
- `_tickets` 中已用条目在消费后立即移除，内存残留仅剩未消费的过期条目。

该问题来自 2026-08-06 第三轮代码复审，记录编号 R3-001，严重度为低
（核心问题已修复：长期 Token 已确认不进入任何 URL；残余项均为加固与验证缺口）。

## 处置记录（2026-08-06 实施）

四项残余已全部落地修复：

| 残余 | 修复 |
|---|---|
| 1. 401 请求 ticket 不剥离 | `access.py`：ticket/token 查询参数剥离无条件前置到认证分支之前（含 401 路径），并扩展 `token` 参数一并剥离（误把长期 Token 放 URL 的 401 请求日志同样干净） |
| 2. EventSource 自动重连 401 噪声 | 携带 ticket 的 `/events` 认证失败静默返回 `204` 且不记审计（128 位随机单次凭证无爆破价值）；未携带 ticket 的未认证请求仍 401 + 审计。README 已同步 |
| 3. 代理日志验证无自动化测试 | 新增 `tests/test_proxy_logs.py`：同进程后台线程运行真实 uvicorn（共享模式），捕获 `uvicorn.access` 日志，断言 401/204/200 请求日志均不含 `ticket=` 与长期 Token 值；全平台 CI 运行 |
| 4. consume 后条目残留 | `sse_ticket.py`：`consume` 成功即 `_tickets.pop`（消费即焚），移除 `used` 字段；作用域不匹配仍不消耗 ticket（正确 job 可复用） |

验证：`tests/test_sse_ticket.py`（消费即删断言）、`test_access_control.py`（ticket 失败 204 + 审计静默断言）、`test_proxy_logs.py`（真实日志链路）全部通过；完整测试套件 426 passed。
