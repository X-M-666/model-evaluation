# -*- coding: utf-8 -*-
"""SSRF 防护：校验评测上游 URL（模型 API 入口），阻止服务端向非预期目标发起请求。

默认只允许公网 http/https 目标；解析出的任一 IP 非公网（loopback/私网/链路本地/
组播/CGNAT/云元数据/保留段等）即拒绝。企业内网自建模型场景可显式放行：
    MODEL_DUEL_ALLOW_PRIVATE_UPSTREAM=1
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx
from httpcore import AsyncNetworkBackend, AsyncNetworkStream

ALLOW_PRIVATE = os.environ.get("MODEL_DUEL_ALLOW_PRIVATE_UPSTREAM", "") == "1"


class UpstreamUrlError(ValueError):
    """上游 URL 非法或指向非公网目标。"""


def _is_public_ip(ip: str) -> bool:
    """判定 IP 是否为公网地址。

    Python 3.12 的 ipaddress.is_global 已覆盖私网/链路本地/保留/CGNAT/云元数据，
    但对 D 类组播（224.0.0.0/4）误判为公网，需显式排除。
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return bool(addr.is_global) and not addr.is_multicast


def _resolve_host(host: str) -> list[str]:
    """解析主机名为全部 IP（失败抛 UpstreamUrlError）。"""
    try:
        infos = socket.getaddrinfo(host, None, 0, 0, socket.SOL_TCP)
    except OSError as exc:
        raise UpstreamUrlError(f"无法解析主机 {host}: {exc}") from exc
    ips = []
    for info in infos:
        ip = info[4][0]
        if ip not in ips:
            ips.append(ip)
    if not ips:
        raise UpstreamUrlError(f"主机 {host} 无解析结果")
    return ips


def resolve_validated(host: str) -> list[str]:
    """解析主机名并校验全部结果 IP 均为公网（DNS 重绑定防护的解析端）。

    - host 为 IP 字面量：直接校验公网性，不触发解析；
    - host 为域名：解析出全部 IP，任一非公网即整体拒绝（防止攻击者把某个
      A/AAAA 记录指向内网后再切换回公网实现重绑定）；
    - ALLOW_PRIVATE=1 时跳过公网性校验，仅返回解析结果。
    每次调用都重新解析（不缓存），保证连接前一刻的真实解析结果。
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        ips = _resolve_host(host)
        if not ALLOW_PRIVATE:
            for ip in ips:
                if not _is_public_ip(ip):
                    raise UpstreamUrlError(f"目标 {host} 解析到非公网地址: {ip}")
        return ips
    else:
        if not ALLOW_PRIVATE and not _is_public_ip(host):
            raise UpstreamUrlError(f"目标地址非公网: {host}")
        return [host]


def validate_upstream_url(url: str) -> str:
    """校验模型 API 上游 URL，返回规范化后的 URL。

    拒绝：非 http/https 协议、含 userinfo（@）、解析后任一 IP 非公网。
    ALLOW_PRIVATE=1 时跳过 IP 公网性校验（协议与结构仍校验）。
    """
    if not isinstance(url, str) or not url.strip():
        raise UpstreamUrlError("URL 不能为空")

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise UpstreamUrlError(f"仅支持 http/https 协议: {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise UpstreamUrlError("URL 不允许包含用户名/密码信息")
    if not parsed.hostname:
        raise UpstreamUrlError("URL 缺少主机名")

    if not ALLOW_PRIVATE:
        # 域名或 IP 均走统一解析校验路径；解析失败同样拒绝
        resolve_validated(parsed.hostname)

    return url.strip()


def build_upstream_client(**kwargs) -> httpx.AsyncClient:
    """构造带 SSRF 连接校验的 httpx 客户端（连通性测试与正式评测共用）。

    连接前即时解析 + 逐 IP 校验，拒绝指向内网/IP 字面量的连接
    （DNS 重绑定防护的连接端）。TLS SNI/证书校验由 httpcore 在连接层
    以原始域名进行（见 ValidatingNetworkBackend 说明）。
    """
    kwargs.setdefault("follow_redirects", False)
    kwargs.setdefault("transport", ValidatingTransport())
    return httpx.AsyncClient(**kwargs)


class ValidatingNetworkBackend(AsyncNetworkBackend):
    """在 httpcore 建 TCP 连接前重新解析并校验上游目标（官方扩展点）。

    httpx 0.27+ 的 AsyncHTTPTransport 内部是 httpcore 连接池，每次新建
    连接（含重试）都会经过 network_backend.connect_tcp。此处把
    "域名 -> 连接 IP" 绑定为连接前一刻的解析结果，将校验过的 IP 交给
    内部后端连接；TLS 由 httpcore 连接层后续以原始域名做 SNI/证书校验，
    因此域名伪造与证书绕过均不可行。
    """

    def __init__(self, inner: AsyncNetworkBackend):
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> AsyncNetworkStream:
        if isinstance(host, bytes):
            host = host.decode("ascii")
        try:
            ips = await asyncio.to_thread(resolve_validated, host)
        except UpstreamUrlError as exc:
            raise httpx.ConnectError(f"SSRF 校验失败: {exc}") from exc

        last_error: OSError | None = None
        for ip in ips:
            try:
                return await self._inner.connect_tcp(
                    ip, port, timeout=timeout,
                    local_address=local_address, socket_options=socket_options,
                )
            except OSError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise httpx.ConnectError(f"无法连接到 {host}")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class ValidatingTransport(httpx.AsyncHTTPTransport):
    """注入 SSRF 校验网络后端的 httpx 传输层。

    httpx 0.27+ 移除了 _connect_tcp 钩子（改为内部 httpcore 连接池），
    故在构造后把连接池的 network_backend 替换为校验后端。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pool._network_backend = ValidatingNetworkBackend(self._pool._network_backend)
