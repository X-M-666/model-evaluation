# -*- coding: utf-8 -*-
"""SSRF 防护：校验评测上游 URL（模型 API 入口），阻止服务端向非预期目标发起请求。

默认只允许公网 http/https 目标；解析出的任一 IP 非公网（loopback/私网/链路本地/
组播/CGNAT/云元数据/保留段等）即拒绝。企业内网自建模型场景可显式放行：
    MODEL_DUEL_ALLOW_PRIVATE_UPSTREAM=1
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

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

    host = parsed.hostname
    if not ALLOW_PRIVATE:
        # 主机名本身是 IP 时直接校验（避免依赖解析）
        ip_ok = False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            ip_ok = True
        if ip_ok:
            if not _is_public_ip(host):
                raise UpstreamUrlError(f"目标地址非公网: {host}")
        else:
            for ip in _resolve_host(host):
                if not _is_public_ip(ip):
                    raise UpstreamUrlError(f"目标 {host} 解析到非公网地址: {ip}")

    return url.strip()
