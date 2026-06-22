"""Rental port reachability helpers.

Validators use this from an external vantage point before returning SSH
details to a renter. A miner-local or monitor-local check can prove the host
port is bound, but it cannot prove provider-firewall reachability.
"""
from __future__ import annotations

import socket


def probe_tcp_port(host: str, port: int, timeout: float = 2.5) -> bool:
    """Return True if a TCP connection to ``host:port`` succeeds quickly."""
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        return False
    if not host or port_int < 1 or port_int > 65535:
        return False
    try:
        with socket.create_connection((host, port_int), timeout=timeout):
            return True
    except Exception:
        return False
