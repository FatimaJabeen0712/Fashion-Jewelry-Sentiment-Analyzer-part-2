import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Tests must never touch the network (no live API calls)."""
    real_connect = socket.socket.connect

    def guard(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost") or not isinstance(address, tuple):
            return real_connect(self, address, *args, **kwargs)   # local only (e.g. Spark's JVM bridge)
        raise RuntimeError("Network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
