"""Baostock 会话生命周期：登录校验、超时重建、失败可识别。"""

import json
import socket
import zlib

import pytest

from sequoia_x.data.baostock_session import (
    BaostockRequestError,
    BaostockSession,
    ConnectionFailure,
    LoginFailure,
    RequestTimeout,
)


class FakeSocket:
    def __init__(self, identity: int, responses: list[bytes | BaseException]) -> None:
        self.identity = identity
        self.responses = list(responses)
        self.timeout: float | None = None
        self.sent: list[bytes] = []
        self.closed = False

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def connect(self, address: tuple[str, int]) -> None:
        return None

    def send(self, data: bytes) -> int:
        self.sent.append(data)
        return len(data)

    def recv(self, size: int) -> bytes:
        if not self.responses:
            raise TimeoutError("timed out")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


_CLIENT_VERSION = "00.9.10"


def _frame(msg_type: str, body: str) -> bytes:
    """按 baostock 20 字节头组一帧非压缩应答，结尾换行。"""
    header = f"{_CLIENT_VERSION}\x01{msg_type}\x01{len(body):010d}"
    return f"{header}{body}\n".encode()


def login_frame(error_code: str = "0", error_msg: str = "success") -> bytes:
    return _frame("01", f"{error_code}\x01{error_msg}\x01login\x01anonymous")


def query_frame(rows: list[list[str]]) -> bytes:
    # 非压缩应答：error_code, error_msg, method, user_id, cur_page, per_page, record JSON
    payload = json.dumps({"record": rows}, separators=(",", ":"))
    body = f"0\x01success\x01query_history_k_data_plus\x01anonymous\x011\x0110000\x01{payload}"
    return _frame("12", body)


def compressed_k_frame(rows: list[list[str]]) -> list[bytes]:
    """K 线应答类型 96 是压缩帧，gzip 体内可能含换行，结尾才是 CDATA。"""
    payload = json.dumps({"record": rows}, separators=(",", ":"))
    body = f"0\x01success\x01query_history_k_data_plus\x01anonymous\x011\x0110000\x01{payload}"
    compressed = zlib.compress(body.encode())
    assert b"\n" in compressed
    header = f"{_CLIENT_VERSION}\x0196\x01{len(compressed):010d}".encode()
    return [header + compressed, b"<![CDATA[]]>\n"]


@pytest.fixture
def scripted_sockets(monkeypatch: pytest.MonkeyPatch):
    """按创建顺序把预设应答装到新 socket 上。"""
    created: list[FakeSocket] = []
    queued: list[list[bytes | BaseException]] = []

    def factory(*_args, **_kwargs) -> FakeSocket:
        responses = queued.pop(0) if queued else []
        sock = FakeSocket(len(created) + 1, responses)
        created.append(sock)
        return sock

    monkeypatch.setattr(socket, "socket", factory)

    class Script:
        sockets = created

        def enqueue(self, *responses: bytes | BaseException) -> None:
            queued.append(list(responses))

    return Script()


def test_login_failure_does_not_query(scripted_sockets) -> None:
    """登录失败时不会发起行情查询。"""
    session = BaostockSession(timeout=30, max_retries=0)
    scripted_sockets.enqueue(login_frame("10001001", "用户名或密码错误"))

    with pytest.raises(LoginFailure) as exc:
        session.query_history_k_data_plus("sh.600000", "date,close", "2024-01-01", "2024-01-02")

    sock = scripted_sockets.sockets[0]
    assert exc.value.error_code == "10001001"
    assert sock.closed
    assert all(b"query_history_k_data_plus" not in payload for payload in sock.sent)
    assert len(scripted_sockets.sockets) == 1


def test_timeout_retry_uses_a_new_connection(scripted_sockets) -> None:
    """连接超时后，后续重试使用新连接。"""
    session = BaostockSession(timeout=45, max_retries=1)
    scripted_sockets.enqueue(login_frame(), TimeoutError("timed out"))
    scripted_sockets.enqueue(login_frame(), query_frame([["2024-01-02", "10.5"]]))

    rows = session.query_history_k_data_plus(
        "sh.600000", "date,close", "2024-01-01", "2024-01-02"
    )

    first, second = scripted_sockets.sockets
    assert first.closed
    assert first is not second
    assert second.timeout == 45
    assert rows == [["2024-01-02", "10.5"]]
    assert any(b"query_history_k_data_plus" in payload for payload in second.sent)


def test_compressed_kline_with_embedded_newline_is_recovered(scripted_sockets) -> None:
    """压缩 K 线体内的换行不能被当成帧结束，否则恢复后的查询解不出记录。"""
    session = BaostockSession(timeout=30, max_retries=0)
    scripted_sockets.enqueue(login_frame(), *compressed_k_frame([["2024-01-02", "10.5"]]))

    rows = session.query_history_k_data_plus(
        "sh.600000", "date,close", "2024-01-01", "2024-01-02"
    )

    assert rows == [["2024-01-02", "10.5"]]


def test_timeout_and_connection_failure_are_identifiable(scripted_sockets) -> None:
    """超时和连接失败能够被记录为可识别的失败结果，而不是静默跳过。"""
    session = BaostockSession(timeout=30, max_retries=0)
    scripted_sockets.enqueue(login_frame(), TimeoutError("timed out"))

    with pytest.raises(RequestTimeout) as timeout_exc:
        session.query_history_k_data_plus("sh.600000", "date,close", "2024-01-01", "2024-01-02")
    assert timeout_exc.value.kind == "timeout"
    assert timeout_exc.value.code == "sh.600000"

    scripted_sockets.enqueue(ConnectionResetError("connection reset"))
    with pytest.raises(ConnectionFailure) as conn_exc:
        session.query_history_k_data_plus("sz.000001", "date,close", "2024-01-01", "2024-01-02")
    assert conn_exc.value.kind == "connection"
    assert conn_exc.value.code == "sz.000001"
    assert isinstance(conn_exc.value, BaostockRequestError)
    assert not isinstance(conn_exc.value, RequestTimeout)
