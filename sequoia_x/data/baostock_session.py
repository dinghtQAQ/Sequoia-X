"""可恢复的 Baostock 会话：超时、登录校验、失败后重建连接。"""

from __future__ import annotations

import json
import socket
import zlib
from dataclasses import dataclass

import baostock.common.contants as cons
import baostock.common.context as context
import baostock.data.messageheader as msgheader

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 45.0
_MIN_TIMEOUT = 30.0
_MAX_TIMEOUT = 60.0
_RECV_SIZE = 8192
_END_MARK = b"<![CDATA[]]>\n"


class BaostockRequestError(Exception):
    """可识别的行情请求失败，调用方不得当成空结果跳过。"""

    kind = "request"

    def __init__(self, message: str, *, code: str = "", error_code: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code


class LoginFailure(BaostockRequestError):
    """登录未成功。此时不得继续发起行情查询。"""

    kind = "login"


class RequestTimeout(BaostockRequestError):
    """连接或读写超过设定超时。"""

    kind = "timeout"


class ConnectionFailure(BaostockRequestError):
    """连接建立、发送或接收失败。"""

    kind = "connection"


@dataclass
class _LoginResult:
    error_code: str
    error_msg: str

    @property
    def ok(self) -> bool:
        return self.error_code == cons.BSERR_SUCCESS


class BaostockSession:
    """一次可恢复的 Baostock TCP 会话。

    超时落在 30～60 秒。登录结果必须成功才允许查询。
    超时或连接异常会关闭当前连接；``call`` 在限定次数内重建后再试。
    """

    def __init__(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
        user_id: str = "anonymous",
        password: str = "123456",
    ) -> None:
        if not _MIN_TIMEOUT <= timeout <= _MAX_TIMEOUT:
            raise ValueError(f"timeout 必须在 {_MIN_TIMEOUT:g}～{_MAX_TIMEOUT:g} 秒之间，收到 {timeout}")
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_id = user_id
        self.password = password
        self._sock: socket.socket | None = None
        self._logged_in = False

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
    ) -> list[list[str]]:
        """拉取日 K，返回行列表。失败抛出可识别异常，不返回空列表冒充成功。"""
        body = cons.MESSAGE_SPLIT.join(
            [
                "query_history_k_data_plus",
                self.user_id,
                "1",
                str(cons.BAOSTOCK_PER_PAGE_COUNT),
                code,
                fields,
                start_date,
                end_date,
                frequency,
                adjustflag,
            ]
        )
        raw = self.call(cons.MESSAGE_TYPE_GETKDATAPLUS_REQUEST, body, code=code)
        return _checked_records(raw, code, "行情查询失败")

    def query_stock_basic(self, code: str = "", code_name: str = "") -> list[list[str]]:
        """查询证券基本资料。登录失败或连接失败时抛出，不返回空列表冒充成功。"""
        body = cons.MESSAGE_SPLIT.join(
            [
                "query_stock_basic",
                self.user_id,
                "1",
                str(cons.BAOSTOCK_PER_PAGE_COUNT),
                code,
                code_name,
            ]
        )
        raw = self.call(cons.MESSAGE_TYPE_QUERYSTOCKBASIC_REQUEST, body, code=code or code_name)
        return _checked_records(raw, code or code_name or "stock_basic", "证券资料查询失败")

    def call(self, msg_type: str, body: str, *, code: str = "") -> str:
        """发送一帧并读完应答。连接类失败按 ``max_retries`` 次重建后重试。"""
        attempts = self.max_retries + 1
        last_error: BaostockRequestError | None = None
        for attempt in range(attempts):
            try:
                self._ensure_logged_in(code)
                return self._roundtrip(msg_type, body, code)
            except LoginFailure:
                raise
            except (RequestTimeout, ConnectionFailure) as exc:
                last_error = exc
                self.close()
                logger.warning(
                    f"[{code or '-'}] {exc.kind} 失败（第 {attempt + 1}/{attempts} 次）: {exc}"
                )
                if attempt >= self.max_retries:
                    raise
        raise last_error or ConnectionFailure("请求失败", code=code)

    def close(self) -> None:
        """关闭当前连接。关闭失败也要丢掉句柄，避免下次重试复用死连接。"""
        sock = self._sock
        self._sock = None
        self._logged_in = False
        if sock is None:
            return
        try:
            sock.close()
        except OSError as exc:
            logger.warning(f"关闭 baostock 连接失败: {exc}")
        if getattr(context, "default_socket", None) is sock:
            context.default_socket = None

    def _ensure_logged_in(self, code: str) -> None:
        if self._logged_in and self._sock is not None:
            return
        self.close()
        self._connect()
        result = self._login(code)
        if not result.ok:
            self.close()
            message = f"baostock 登录失败: {result.error_code} {result.error_msg}"
            logger.error(message)
            raise LoginFailure(message, code=code, error_code=result.error_code)
        self._logged_in = True

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((cons.BAOSTOCK_SERVER_IP, cons.BAOSTOCK_SERVER_PORT))
        except TimeoutError as exc:
            sock.close()
            raise RequestTimeout(f"连接 baostock 超时: {exc}", code="") from exc
        except OSError as exc:
            sock.close()
            raise ConnectionFailure(f"连接 baostock 失败: {exc}", code="") from exc
        self._sock = sock
        context.default_socket = sock

    def _login(self, code: str) -> _LoginResult:
        options = getattr(context, "apiKey", "0") or "0"
        body = cons.MESSAGE_SPLIT.join(["login", self.user_id, self.password, str(options)])
        try:
            raw = self._roundtrip(cons.MESSAGE_TYPE_LOGIN_REQUEST, body, code)
        except RequestTimeout as exc:
            raise RequestTimeout(str(exc), code=exc.code) from exc
        except ConnectionFailure as exc:
            raise ConnectionFailure(str(exc), code=exc.code) from exc
        parts = raw.split(cons.MESSAGE_SPLIT)
        error_code = parts[0] if parts else cons.BSERR_RECVSOCK_FAIL
        error_msg = parts[1] if len(parts) > 1 else "登录应答无法解析"
        return _LoginResult(error_code, error_msg)

    def _roundtrip(self, msg_type: str, body: str, code: str = "") -> str:
        header = msgheader.to_message_header(msg_type, len(body))
        head_body = header + body
        crc32str = zlib.crc32(bytes(head_body, encoding="utf-8"))
        payload = head_body + cons.MESSAGE_SPLIT + str(crc32str) + "\n"
        sock = self._sock
        if sock is None:
            raise ConnectionFailure("没有可用的 baostock 连接")
        try:
            sock.send(bytes(payload, encoding="utf-8"))
            received = self._recv_all(sock)
        except TimeoutError as exc:
            raise RequestTimeout(f"baostock 读写超时: {exc}", code=code) from exc
        except OSError as exc:
            raise ConnectionFailure(f"baostock 连接异常: {exc}", code=code) from exc
        if not received:
            raise ConnectionFailure("baostock 连接已关闭", code=code)
        if len(received) < cons.MESSAGE_HEADER_LENGTH:
            raise ConnectionFailure("baostock 应答过短", code=code)
        return _decode_response(received, code)

    def _recv_all(self, sock: socket.socket) -> bytes:
        chunks = b""
        while True:
            piece = sock.recv(_RECV_SIZE)
            if not piece:
                break
            chunks += piece
            if _frame_complete(chunks):
                break
        return chunks


def _frame_complete(received: bytes) -> bool:
    """压缩帧以 CDATA 结束；明文帧以换行结束。压缩体里的换行不是帧尾。"""
    if received.endswith(_END_MARK):
        return True
    if len(received) < cons.MESSAGE_HEADER_LENGTH or not received.endswith(b"\n"):
        return False
    try:
        header = received[: cons.MESSAGE_HEADER_LENGTH].decode("utf-8")
        msg_type = header.split(cons.MESSAGE_SPLIT)[1]
    except (UnicodeDecodeError, IndexError):
        return False
    return msg_type not in cons.COMPRESSED_MESSAGE_TYPE_TUPLE


def _is_compressed(received: bytes) -> bool:
    if len(received) < cons.MESSAGE_HEADER_LENGTH:
        return False
    try:
        header = received[: cons.MESSAGE_HEADER_LENGTH].decode("utf-8")
    except UnicodeDecodeError:
        return False
    parts = header.split(cons.MESSAGE_SPLIT)
    return len(parts) > 1 and parts[1] in cons.COMPRESSED_MESSAGE_TYPE_TUPLE


def _decode_response(received: bytes, code: str) -> str:
    header = received[: cons.MESSAGE_HEADER_LENGTH].decode("utf-8", errors="replace")
    if _is_compressed(received):
        parts = header.split(cons.MESSAGE_SPLIT)
        try:
            body_length = int(parts[2])
        except (IndexError, ValueError) as exc:
            raise ConnectionFailure(f"[{code}] 压缩应答头无法解析", code=code) from exc
        start = cons.MESSAGE_HEADER_LENGTH
        try:
            body = zlib.decompress(received[start : start + body_length]).decode("utf-8")
        except (zlib.error, UnicodeDecodeError) as exc:
            raise ConnectionFailure(f"[{code}] 压缩应答无法解压: {exc}", code=code) from exc
        return body
    return received[cons.MESSAGE_HEADER_LENGTH :].decode("utf-8", errors="replace").rstrip("\n")


def _checked_records(raw: str, code: str, failure: str) -> list[list[str]]:
    parts = raw.split(cons.MESSAGE_SPLIT)
    error_code = parts[0] if parts else ""
    error_msg = parts[1] if len(parts) > 1 else ""
    if error_code != cons.BSERR_SUCCESS:
        raise BaostockRequestError(
            f"[{code}] {failure}: {error_code} {error_msg}",
            code=code,
            error_code=error_code,
        )
    return _records(parts, code)


def _records(parts: list[str], code: str) -> list[list[str]]:
    """从应答体取出 record 列表。空数据是合法的空结果，解析失败不是。"""
    payload = parts[6] if len(parts) > 6 else ""
    if not payload.strip():
        return []
    try:
        parsed = json.loads("".join(payload.split()))
    except json.JSONDecodeError as exc:
        raise BaostockRequestError(f"[{code}] 行情应答无法解析: {exc}", code=code) from exc
    records = parsed.get("record", [])
    return [list(row) for row in records]
