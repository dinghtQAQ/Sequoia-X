"""Bark 通知行为测试。"""

import json
import logging
from unittest.mock import MagicMock, patch

from sequoia_x.core.config import Settings
from sequoia_x.notify.bark import BarkNotifier


def make_settings(
    bark_server: str = "https://api.day.app",
    bark_key: str = "device-key",
) -> Settings:
    return Settings(
        db_path="data/test.db",
        start_date="2024-01-01",
        bark_server=bark_server,
        bark_key=bark_key,
    )


def test_send_posts_all_symbols_to_bark_push_url() -> None:
    """send() 把全部代码发到 BARK_SERVER/BARK_KEY，并用策略名做分组。"""
    settings = make_settings(
        bark_server="https://api.day.app/",
        bark_key="abc123",
    )
    notifier = BarkNotifier(settings)

    with patch("requests.post") as mock_post, patch.object(
        BarkNotifier, "_get_stock_names", return_value={"000001": "平安银行"}
    ):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"code": 200})
        notifier.send(
            symbols=["000001", "600519"],
            strategy_name="MaVolumeStrategy",
            webhook_key="ma_volume",
        )

    assert mock_post.call_args.args[0] == "https://api.day.app/abc123"
    body = json.loads(mock_post.call_args.kwargs["data"])
    assert body["title"] == "Sequoia-X 选股播报 | MaVolumeStrategy"
    assert body["group"] == "ma_volume"
    assert "000001" in body["body"]
    assert "600519" in body["body"]
    assert "平安银行" in body["body"]
    assert "https://xueqiu.com/S/SZ000001" in body["body"]
    assert "https://xueqiu.com/S/SH600519" in body["body"]


def test_http_failure_logs_error_without_raising() -> None:
    """非成功响应时记录 ERROR，不把异常抛给调用方。"""
    import sequoia_x.notify.bark as bark_module

    notifier = BarkNotifier(make_settings())
    bark_logger = logging.getLogger(bark_module.__name__)
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(logging.ERROR)
    bark_logger.addHandler(handler)
    try:
        with patch("requests.post") as mock_post, patch.object(
            BarkNotifier, "_get_stock_names", return_value={}
        ):
            mock_post.return_value = MagicMock(
                status_code=500,
                text="error",
                json=lambda: {"code": 500},
            )
            notifier.send(symbols=["000001"], strategy_name="Test", webhook_key="default")
    finally:
        bark_logger.removeHandler(handler)

    assert any(record.levelno == logging.ERROR for record in records)
