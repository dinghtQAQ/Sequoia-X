"""Bark 通知模块：将选股结果通过 Bark 推送至设备。"""

import json
from datetime import date

import requests

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class BarkNotifier:
    """Bark 推送器。

    与 FeishuNotifier 保持同一调用面：send(symbols, strategy_name, webhook_key)。
    webhook_key 不再路由到不同 URL，而是作为 Bark 的 group，便于按策略归档。
    推送地址由 Settings.bark_server 与 Settings.bark_key 拼接。
    """

    def __init__(self, settings: Settings) -> None:
        """
        初始化 BarkNotifier。

        Args:
            settings: Settings 实例，提供 Bark 服务器与设备 Key。
        """
        self.settings = settings

    @staticmethod
    def _to_xueqiu_code(code: str) -> str:
        """将纯数字代码转为雪球格式：6开头→SH，4/8开头→BJ，其余→SZ。"""
        if code.startswith("6"):
            return f"SH{code}"
        elif code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    @staticmethod
    def _get_stock_names(symbols: list[str]) -> dict[str, str]:
        """通过 baostock 批量查询股票名称，返回 {code: name} 映射。"""
        from sequoia_x.data.baostock_session import (
            BaostockRequestError,
            BaostockSession,
            LoginFailure,
        )

        session = BaostockSession(timeout=30, max_retries=0)
        mapping = {}
        try:
            for code in symbols:
                prefix = "sh" if code.startswith(("6", "9")) else "sz"
                try:
                    rows = session.query_stock_basic(code=f"{prefix}.{code}")
                except LoginFailure as exc:
                    logger.error(f"股票名称查询登录失败，停止查询: {exc}")
                    break
                except BaostockRequestError as exc:
                    logger.error(f"[{code}] 股票名称查询失败: {exc.kind}: {exc}")
                    continue
                if rows:
                    mapping[code] = rows[0][1]  # 第2个字段是股票名称
        finally:
            session.close()
        return mapping

    def _build_message(self, symbols: list[str], strategy_name: str, group: str) -> dict:
        today = date.today().strftime("%Y-%m-%d")
        names = self._get_stock_names(symbols)

        lines: list[str] = []
        for code in symbols:
            xq_code = self._to_xueqiu_code(code)
            name = names.get(code, xq_code)
            lines.append(f"{name} {code} https://xueqiu.com/S/{xq_code}")

        symbol_text = "\n".join(lines) if lines else "（无选股结果）"
        return {
            "title": f"Sequoia-X 选股播报 | {strategy_name}",
            "body": f"日期：{today}\n策略：{strategy_name}\n选股数量：{len(symbols)}\n{symbol_text}",
            "group": group,
        }

    def send(
        self,
        symbols: list[str],
        strategy_name: str,
        webhook_key: str = "default",
    ) -> None:
        """
        将选股结果格式化为 Bark 消息并 POST 至设备推送地址。

        Args:
            symbols: 选股结果代码列表。
            strategy_name: 策略名称，用于消息标题。
            webhook_key: 策略标识，作为 Bark group。

        Raises:
            不抛出异常，HTTP 失败时记录 ERROR 日志。
        """
        url = self.settings.get_webhook_url(webhook_key)
        payload = self._build_message(symbols, strategy_name, webhook_key)

        try:
            resp = requests.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp_json = resp.json()
            if resp.status_code != 200 or resp_json.get("code") != 200:
                logger.error(
                    f"Bark 推送失败 [{webhook_key}] "
                    f"HTTP状态={resp.status_code} Bark响应={resp.text}"
                )
            else:
                logger.info(f"Bark 推送成功 [{webhook_key}]，共 {len(symbols)} 只股票")

        except requests.RequestException as exc:
            logger.error(f"Bark 推送请求异常 [{webhook_key}]：{exc}")
