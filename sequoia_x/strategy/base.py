"""策略基类模块：定义所有选股策略的抽象接口。"""

from abc import ABC, abstractmethod

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


class BaseStrategy(ABC):
    """选股策略抽象基类。

    所有具体策略必须继承此类并实现 run() 方法。

    Attributes:
        webhook_key: 策略对应的通知分组标识，用于 Bark group。
            默认为 'default'。子类可覆盖，例如 'ma_volume'。
    """

    webhook_key: str = "default"

    def __init__(self, engine: DataEngine, settings: Settings) -> None:
        """
        初始化策略。

        Args:
            engine: DataEngine 实例，用于读取行情数据。
            settings: Settings 实例，用于读取配置。
        """
        self.engine = engine
        self.settings = settings

    @abstractmethod
    def run(self) -> list[str]:
        """
        执行选股逻辑，返回选中的股票代码列表。

        Returns:
            满足策略条件的股票代码列表，如 ['000001', '600519']。
            无选股结果时返回空列表。
        """
        ...
