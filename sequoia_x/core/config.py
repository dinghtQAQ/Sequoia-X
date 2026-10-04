"""配置管理模块：通过 pydantic-settings 从环境变量或 .env 文件加载系统配置。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_path: str = "data/sequoia_v2.db"
    start_date: str = "2024-01-01"
    bark_server: str = "https://api.day.app"
    bark_key: str  # 必填字段，缺失时抛出 ValidationError
    feishu_webhook_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # <--- 加上这一行！让 Pydantic 放行未定义的变量
    )

    def get_webhook_url(self, webhook_key: str) -> str:
        """
        返回 Bark 推送地址。

        webhook_key 不再选择不同 URL，由通知层用作 Bark group。
        地址由 bark_server 与 bark_key 拼接。

        Args:
            webhook_key: 策略标识，保留以兼容现有调用面。

        Returns:
            Bark 设备推送 URL。
        """
        return f"{self.bark_server.rstrip('/')}/{self.bark_key}"


_settings: Settings | None = None


def get_settings() -> Settings:
    """返回全局 Settings 单例。

    首次调用时从环境变量或 .env 文件加载配置。
    若必填字段（bark_key）缺失，抛出 pydantic_core.ValidationError。

    Returns:
        Settings: 全局唯一的配置实例。

    Raises:
        pydantic_core.ValidationError: 当必填字段缺失或字段类型不匹配时抛出。
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
