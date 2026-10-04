"""Bark 配置行为测试。"""

import os

import pytest
from pydantic import ValidationError

from sequoia_x.core.config import Settings


def test_bark_settings_come_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """BARK_SERVER 与 BARK_KEY 从环境变量进入 Settings。"""
    monkeypatch.setenv("BARK_SERVER", "https://bark.example")
    monkeypatch.setenv("BARK_KEY", "device-key")

    settings = Settings(_env_file=None)

    assert settings.bark_server == "https://bark.example"
    assert settings.bark_key == "device-key"
    assert settings.get_webhook_url("ma_volume") == "https://bark.example/device-key"


def test_missing_bark_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺少 BARK_KEY 时 Settings 不能启动。"""
    monkeypatch.setenv("BARK_SERVER", "https://api.day.app")
    monkeypatch.delenv("BARK_KEY", raising=False)
    env_backup = os.environ.pop("BARK_KEY", None)
    try:
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)
        assert "bark_key" in str(exc_info.value).lower()
    finally:
        if env_backup is not None:
            os.environ["BARK_KEY"] = env_backup
