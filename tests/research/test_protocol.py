"""从公开接口验证研究协议生命周期。"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from sequoia_x.research import ResearchProtocol, ResearchRegistry


def test_freeze_and_reopen_a_valid_protocol(tmp_path: Path, protocol_data: dict) -> None:
    protocol = ResearchProtocol.model_validate(protocol_data)
    registry = ResearchRegistry(tmp_path / "research.sqlite")

    frozen = registry.freeze(protocol)
    reopened = ResearchRegistry(tmp_path / "research.sqlite").get_protocol(frozen.protocol_id)

    assert reopened == frozen
    assert reopened.revision == 1
    assert reopened.parent_id is None
    assert reopened.protocol == protocol
    assert reopened.protocol.exploration_reasons() == ()
    assert reopened.protocol.trading.holding_sessions == 60
    assert len(frozen.protocol_id) == 64


def test_missing_decisions_are_exploratory(tmp_path: Path, protocol_data: dict) -> None:
    protocol_data["acceptance"] = {}
    protocol_data["statistics"] = {}
    protocol = ResearchProtocol.model_validate(protocol_data)

    frozen = ResearchRegistry(tmp_path / "research.sqlite").freeze(protocol)

    assert "缺少 acceptance.minimum_return_delta" in frozen.protocol.exploration_reasons()
    assert "缺少 statistics.power_target" in frozen.protocol.exploration_reasons()


@pytest.mark.parametrize(
    "change",
    [
        {"validation": {"start": "2022-12-31", "end": "2023-12-31"}},
        {"final_test": {"start": "2023-12-31", "end": "2025-12-31"}},
        {"data_cutoff": "2025-12-30"},
        {"train": {"start": "2022-01-01", "end": "2020-01-01"}},
        {"unknown_threshold": 0},
    ],
)
def test_invalid_protocol_is_rejected(protocol_data: dict, change: dict) -> None:
    with pytest.raises(ValidationError):
        ResearchProtocol.model_validate({**protocol_data, **change})


def test_modification_creates_an_immutable_linked_revision(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    first = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    protocol_data["acceptance"]["minimum_return_delta"] = 0.01
    second = registry.freeze(ResearchProtocol.model_validate(protocol_data))

    assert first.protocol_id != second.protocol_id
    assert second.parent_id == first.protocol_id
    assert second.revision == 2
    assert registry.get_protocol(first.protocol_id).protocol.acceptance.minimum_return_delta == 0.02
    assert registry.protocols("ma-baseline") == (first, second)
    with pytest.raises(ValidationError):
        first.protocol.acceptance.minimum_return_delta = 0


def test_external_database_modification_is_detected(tmp_path: Path, protocol_data: dict) -> None:
    path = tmp_path / "research.sqlite"
    registry = ResearchRegistry(path)
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE events SET payload = replace(payload, 'ma-v1', 'forged-v1')")

    with pytest.raises(ValueError, match="完整性校验失败"):
        ResearchRegistry(path).get_protocol(frozen.protocol_id)


def test_unknown_or_previously_selected_test_data_is_exploratory(protocol_data: dict) -> None:
    protocol_data.pop("final_test_status")
    assert any(
        "未触碰" in reason
        for reason in (ResearchProtocol.model_validate(protocol_data).exploration_reasons())
    )
    protocol_data["final_test_status"] = "used_for_selection"
    assert ResearchProtocol.model_validate(protocol_data).exploration_reasons()


def test_cli_validates_freezes_and_reads_history(tmp_path: Path, protocol_data: dict) -> None:
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol_data), encoding="utf-8")
    database = tmp_path / "research.sqlite"

    def invoke(*args: str) -> dict | list:
        result = subprocess.run(
            [sys.executable, "-m", "sequoia_x.research", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    validated = invoke("validate", str(source))
    assert isinstance(validated, dict)
    assert validated["status"] == "ready_for_confirmation"
    frozen = invoke("freeze", str(source), "--registry", str(database))
    history = invoke("history", "ma-baseline", "--registry", str(database))
    assert history == [frozen]
