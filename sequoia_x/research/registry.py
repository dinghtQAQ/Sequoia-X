"""带哈希链的本地追加式研究登记簿。"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from .evaluation import evaluate
from .experiments import Decision, EvaluationEvidence, EvidenceKind, Experiment, Purpose, Stage
from .protocol import DateWindow, FrozenProtocol, ResearchProtocol


class ResearchRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL UNIQUE)"
            )

    @staticmethod
    def _read(connection: sqlite3.Connection) -> list[dict[str, Any]]:
        previous = ""
        events = []
        for payload, digest in connection.execute(
            "SELECT payload, digest FROM events ORDER BY sequence"
        ):
            expected = hashlib.sha256((previous + payload).encode("utf-8")).hexdigest()
            if digest != expected:
                raise ValueError("研究登记簿完整性校验失败：记录已被修改")
            event = json.loads(payload)
            events.append({**event, "id": digest})
            previous = digest
        return events

    @staticmethod
    def _append(connection: sqlite3.Connection, event: dict[str, Any]) -> str:
        row = connection.execute(
            "SELECT digest FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = row[0] if row else ""
        payload = json.dumps(
            event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        digest = hashlib.sha256((previous + payload).encode("utf-8")).hexdigest()
        connection.execute("INSERT INTO events(payload, digest) VALUES (?, ?)", (payload, digest))
        return digest

    @staticmethod
    def _frozen(event: dict[str, Any]) -> FrozenProtocol:
        return FrozenProtocol(
            protocol_id=event["id"],
            revision=event["revision"],
            parent_id=event["parent_id"],
            frozen_at=event["frozen_at"],
            protocol=ResearchProtocol.model_validate(event["protocol"]),
        )

    def freeze(self, protocol: ResearchProtocol) -> FrozenProtocol:
        # 重新校验，避免 model_copy(update=...) 等绕过 Pydantic 校验的对象进入登记簿。
        protocol = ResearchProtocol.model_validate_json(protocol.model_dump_json())
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            events = self._read(connection)
            previous = [
                e
                for e in events
                if e["kind"] == "protocol" and e["protocol"]["study_id"] == protocol.study_id
            ]
            event = {
                "kind": "protocol",
                "revision": len(previous) + 1,
                "parent_id": previous[-1]["id"] if previous else None,
                "frozen_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.10
                "protocol": protocol.model_dump(mode="json"),
            }
            protocol_id = self._append(connection, event)
            return self._frozen({**event, "id": protocol_id})

    def get_protocol(self, protocol_id: str) -> FrozenProtocol:
        with sqlite3.connect(self.path) as connection:
            for event in self._read(connection):
                if event["id"] == protocol_id and event["kind"] == "protocol":
                    return self._frozen(event)
        raise KeyError(f"未知协议：{protocol_id}")

    def protocols(self, study_id: str) -> tuple[FrozenProtocol, ...]:
        with sqlite3.connect(self.path) as connection:
            return tuple(
                self._frozen(e)
                for e in self._read(connection)
                if e["kind"] == "protocol" and e["protocol"]["study_id"] == study_id
            )

    @staticmethod
    def _experiment(event: dict[str, Any]) -> Experiment:
        return Experiment.model_validate(
            {key: value for key, value in event.items() if key not in ("kind", "id")}
            | {
                "experiment_id": event["id"],
            }
        )

    def _test_data_used(
        self,
        events: list[dict[str, Any]],
        window: DateWindow,
        exclude_id: str | None = None,
    ) -> bool:
        return any(
            e["kind"] == "experiment"
            and e["id"] != exclude_id
            and window.overlaps(self._experiment(e).used_window)
            for e in events
        )

    def start_experiment(
        self,
        protocol_id: str,
        *,
        parameters: dict[str, JsonValue],
        stage: Stage,
        purpose: Purpose,
        evidence_kind: EvidenceKind,
    ) -> Experiment:
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            events = self._read(connection)
            frozen = next(
                (
                    self._frozen(e)
                    for e in events
                    if e["id"] == protocol_id and e["kind"] == "protocol"
                ),
                None,
            )
            if frozen is None:
                raise KeyError(f"未知协议：{protocol_id}")
            protocol = frozen.protocol
            if purpose == "confirmation":
                if protocol.exploration_reasons():
                    raise ValueError(
                        "协议缺少预定决策，只能探索：" + "; ".join(protocol.exploration_reasons())
                    )
                if stage != "final_test":
                    raise ValueError("确认性验收必须使用最终测试区间")
                current = self._latest_approved(events, protocol.study_id, evidence_kind)
                if current is not None and protocol.comparison_role == "initial_reference":
                    raise ValueError("已有通过版本，不能重新使用初始参考模式")
                if current is not None and protocol.comparison_version != current:
                    raise ValueError("确认性实验必须相对最近通过版本进行配对比较")
                if self._test_data_used(events, protocol.final_test):
                    raise ValueError("最终测试日期已使用或已保留，不能再声明为未触碰测试集")
            experiment = Experiment(
                experiment_id="pending",
                protocol_id=protocol_id,
                study_id=protocol.study_id,
                strategy_version=protocol.strategy_version,
                comparison_version=protocol.comparison_version,
                evidence_kind=evidence_kind,
                stage=stage,
                purpose=purpose,
                used_window=getattr(protocol, stage),
                parameters_json=json.dumps(
                    parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                started_at=datetime.now(timezone.utc),  # noqa: UP017 - Python 3.10
            )
            event = {
                "kind": "experiment",
                **experiment.model_dump(
                    mode="json",
                    exclude={"experiment_id"},
                ),
            }
            experiment_id = self._append(connection, event)
            return self._experiment({**event, "id": experiment_id})

    def experiments(self, study_id: str) -> tuple[Experiment, ...]:
        with sqlite3.connect(self.path) as connection:
            return tuple(
                self._experiment(e)
                for e in self._read(connection)
                if e["kind"] == "experiment" and e["study_id"] == study_id
            )

    @staticmethod
    def _latest_approved(
        events: list[dict[str, Any]],
        study_id: str,
        evidence_kind: EvidenceKind,
    ) -> str | None:
        for event in reversed(events):
            if (
                event["kind"] == "decision"
                and event["study_id"] == study_id
                and event["evidence_kind"] == evidence_kind
                and event["status"] == "adopted"
            ):
                return event["active_version"]
        return None

    def latest_approved(self, study_id: str, evidence_kind: EvidenceKind) -> str | None:
        with sqlite3.connect(self.path) as connection:
            return self._latest_approved(self._read(connection), study_id, evidence_kind)

    @staticmethod
    def _decision(event: dict[str, Any]) -> Decision:
        return Decision.model_validate({k: v for k, v in event.items() if k not in ("kind", "id")})

    def decisions(self, study_id: str) -> tuple[Decision, ...]:
        with sqlite3.connect(self.path) as connection:
            return tuple(
                self._decision(e)
                for e in self._read(connection)
                if e["kind"] == "decision" and e["study_id"] == study_id
            )

    def finish_experiment(
        self,
        experiment_id: str,
        *,
        evidence: EvaluationEvidence | None = None,
        failure_reason: str | None = None,
    ) -> Decision:
        if evidence is not None:
            evidence = EvaluationEvidence.model_validate_json(evidence.model_dump_json())
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            events = self._read(connection)
            run = next(
                (
                    self._experiment(e)
                    for e in events
                    if e["kind"] == "experiment" and e["id"] == experiment_id
                ),
                None,
            )
            if run is None:
                raise KeyError(f"未知实验：{experiment_id}")
            if any(e["kind"] == "decision" and e["experiment_id"] == experiment_id for e in events):
                raise ValueError("实验已有结果，不能覆盖；请登记新实验")
            frozen = next(self._frozen(e) for e in events if e["id"] == run.protocol_id)
            current = self._latest_approved(events, run.study_id, run.evidence_kind)
            active = current or run.comparison_version
            status = "insufficient_evidence"
            reasons = (failure_reason or "缺少评估证据",)
            if run.purpose == "exploration":
                status, reasons = (
                    "exploratory",
                    (failure_reason or "探索性实验不得生成上线通过结论",),
                )
            elif failure_reason:
                pass
            elif current is not None and frozen.protocol.comparison_role == "initial_reference":
                reasons = ("已有通过版本，已登记的初始参考实验也不能重新起步",)
            elif current is not None and current != run.comparison_version:
                reasons = ("对照不是最近通过版本，须重新冻结配对比较协议",)
            elif self._test_data_used(events, run.used_window, exclude_id=run.experiment_id):
                reasons = ("最终测试日期已使用于其他实验，不能生成独立确认结论",)
            elif evidence is not None and (
                evidence.experiment_id != run.experiment_id
                or evidence.evidence_kind != run.evidence_kind
                or evidence.evaluation_window != run.used_window
            ):
                reasons = ("证据的实验标识、来源或评估区间与实验登记不符",)
            elif evidence is not None:
                status, reasons = evaluate(frozen.protocol, frozen.protocol_id, evidence)
                if status == "adopted":
                    active = run.strategy_version
            decision = Decision(
                experiment_id=experiment_id,
                protocol_id=run.protocol_id,
                study_id=run.study_id,
                evidence_kind=run.evidence_kind,
                status=status,
                active_version=active,
                reasons=reasons,
                evidence=evidence,
                failure_reason=failure_reason,
                finished_at=datetime.now(timezone.utc),  # noqa: UP017 - Python 3.10
            )
            self._append(connection, {"kind": "decision", **decision.model_dump(mode="json")})
            return decision
