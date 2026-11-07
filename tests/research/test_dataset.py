"""数据契约的公开边界；所有手工行情均为 synthetic，非市场证据。"""

from sequoia_x.research.dataset import DatasetSnapshot, audit_dataset


def sample_data() -> dict:
    return {
        "manifest": {
            "source": "hand-worked synthetic fixture",
            "source_version": "1",
            "universe_id": "fixture-two-securities",
            "symbols": ["600001"],
            "start": "2024-01-02",
            "end": "2024-01-03",
            "as_of": "2024-01-03",
            "evidence_kind": "synthetic",
            "universe_complete": True,
            "states_complete": True,
            "actions_complete": True,
            "calendar_complete": True,
        },
        "calendar": ["2024-01-02", "2024-01-03"],
        "bars": [
            {
                "symbol": "600001",
                "date": day,
                "available_on": day,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 10000,
            }
            for day, price in [("2024-01-02", 10), ("2024-01-03", 9)]
        ],
        "states": [
            {
                "symbol": "600001",
                "effective_date": day,
                "available_on": day,
                "listed": True,
                "suspended": False,
                "is_st": False,
                "board": "main",
                "lot_size": 100,
                "limit_rule": "bounded",
                "limit_up": 11,
                "limit_down": 8,
            }
            for day in ["2024-01-02", "2024-01-03"]
        ],
        "actions": [
            {
                "symbol": "600001",
                "ex_date": "2024-01-03",
                "available_on": "2024-01-02",
                "cash_per_share": 1,
                "pay_date": "2024-01-03",
            }
        ],
    }


def test_synthetic_sample_is_replayable_but_never_real_profit_evidence() -> None:
    sample = DatasetSnapshot.model_validate(sample_data())
    report = audit_dataset(sample)
    assert report.status == "data_unqualified"
    assert report.replay_eligible
    assert report.evidence_kind == "synthetic"
    assert report.bar_counts == {"trading": 2, "suspended": 0, "missing": 0, "anomalous": 0}
    assert any("synthetic" in reason for reason in report.reasons)
    # 100 股除息前 1000，除息后 900 市值 + 100 实收现金；不是凭空赚取100。
    assert sample.bars[0].close == 10
    assert sample.bars[1].close == 9
    assert sample.actions[0].cash_per_share == 1


def test_observed_coverage_gaps_and_adjusted_prices_cannot_qualify() -> None:
    data = sample_data()
    data["manifest"]["evidence_kind"] = "observed"
    data["manifest"]["actions_complete"] = False
    data["bars"][0]["price_basis"] = "back_adjusted"
    data["states"][1]["limit_rule"] = "unknown"
    sample = DatasetSnapshot.model_validate(data)
    report = audit_dataset(sample)
    assert not report.replay_eligible
    assert report.status == "data_unqualified"
    assert any("公司行动" in reason for reason in report.reasons)
    assert any("原始" in reason for reason in report.reasons)
    assert any("涨跌停" in reason for reason in report.reasons)


def test_future_known_state_and_action_are_invisible_and_daily_limits_do_not_carry() -> None:
    from datetime import date

    data = sample_data()
    data["states"][1]["available_on"] = "2024-01-04"
    data["actions"][0]["available_on"] = "2024-01-04"
    data["bars"][1]["available_on"] = "2024-01-04"
    sample = DatasetSnapshot.model_validate(data)
    assert sample.state_at("600001", date(2024, 1, 3)) is None
    assert sample.actions_as_of(date(2024, 1, 3)) == ()
    assert len(sample.bars_as_of(date(2024, 1, 3))) == 1
    report = audit_dataset(sample)
    assert report.missing_state_sessions == 1
    assert report.missing_bar_sessions == 1
    assert not report.replay_eligible


def test_sample_save_load_is_stable_and_detects_tampering(tmp_path) -> None:
    import json

    import pytest

    from sequoia_x.research.dataset import load_sample, save_sample

    sample = DatasetSnapshot.model_validate(sample_data())
    path = tmp_path / "sample.json"
    assert save_sample(sample, path) == sample.snapshot_id
    loaded = load_sample(path)
    assert loaded == sample
    assert loaded.snapshot_id == sample.snapshot_id
    payload = json.loads(path.read_text())
    payload["snapshot"]["bars"][0]["close"] = 11
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="指纹"):
        load_sample(path)


def test_source_record_order_does_not_change_dataset_identity() -> None:
    data = sample_data()
    original = DatasetSnapshot.model_validate(data)
    data["bars"].reverse()
    data["calendar"].reverse()
    data["states"].reverse()
    assert DatasetSnapshot.model_validate(data).snapshot_id == original.snapshot_id


def test_existing_adjusted_cache_is_sampled_read_only_and_remains_unqualified(tmp_path) -> None:
    import sqlite3
    from datetime import date
    from hashlib import sha256

    from sequoia_x.research.dataset import sample_sqlite_cache

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE stock_daily(symbol,date,open,high,low,close,volume)")
        connection.executemany(
            "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?)",
            [
                ("600001", "2024-01-02", 20, 20, 20, 20, 10000),
                ("600001", "2024-01-03", 20, 20, 20, 20, 0),
            ],
        )
    before = sha256(path.read_bytes()).hexdigest()
    sample = sample_sqlite_cache(
        path, ("600001",), date(2024, 1, 2), date(2024, 1, 3), as_of=date(2024, 1, 3)
    )
    assert sha256(path.read_bytes()).hexdigest() == before
    assert sample.manifest.evidence_kind == "observed"
    assert sample.bars[0].price_basis == "back_adjusted"
    assert sample.bars[0].close == 20
    report = audit_dataset(sample)
    assert report.status == "data_unqualified"
    assert not report.replay_eligible
    assert report.zero_volume_sessions == 1
    assert report.missing_state_sessions == 2
    assert not sample.manifest.actions_complete


def test_adjusted_warmup_and_future_indicator_prices_cannot_escape_audit() -> None:
    from datetime import date

    data = sample_data()
    data["manifest"]["start"] = "2024-01-03"
    data["bars"][0]["price_basis"] = "back_adjusted"
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert not report.replay_eligible
    assert any("原始" in reason for reason in report.reasons)

    data = sample_data()
    data["manifest"]["indicator_price_basis"] = "point_in_time_adjusted"
    data["manifest"]["indicator_source_version"] = "fixture-announcement-v1"
    data["bars"][0].update(
        {
            "indicator_open": 1,
            "indicator_high": 1,
            "indicator_low": 1,
            "indicator_close": 1,
            "indicator_available_on": "2024-01-04",
        }
    )
    snapshot = DatasetSnapshot.model_validate(data)
    assert snapshot.bars_as_of(date(2024, 1, 2)) == ()
    assert not audit_dataset(snapshot).replay_eligible


def test_suspension_missing_and_zero_volume_are_separate_states() -> None:
    data = sample_data()
    data["states"][0]["suspended"] = True
    data["bars"][0]["status"] = "suspended"
    data["bars"][0]["volume"] = 0
    data["bars"][1]["status"] = "missing"
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert report.bar_counts == {"trading": 0, "suspended": 1, "missing": 1, "anomalous": 0}
    assert report.zero_volume_sessions == 1
    assert not report.replay_eligible


def test_unknown_delisted_members_are_not_dropped_from_declared_universe() -> None:
    data = sample_data()
    data["manifest"]["symbols"].append("600002")
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert report.missing_state_sessions == 2
    assert report.missing_bar_sessions == 2
    assert not report.replay_eligible


def test_duplicate_actions_are_rejected_before_double_counting_cash() -> None:
    data = sample_data()
    data["actions"].append(data["actions"][0].copy())
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert not report.replay_eligible
    assert any("重复公司行动" in reason for reason in report.reasons)


def test_same_economic_action_with_changed_announcement_is_not_paid_twice() -> None:
    data = sample_data()
    data["actions"].append({**data["actions"][0], "available_on": "2024-01-01"})
    snapshot = DatasetSnapshot.model_validate(data)
    assert snapshot.actions[0].action_id == snapshot.actions[1].action_id
    assert not audit_dataset(snapshot).replay_eligible


def test_malformed_point_in_time_indicator_ohlc_cannot_qualify() -> None:
    data = sample_data()
    data["manifest"]["indicator_price_basis"] = "point_in_time_adjusted"
    data["manifest"]["indicator_source_version"] = "fixture-indicators"
    for bar in data["bars"]:
        bar.update(
            {
                "indicator_open": 10,
                "indicator_high": 1,
                "indicator_low": 20,
                "indicator_close": 10,
                "indicator_available_on": bar["date"],
            }
        )
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert not report.replay_eligible
    assert any("指标OHLC" in reason for reason in report.reasons)


def test_warmup_only_sample_cannot_claim_research_window_coverage() -> None:
    data = sample_data()
    data["manifest"]["start"] = "2024-01-03"
    data["calendar"] = data["calendar"][:1]
    data["bars"] = data["bars"][:1]
    data["states"] = data["states"][:1]
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert not report.replay_eligible
    assert any("研究区间没有交易日历" in reason for reason in report.reasons)


def test_conflicting_revisions_on_same_ex_date_are_not_independent_distributions() -> None:
    data = sample_data()
    data["actions"].append(
        {**data["actions"][0], "cash_per_share": 2, "available_on": "2024-01-01"}
    )
    report = audit_dataset(DatasetSnapshot.model_validate(data))
    assert not report.replay_eligible
    assert any("同日公司行动" in reason for reason in report.reasons)
