"""恢复版 flag 的时点观察、形态归属和公共组合账户接入。"""

from collections.abc import Mapping
from datetime import date
from math import isfinite
from typing import Literal

import pandas as pd
from pydantic import Field

from .dataset import DatasetSnapshot, audit_dataset
from .portfolio import PortfolioPolicy, PortfolioResult, run_portfolio
from .protocol import DateWindow, FrozenModel, FrozenProtocol
from .replay import ExecutionPolicy, Signal, signal_frame


class FlagCondition(FrozenModel):
    date: date
    status: Literal["warmup", "data_unavailable", "formed", "not_formed"]
    formed: bool | None
    features: dict[str, float] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def observe_flag(frame: pd.DataFrame, *, on: date) -> FlagCondition:
    """仅观察 on 日及以前的行情；形态成立不代表订单已触发或成交。"""
    required = {"date", "high", "low", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError("flag 行情缺少 date/high/low/volume 列")
    dates = pd.to_datetime(frame["date"]).dt.date
    visible = frame.loc[dates <= on].copy()
    visible["date"] = dates.loc[visible.index]
    visible = visible.sort_values("date")
    if visible["date"].duplicated().any():
        raise ValueError("flag 行情日期重复")
    if visible.empty or visible.iloc[-1]["date"] != on:
        return FlagCondition(
            date=on, status="data_unavailable", formed=None, reasons=("缺少当日有效行情",)
        )
    if len(visible) < 40:
        return FlagCondition(date=on, status="warmup", formed=None, reasons=("不足40根有效行情",))

    recent = visible.tail(40)[["high", "low", "volume"]].apply(pd.to_numeric, errors="coerce")
    if (
        not all(isfinite(value) and value > 0 for value in recent.to_numpy().flat)
        or (recent["high"] < recent["low"]).any()
    ):
        return FlagCondition(
            date=on, status="data_unavailable", formed=None,
            reasons=("最近40根行情存在无效价格、缺失或零成交量",),
        )
    tail40, tail10 = recent, recent.tail(10)
    high40, low40 = float(tail40["high"].max()), float(tail40["low"].min())
    high10, low10 = float(tail10["high"].max()), float(tail10["low"].min())
    previous_volume = float(recent["volume"].iloc[-21:-1].mean())
    volume = float(recent["volume"].iloc[-1])
    features = {
        "high40": high40, "low40": low40, "high10": high10, "low10": low10,
        "previous_vol_ma20": previous_volume, "volume": volume,
        "momentum_ratio": high40 / low40,
        "consolidation_ratio": high10 / low10,
        "high_level_ratio": low10 / high40,
        "volume_ratio": volume / previous_volume,
    }
    formed = (
        high40 / low40 > 1.6
        and high10 / low10 < 1.15
        and low10 >= high40 * 0.8
        and volume < previous_volume * 0.6
    )
    return FlagCondition(
        date=on, status="formed" if formed else "not_formed", formed=formed, features=features
    )


class FlagObservation(FlagCondition):
    symbol: str
    in_research_window: bool
    episode_id: str | None = None
    episode_start_date: date | None = None
    is_episode_start: bool = False
    signal_id: str | None = None


class FlagCandidateDay(FrozenModel):
    date: date
    formed_candidates: int
    episode_starts: int


class FlagCoverage(FrozenModel):
    total_observations: int
    pre_window_observations: int
    formed_observations: int
    repeated_observations: int
    episode_count: int
    episode_start_dates: tuple[date, ...]
    signal_dates: tuple[date, ...]
    warmup_observations: int
    unavailable_observations: int
    candidates_by_date: tuple[FlagCandidateDay, ...]


class FlagReplay(FrozenModel):
    implementation_version: Literal["restored-flag-episodes-v1"] = "restored-flag-episodes-v1"
    data_snapshot_id: str
    through: date
    evidence_kind: Literal["synthetic", "observed"]
    data_status: Literal["qualified", "data_unqualified"]
    qualification_reasons: tuple[str, ...]
    observations: tuple[FlagObservation, ...]
    signals: tuple[Signal, ...]
    coverage: FlagCoverage


def replay_flag_signals(
    snapshot: DatasetSnapshot, *, through: date | None = None,
    window: DateWindow | None = None,
) -> FlagReplay:
    """保留所有日常成立日；episode 仅描述相关性，不添加新的入场规则。"""
    start = window.start if window else snapshot.manifest.start
    end = window.end if window else snapshot.manifest.end
    if start < snapshot.manifest.start or end > snapshot.manifest.end:
        raise ValueError("flag 重放窗口超出数据样本范围")
    cutoff = min(through or end, end, snapshot.manifest.as_of)
    observations: list[FlagObservation] = []
    signals: list[Signal] = []
    active: dict[str, tuple[str, date]] = {}
    for session in snapshot.calendar:
        if session > cutoff:
            break
        for symbol in snapshot.manifest.symbols:
            condition = observe_flag(signal_frame(snapshot, symbol, session), on=session)
            first = False
            signal = None
            if condition.formed:
                if symbol not in active:
                    active[symbol] = (f"flag:{symbol}:{session.isoformat()}", session)
                    first = True
                if session >= start:
                    signal = Signal(
                        symbol=symbol, signal_date=session, strategy="flag",
                        features=condition.features, episode_id=active[symbol][0],
                    )
                    signals.append(signal)
            elif condition.formed is False:
                active.pop(symbol, None)
            episode = active.get(symbol)
            observations.append(FlagObservation(
                **condition.model_dump(), symbol=symbol,
                in_research_window=session >= start,
                episode_id=episode[0] if episode else None,
                episode_start_date=episode[1] if episode else None,
                is_episode_start=first, signal_id=signal.signal_id if signal else None,
            ))

    in_window = [item for item in observations if item.in_research_window]
    formed = [item for item in in_window if item.formed]
    days = sorted({item.date for item in in_window})
    audit = audit_dataset(snapshot)
    coverage = FlagCoverage(
        total_observations=len(observations),
        pre_window_observations=len(observations) - len(in_window),
        formed_observations=len(formed),
        repeated_observations=sum(not item.is_episode_start for item in formed),
        episode_count=len({item.episode_id for item in formed}),
        episode_start_dates=tuple(sorted({
            item.episode_start_date for item in formed if item.episode_start_date is not None
        })),
        signal_dates=tuple(sorted({item.signal_date for item in signals})),
        warmup_observations=sum(item.status == "warmup" for item in in_window),
        unavailable_observations=sum(item.status == "data_unavailable" for item in in_window),
        candidates_by_date=tuple(FlagCandidateDay(
            date=day,
            formed_candidates=sum(item.date == day for item in formed),
            episode_starts=sum(item.date == day and item.is_episode_start for item in formed),
        ) for day in days),
    )
    return FlagReplay(
        data_snapshot_id=snapshot.snapshot_id, through=cutoff,
        evidence_kind=snapshot.manifest.evidence_kind, data_status=audit.status,
        qualification_reasons=audit.reasons, observations=tuple(observations),
        signals=tuple(signals), coverage=coverage,
    )


class FlagBaseline(FrozenModel):
    replay: FlagReplay
    portfolio: PortfolioResult


def run_flag_portfolio(
    snapshot: DatasetSnapshot,
    frozen_protocol: FrozenProtocol,
    execution_policy: ExecutionPolicy,
    selection_policy: PortfolioPolicy,
    *,
    benchmark: Mapping[date, float] | None = None,
    window: DateWindow | None = None,
) -> FlagBaseline:
    """形态观察与同一公开组合账户一起返回；不为 flag 添加独立执行假设。"""
    replay = replay_flag_signals(snapshot, window=window)
    portfolio = run_portfolio(
        snapshot, frozen_protocol, execution_policy, selection_policy,
        signals=replay.signals, benchmark=benchmark, window=window,
    )
    return FlagBaseline(replay=replay, portfolio=portfolio)
