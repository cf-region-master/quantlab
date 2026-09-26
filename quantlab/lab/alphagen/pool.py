from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .data import MarketData, MarketSplit
from .expressions import Expr


def normalize_cs(x: Tensor) -> Tensor:
    """Cross-sectional z-score using native NaN reductions, without a boolean pair mask."""
    mean = torch.nanmean(x, dim=-1, keepdim=True)
    centered = x - mean
    std = torch.nanmean(centered.square(), dim=-1, keepdim=True).sqrt()
    return centered / std.clamp_min(1e-6)


def batch_corr(x: Tensor, y: Tensor) -> Tensor:
    """Pearson correlation over the last axis; unavailable cross-sections become zero."""
    # Addition with 0 propagates the other input's NaN without allocating a boolean mask.
    paired_x = x + y * 0
    paired_y = y + x * 0
    xc = paired_x - torch.nanmean(paired_x, dim=-1, keepdim=True)
    yc = paired_y - torch.nanmean(paired_y, dim=-1, keepdim=True)
    numerator = torch.nansum(xc * yc, dim=-1)
    denominator = (
        torch.nansum(xc.square(), dim=-1) * torch.nansum(yc.square(), dim=-1)
    ).sqrt()
    return torch.nan_to_num(numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class PreparedCandidate:
    expr: Expr
    token: str
    factors: dict[str, Tensor]
    single_ics: dict[str, float]
    coverages: dict[str, float]
    ic_series: dict[str, Tensor]


@dataclass
class PoolEntry:
    expr: Expr
    token: str
    single_ics: dict[str, float]
    coverages: dict[str, float]
    ic_series: dict[str, Tensor]

    @property
    def single_ic(self) -> float:
        return self.single_ics["train"]


@dataclass
class CandidateEvent:
    token: str
    ic: float
    reward: float
    depth: int
    length: int
    seen_before: bool
    accepted: bool


class Calculator:
    """Evaluate each complete expression in chunks against pre-separated data splits."""

    def __init__(self, market: MarketData, split: str, eval_chunk_bars: int = 4096,
                 factor_storage_dtype: str = "fp16"):
        self.market, self.split, self.eval_chunk_bars = market, split, eval_chunk_bars
        self.factor_storage_dtype = torch.float16 if factor_storage_dtype == "fp16" else torch.float32
        self.targets = {
            name: normalize_cs(part.target) for name, part in market.splits.items()
        }
        self._stream_local = threading.local()

    def view(self, split: str) -> "Calculator":
        view = object.__new__(Calculator)
        view.__dict__ = self.__dict__.copy()
        view.split = split
        return view

    @property
    def device(self) -> torch.device:
        return next(iter(self.market.splits.values())).values.device

    def _stream(self):
        if self.device.type != "cuda":
            return None
        if not hasattr(self._stream_local, "stream"):
            self._stream_local.stream = torch.cuda.Stream(device=self.device)
        return self._stream_local.stream

    def _chunk_factor(self, expr: Expr, part: MarketSplit, start: int, stop: int) -> Tensor:
        context_start = max(0, start - expr.lookback)
        value = expr.evaluate(part.values[context_start:stop])[start - context_start:]
        value[~torch.isfinite(value)] = torch.nan
        if self.market.same_day_label and expr.lookback:
            value[part.bar_in_day[start:stop] < expr.lookback] = torch.nan
        return normalize_cs(value)

    @staticmethod
    def _series_stats(series: Tensor) -> tuple[float, float]:
        valid = series != 0
        coverage = float(valid.float().mean()) if len(series) else 0.0
        ic = float(series[valid].mean()) if valid.any() else float("nan")
        return ic, coverage

    def prepare(self, expr: Expr, token: str) -> PreparedCandidate:
        factors: dict[str, Tensor] = {}
        series_gpu: dict[str, list[Tensor]] = {name: [] for name in self.market.splits}
        stream = self._stream()
        stream_context = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
        with torch.no_grad(), stream_context:
            for name, part in self.market.splits.items():
                bars, stocks = part.target.shape
                factor = torch.empty((bars, stocks), dtype=self.factor_storage_dtype, device=self.device)
                for start in range(0, bars, self.eval_chunk_bars):
                    stop = min(bars, start + self.eval_chunk_bars)
                    value = self._chunk_factor(expr, part, start, stop)
                    factor[start:stop].copy_(value)
                    series_gpu[name].append(batch_corr(value, self.targets[name][start:stop]))
                factors[name] = factor
        if stream is not None:
            stream.synchronize()
        ic_series = {name: torch.cat(parts).cpu() for name, parts in series_gpu.items()}
        stats = {name: self._series_stats(values) for name, values in ic_series.items()}
        single_ics = {name: value[0] for name, value in stats.items()}
        coverages = {name: value[1] for name, value in stats.items()}
        return PreparedCandidate(expr, token, factors, single_ics, coverages, ic_series)

    def mutual_ics(self, candidate: Tensor, pool_factors: Tensor, n: int) -> np.ndarray:
        if n == 0:
            return np.empty(0, dtype=np.float64)
        sums = torch.zeros(n, dtype=torch.float64, device=self.device)
        counts = torch.zeros(n, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            for start in range(0, len(candidate), self.eval_chunk_bars):
                stop = min(len(candidate), start + self.eval_chunk_bars)
                matrix = pool_factors[:n, start:stop].float()
                corr = batch_corr(matrix, candidate[start:stop].float().unsqueeze(0))
                valid = corr != 0
                sums.add_(corr.sum(1).double())
                counts.add_(valid.sum(1))
        result = sums / counts.clamp_min(1)
        result[counts == 0] = torch.nan
        return result.cpu().numpy()

    def pool_metrics(self, factor_store: dict[str, Tensor], n: int, weights: np.ndarray,
                     split: str, override: tuple[int, Tensor] | None = None) -> dict[str, float | int]:
        if n == 0:
            return {"ic": 0.0, "icir": 0.0, "coverage": 0.0, "effective_bars": 0}
        factors = factor_store[split]
        target = self.targets[split]
        weight = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        pieces: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, len(target), self.eval_chunk_bars):
                stop = min(len(target), start + self.eval_chunk_bars)
                matrix = factors[:n, start:stop].float().nan_to_num()
                ensemble = torch.einsum("n,nbs->bs", weight, matrix)
                if override is not None:
                    index, candidate = override
                    old = factors[index, start:stop].float().nan_to_num()
                    new = candidate[start:stop].float().nan_to_num()
                    ensemble.add_(new - old, alpha=float(weight[index]))
                pieces.append(batch_corr(ensemble, target[start:stop]).cpu())
        series = torch.cat(pieces)
        valid = series != 0
        effective = int(valid.sum())
        coverage = effective / len(series) if len(series) else 0.0
        if effective == 0:
            return {"ic": float("nan"), "icir": float("nan"), "coverage": coverage, "effective_bars": 0}
        values = series[valid]
        mean, std = values.mean(), values.std(unbiased=False)
        return {
            "ic": float(mean), "icir": float(mean / std.clamp_min(1e-12)),
            "coverage": coverage, "effective_bars": effective,
        }


class AlphaPool:
    def __init__(self, capacity: int, train: Calculator, eviction: Literal["baseline", "counterfactual"],
                 novelty_coef: float = 0.0, ridge: float = 1e-3, mutual_threshold: float = 0.995,
                 factor_workers: int = 4):
        self.capacity, self.train, self.eviction = capacity, train, eviction
        self.novelty_coef, self.ridge = novelty_coef, ridge
        self.mutual_threshold, self.factor_workers = mutual_threshold, factor_workers
        self.entries: list[PoolEntry] = []
        self.weights = np.empty(0, dtype=np.float32)
        self.corr = np.eye(capacity, dtype=np.float64)
        self.factor_store = {
            name: torch.empty((capacity, *part.target.shape), dtype=train.factor_storage_dtype,
                              device=train.device)
            for name, part in train.market.splits.items()
        }
        self.events: list[CandidateEvent] = []
        self.counts: dict[str, int] = {}
        self._current_ic = 0.0
        self._eval_candidates = 0
        self._eval_seconds = 0.0
        self._executor = ThreadPoolExecutor(max_workers=factor_workers) if factor_workers > 1 else None

    def _optimize(self, single: np.ndarray, corr: np.ndarray) -> np.ndarray:
        if not len(single):
            return np.empty(0, dtype=np.float32)
        system = corr + self.ridge * np.eye(len(single))
        try:
            weights = np.linalg.solve(system, single)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(system, single, rcond=None)[0]
        norm = np.abs(weights).sum()
        return (weights / norm if norm > 1e-12 else weights).astype(np.float32)

    def metrics(self, calculator: Calculator | None = None, exact: bool = True) -> dict[str, float | int]:
        split = "train" if calculator is None else calculator.split
        return self.train.pool_metrics(self.factor_store, len(self.entries), self.weights, split)

    @staticmethod
    def _entry(item: PreparedCandidate) -> PoolEntry:
        return PoolEntry(item.expr, item.token, item.single_ics, item.coverages, item.ic_series)

    def _copy_factor(self, index: int, factors: dict[str, Tensor]) -> None:
        for name, value in factors.items():
            self.factor_store[name][index].copy_(value)

    def _commit(self, item: PreparedCandidate) -> float:
        token, single_ic = item.token, item.single_ics["train"]
        count = self.counts.get(token, 0)
        self.counts[token] = count + 1
        novelty = self.novelty_coef / (1 + count) if np.isfinite(single_ic) else 0.0
        old_ic = self._current_ic
        accepted = np.isfinite(single_ic)
        n = len(self.entries)
        mutual = self.train.mutual_ics(item.factors["train"], self.factor_store["train"], n)
        if n and (not np.isfinite(mutual).all() or np.max(np.abs(mutual)) >= self.mutual_threshold):
            accepted = False
        attempted_ic = old_ic
        if accepted and n < self.capacity:
            matrix = np.eye(n + 1)
            matrix[:n, :n] = self.corr[:n, :n]
            matrix[n, :n] = matrix[:n, n] = mutual
            proposed = [*self.entries, self._entry(item)]
            weights = self._optimize(np.array([entry.single_ic for entry in proposed]), matrix)
            self._copy_factor(n, item.factors)
            attempted_ic = float(self.train.pool_metrics(
                self.factor_store, n + 1, weights, "train"
            )["ic"])
            if np.isfinite(attempted_ic):
                self.entries, self.weights, self._current_ic = proposed, weights, attempted_ic
                self.corr[:n + 1, :n + 1] = matrix
            else:
                accepted = False
                attempted_ic = old_ic
        elif accepted:
            extended = np.eye(n + 1)
            extended[:n, :n] = self.corr[:n, :n]
            extended[n, :n] = extended[:n, n] = mutual
            extended_entries = [*self.entries, self._entry(item)]
            extended_weights = self._optimize(
                np.array([entry.single_ic for entry in extended_entries]), extended
            )
            removable = np.arange(n) if self.eviction == "counterfactual" else np.arange(n + 1)
            remove = int(removable[np.argmin(np.abs(extended_weights[removable]))])
            if remove == n:
                accepted = False
            else:
                matrix = self.corr[:n, :n].copy()
                matrix[remove, :] = mutual
                matrix[:, remove] = mutual
                matrix[remove, remove] = 1.0
                proposed = self.entries.copy()
                proposed[remove] = self._entry(item)
                weights = self._optimize(np.array([entry.single_ic for entry in proposed]), matrix)
                attempted_ic = float(self.train.pool_metrics(
                    self.factor_store, n, weights, "train", (remove, item.factors["train"])
                )["ic"])
                rollback = self.eviction == "counterfactual" and attempted_ic < old_ic
                if np.isfinite(attempted_ic) and not rollback:
                    self._copy_factor(remove, item.factors)
                    self.entries, self.weights, self._current_ic = proposed, weights, attempted_ic
                    self.corr[:n, :n] = matrix
                else:
                    accepted = False
                    if not np.isfinite(attempted_ic):
                        attempted_ic = old_ic
        reward = attempted_ic - old_ic + novelty
        self.events.append(CandidateEvent(
            token, float(single_ic), reward, item.expr.depth, len(token.split()), count > 0, accepted
        ))
        return reward

    def try_candidates_batch(self, candidates: list[tuple[Expr, str]]) -> list[float]:
        started = time.perf_counter()
        if self.factor_workers <= 1 or len(candidates) <= 1:
            prepared = [self.train.prepare(*item) for item in candidates]
        else:
            assert self._executor is not None
            prepared = list(self._executor.map(lambda item: self.train.prepare(*item), candidates))
        rewards = [self._commit(item) for item in prepared]
        self._eval_candidates += len(candidates)
        self._eval_seconds += time.perf_counter() - started
        return rewards

    def try_candidate(self, expr: Expr, token: str) -> float:
        return self.try_candidates_batch([(expr, token)])[0]

    def drain_events(self) -> list[CandidateEvent]:
        events, self.events = self.events, []
        return events

    def drain_eval_stats(self) -> tuple[int, float]:
        result = self._eval_candidates, self._eval_seconds
        self._eval_candidates, self._eval_seconds = 0, 0.0
        return result
