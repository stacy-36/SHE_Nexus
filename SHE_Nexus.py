from __future__ import annotations
import argparse
from io import StringIO
__version__ = '1.2.0'
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from typing import Mapping
from typing import Mapping, Sequence
from dataclasses import dataclass
from calendar import monthrange
import numpy as np
from collections import defaultdict, deque
from dataclasses import asdict
from hashlib import sha256

class ConfigurationError(ValueError):
    pass

def _positive(value: Any, name: str, *, allow_zero: bool=False) -> float:
    if value is None:
        raise ConfigurationError(f'{name} must be supplied; null is not allowed')
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f'{name} must be finite')
    if allow_zero and number < 0:
        raise ConfigurationError(f'{name} must be non-negative')
    if not allow_zero and number <= 0:
        raise ConfigurationError(f'{name} must be positive')
    return number

def _curve(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ConfigurationError(f'{name} must contain at least two [x, y] points')
    if any((not isinstance(row, (list, tuple)) or len(row) != 2 for row in value)):
        raise ConfigurationError(f'{name} points must contain exactly two values')
    points = tuple(((float(row[0]), float(row[1])) for row in value))
    if any((not math.isfinite(v) for row in points for v in row)):
        raise ConfigurationError(f'{name} points must be finite')
    if any((right[0] <= left[0] for left, right in zip(points, points[1:]))):
        raise ConfigurationError(f'{name} x coordinates must be strictly increasing')
    if any((right[1] < left[1] for left, right in zip(points, points[1:]))):
        raise ConfigurationError(f'{name} y coordinates must be non-decreasing')
    return points

@dataclass(frozen=True)
class ReservoirSpec:
    reservoir_id: str
    downstream: str | None
    dead_storage: float
    max_storage: float
    initial_storage: float
    max_release: float
    turbine_capacity: float
    efficiency: float
    installed_power_mw: float
    energy_target_mwh: float
    transfer_out_max: float
    transfer_in_max: float
    capacity_curve: tuple[tuple[float, float], ...]
    tailwater_curve: tuple[tuple[float, float], ...]
    monthly_storage: Mapping[int, tuple[float, float]]
    min_release: float = 0.0

    @classmethod
    def from_mapping(cls, reservoir_id: str, raw: Mapping[str, Any]) -> 'ReservoirSpec':
        monthly_raw = raw.get('monthly_storage')
        if not isinstance(monthly_raw, Mapping) or set(map(int, monthly_raw)) != set(range(1, 13)):
            raise ConfigurationError(f'reservoirs.{reservoir_id}.monthly_storage must define months 1-12')
        monthly: dict[int, tuple[float, float]] = {}
        for key, limits in monthly_raw.items():
            month = int(key)
            low = _positive(limits[0], f'{reservoir_id}.monthly_storage[{month}].min', allow_zero=True)
            high = _positive(limits[1], f'{reservoir_id}.monthly_storage[{month}].max')
            if low > high:
                raise ConfigurationError(f'{reservoir_id}: monthly minimum exceeds maximum')
            monthly[month] = (low, high)
        spec = cls(reservoir_id=reservoir_id, downstream=raw.get('downstream'), dead_storage=_positive(raw.get('dead_storage'), f'{reservoir_id}.dead_storage', allow_zero=True), max_storage=_positive(raw.get('max_storage'), f'{reservoir_id}.max_storage'), initial_storage=_positive(raw.get('initial_storage'), f'{reservoir_id}.initial_storage', allow_zero=True), max_release=_positive(raw.get('max_release'), f'{reservoir_id}.max_release'), turbine_capacity=_positive(raw.get('turbine_capacity'), f'{reservoir_id}.turbine_capacity'), efficiency=_positive(raw.get('efficiency'), f'{reservoir_id}.efficiency'), installed_power_mw=_positive(raw.get('installed_power_mw'), f'{reservoir_id}.installed_power_mw'), energy_target_mwh=_positive(raw.get('energy_target_mwh'), f'{reservoir_id}.energy_target_mwh'), transfer_out_max=_positive(raw.get('transfer_out_max', 0), f'{reservoir_id}.transfer_out_max', allow_zero=True), transfer_in_max=_positive(raw.get('transfer_in_max', 0), f'{reservoir_id}.transfer_in_max', allow_zero=True), capacity_curve=_curve(raw.get('capacity_curve'), f'{reservoir_id}.capacity_curve'), tailwater_curve=_curve(raw.get('tailwater_curve'), f'{reservoir_id}.tailwater_curve'), monthly_storage=monthly, min_release=_positive(raw.get('min_release', 0), f'{reservoir_id}.min_release', allow_zero=True))
        if not spec.dead_storage <= spec.initial_storage <= spec.max_storage:
            raise ConfigurationError(f'{reservoir_id}: initial storage is outside physical bounds')
        if spec.min_release > spec.max_release:
            raise ConfigurationError(f'{reservoir_id}: minimum release exceeds maximum')
        if spec.efficiency > 1:
            raise ConfigurationError(f'{reservoir_id}: efficiency must be expressed as a fraction <= 1')
        if spec.dead_storage >= spec.max_storage:
            raise ConfigurationError(f'{reservoir_id}: dead storage must be below maximum storage')
        if any((low < spec.dead_storage or high > spec.max_storage for low, high in monthly.values())):
            raise ConfigurationError(f'{reservoir_id}: monthly storage bounds exceed physical bounds')
        if any((b[1] <= a[1] for a, b in zip(spec.capacity_curve, spec.capacity_curve[1:]))):
            raise ConfigurationError(f'{reservoir_id}: capacity-curve storage must strictly increase')
        if spec.capacity_curve[0][1] > spec.dead_storage or spec.capacity_curve[-1][1] < spec.max_storage:
            raise ConfigurationError(f'{reservoir_id}: capacity curve must cover the storage bounds')
        upper = hydropower_energy_upper_bound(spec)
        if spec.energy_target_mwh > upper + 1e-09:
            raise ConfigurationError(f'{reservoir_id}: energy target {spec.energy_target_mwh:g} MWh exceeds the conservative monthly physical upper bound {upper:g} MWh')
        return spec

@dataclass(frozen=True)
class ModelConfig:
    reservoirs: tuple[ReservoirSpec, ...]
    initial_shape_factors: Mapping[str, float]
    satisfaction_thresholds: Mapping[str, float]
    feedback: Mapping[str, float]
    ecological_membership: Mapping[str, Any] = field(default_factory=lambda: {'method': 'continuous_quartiles'})

def hydropower_energy_upper_bound(spec: ReservoirSpec, days: int=28) -> float:
    head_bound = max(max((p[0] for p in spec.capacity_curve)) - min((p[1] for p in spec.tailwater_curve)), 0.0)
    volume_m3 = min(spec.turbine_capacity, spec.max_release) * 1000000.0
    hydraulic_mwh = 1000.0 * 9.81 * spec.efficiency * volume_m3 * head_bound / 3600000000.0
    return min(hydraulic_mwh, spec.installed_power_mw * 24.0 * days)

def load_config(path: str | Path) -> ModelConfig:
    raw = dict(path) if isinstance(path, Mapping) else json.loads(Path(path).read_text(encoding='utf-8'))
    reservoir_raw = raw.get('reservoirs')
    if not isinstance(reservoir_raw, Mapping) or not reservoir_raw:
        raise ConfigurationError('reservoirs must be a non-empty object')
    specs = tuple((ReservoirSpec.from_mapping(key, value) for key, value in reservoir_raw.items()))
    ids = {spec.reservoir_id for spec in specs}
    for spec in specs:
        if spec.downstream is not None and spec.downstream not in ids:
            raise ConfigurationError(f'{spec.reservoir_id}: unknown downstream reservoir {spec.downstream}')
    factors = raw.get('initial_shape_factors', {})
    thresholds = raw.get('satisfaction_thresholds', {})
    required_factors = {'water_supply', 'hydropower', 'environment'}
    if set(factors) != required_factors:
        raise ConfigurationError(f'initial_shape_factors must contain {sorted(required_factors)}')
    required_thresholds = {'water_supply', 'hydropower', 'ecological_flow', 'water_quality_grade'}
    if set(thresholds) != required_thresholds:
        raise ConfigurationError(f'satisfaction_thresholds must contain {sorted(required_thresholds)}')
    factor_values = {key: _positive(value, f'initial_shape_factors.{key}') for key, value in factors.items()}
    total = sum(factor_values.values())
    factor_values = {key: value / total for key, value in factor_values.items()}
    threshold_values = {key: float(value) for key, value in thresholds.items()}
    if any((not math.isfinite(v) for v in threshold_values.values())):
        raise ConfigurationError('satisfaction thresholds must be finite')
    if any((not 0 <= threshold_values[k] <= 1 for k in ('water_supply', 'hydropower', 'ecological_flow'))):
        raise ConfigurationError('satisfaction ratio thresholds must be in [0, 1]')
    membership = raw.get('ecological_membership')
    if not isinstance(membership, Mapping):
        raise ConfigurationError('V1.2.0 requires an explicit ecological_membership definition')
    if membership != {'method': 'continuous_quartiles'}:
        raise ConfigurationError('V1.2.0 requires the continuous_quartiles ecological rule')
    return ModelConfig(reservoirs=specs, initial_shape_factors=factor_values, satisfaction_thresholds=threshold_values, feedback={key: float(value) for key, value in raw.get('feedback', {}).items()}, ecological_membership=membership)

def _validated_mapping(mapping: pd.DataFrame, *, source_key: str, target_key: str) -> pd.DataFrame:
    required = {source_key, target_key, 'weight'}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f'mapping is missing {sorted(missing)}')
    frame = mapping[[source_key, target_key, 'weight']].copy()
    if frame[[source_key, target_key]].isna().any().any():
        raise ValueError('mapping identifiers cannot be empty')
    if frame.duplicated([source_key, target_key]).any():
        raise ValueError('mapping contains duplicate source-target pairs')
    frame['weight'] = frame['weight'].astype(float)
    if (frame['weight'] < 0.0).any():
        raise ValueError('mapping weights must be non-negative')
    totals = frame.groupby(source_key)['weight'].sum()
    invalid = totals[~totals.map(lambda value: math.isclose(value, 1.0, abs_tol=1e-09))]
    if not invalid.empty:
        raise ValueError(f'mapping weights must sum to one for each {source_key}')
    return frame

def allocate_series(values: pd.Series, mapping: pd.DataFrame, *, source_key: str, target_key: str) -> pd.Series:
    frame = _validated_mapping(mapping, source_key=source_key, target_key=target_key)
    sources = set(frame[source_key])
    if sources != set(values.index):
        raise ValueError(f'mapping and values must define identical {source_key} identifiers')
    joined = frame.merge(values.astype(float).rename('source_value'), left_on=source_key, right_index=True, how='left', validate='many_to_one')
    joined['allocated'] = joined['source_value'] * joined['weight']
    result = joined.groupby(target_key)['allocated'].sum().sort_index()
    if not math.isclose(float(result.sum()), float(values.sum()), rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError('allocation did not conserve the source total')
    result.name = values.name
    return result

def allocate_frame(values: pd.DataFrame, mapping: pd.DataFrame, *, source_key: str, target_key: str) -> pd.DataFrame:
    columns = list(values.columns)
    if not columns:
        raise ValueError('values must contain at least one quantity column')
    allocated = {column: allocate_series(values[column].rename(column), mapping, source_key=source_key, target_key=target_key) for column in columns}
    return pd.DataFrame(allocated).sort_index()

def _response(stress: float, maximum: float, steepness: float) -> float:
    if not all((math.isfinite(v) for v in (stress, maximum, steepness))) or maximum < 0 or steepness <= 0:
        raise ValueError('maximum must be non-negative and steepness must be positive')
    positive = max(float(stress), 0.0)
    raw = 2.0 / (1.0 + math.exp(-steepness * positive)) - 1.0
    return maximum * raw

def performance_deficits(performance: Mapping[str, float], thresholds: Mapping[str, float]) -> dict[str, float]:
    keys = ('water_supply', 'hydropower', 'ecological_flow', 'water_quality_grade')
    if any((not math.isfinite(float(performance[k])) or not math.isfinite(float(thresholds[k])) for k in keys)):
        raise ValueError('performance and thresholds must be finite')
    result = {'water_supply': max(thresholds['water_supply'] - performance['water_supply'], 0.0), 'hydropower': max(thresholds['hydropower'] - performance['hydropower'], 0.0), 'ecological_flow': max(thresholds['ecological_flow'] - performance['ecological_flow'], 0.0), 'water_quality': max(performance['water_quality_grade'] - thresholds['water_quality_grade'], 0.0)}
    result['environment'] = max(result['ecological_flow'], result['water_quality'])
    return result

def feedback_diagnostics(performance, base_factors, thresholds, *, maximum_adjustment=0.2, steepness=5.0, enabled=True):
    deficits = performance_deficits(performance, thresholds)
    result = {f'deficit_{name}': value for name, value in deficits.items()}
    result['shape_update_enabled'] = int(enabled)
    for name in ('water_supply', 'hydropower', 'environment'):
        adjustment = _response(deficits[name], maximum_adjustment, steepness) if enabled else 0.0
        result[f'adjustment_{name}'] = adjustment
        result[f'shape_raw_{name}'] = base_factors[name] * (1.0 + adjustment)
    return result

def update_shape_factors(performance: Mapping[str, float], base_factors: Mapping[str, float], thresholds: Mapping[str, float], *, maximum_adjustment: float=0.2, steepness: float=5.0, normalize: bool=True) -> dict[str, float]:
    required_base = {'water_supply', 'hydropower', 'environment'}
    if set(base_factors) != required_base:
        raise ValueError(f'base_factors must contain {sorted(required_base)}')
    if any((not math.isfinite(float(v)) or float(v) <= 0 for v in base_factors.values())):
        raise ValueError('base factors must be finite and positive')
    required_performance = {'water_supply', 'hydropower', 'ecological_flow', 'water_quality_grade'}
    missing = required_performance - set(performance)
    if missing:
        raise ValueError(f'performance is missing {sorted(missing)}')
    deficits = performance_deficits(performance, thresholds)
    supply_stress = deficits['water_supply']
    power_stress = deficits['hydropower']
    environment_stress = deficits['environment']
    factors = {'water_supply': base_factors['water_supply'] * (1.0 + _response(supply_stress, maximum_adjustment, steepness)), 'hydropower': base_factors['hydropower'] * (1.0 + _response(power_stress, maximum_adjustment, steepness)), 'environment': base_factors['environment'] * (1.0 + _response(environment_stress, maximum_adjustment, steepness))}
    if normalize:
        total = sum(factors.values())
        factors = {key: value / total for key, value in factors.items()}
    return factors

def assign_period_factors(periods: list[tuple[int, int]], period_performance: Mapping[tuple[int, int], Mapping[str, float]], initial_factors: Mapping[str, float], thresholds: Mapping[str, float], **kwargs) -> dict[tuple[int, int], dict[str, float]]:
    if not periods:
        return {}
    result = {periods[0]: dict(initial_factors)}
    for previous, current in zip(periods, periods[1:]):
        if previous not in period_performance:
            raise KeyError(f'missing performance for previous period {previous}')
        result[current] = update_shape_factors(period_performance[previous], result[previous], thresholds, **kwargs)
    return result
DEMAND_BASES = {'monthly_mean': 1.0, 'annual_total': 1.0 / 12.0}
FEEDBACK_MODES = {'full', 'none', 'sd_only', 'weights_only'}

def demand_scale(basis: str) -> float:
    if basis not in DEMAND_BASES:
        raise ValueError('demand_basis must be monthly_mean or annual_total')
    return DEMAND_BASES[basis]

def validate_seasonality(values):
    factors = {int(k): float(v) for k, v in values.items()}
    if set(factors) != set(range(1, 13)):
        raise ValueError('demand_seasonality must define months 1-12')
    if any((not math.isfinite(v) or v < 0 for v in factors.values())):
        raise ValueError('demand seasonality must be finite and non-negative')
    if not math.isclose(sum(factors.values()), 12.0, abs_tol=1e-09, rel_tol=0):
        raise ValueError('demand seasonality must have a mean of one')
    return factors

def quality_schedule(raw, name, periods, network):
    if name not in raw['scenarios']:
        raise ValueError(f'unknown quality scenario {name!r}')
    schedule = raw['scenarios'][name]
    labels = {p[0] for p in periods}
    if set(schedule) != labels:
        raise ValueError('quality scenario must cover each configured period exactly once')
    resolved = {}
    for label, condition in schedule.items():
        if condition not in raw['conditions']:
            raise ValueError(f'unknown quality boundary condition {condition!r}')
        boundaries = raw['conditions'][condition]
        for node, values in boundaries.items():
            if node not in network.nodes or network.upstream[node]:
                raise ValueError(f'{node}: boundary must identify a headwater node')
            if set(values) != {'COD', 'NH3N', 'TP'}:
                raise ValueError('boundary must define COD, NH3N, TP')
            if any((not math.isfinite(float(v)) or float(v) < 0 for v in values.values())):
                raise ValueError('boundary concentrations must be finite and non-negative')
        resolved[label] = boundaries
    return resolved
GRADE_SCORE = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'Above V': 5}

def water_quality_score(concentrations: pd.DataFrame, *, keys: Sequence[str]=('year', 'month', 'node')) -> pd.DataFrame:
    required = set(keys) | {'grade'}
    if not required <= set(concentrations.columns):
        raise ValueError(f'water-quality results must contain {sorted(required)}')
    frame = concentrations.copy()
    frame['water_quality_grade'] = frame['grade'].map(GRADE_SCORE)
    if frame['water_quality_grade'].isna().any():
        raise ValueError('unknown water-quality grade')
    return frame.groupby(list(keys), as_index=False)['water_quality_grade'].max()

def aggregate_by_mapping(values: pd.DataFrame, mapping: pd.DataFrame, *, source_key: str, target_key: str='region', metrics: Sequence[str], time_keys: Sequence[str]=('year', 'month')) -> pd.DataFrame:
    if not {source_key, target_key} <= set(mapping.columns):
        raise ValueError('mapping table is missing source/target columns')
    required = set(time_keys) | {source_key} | set(metrics)
    if not required <= set(values.columns):
        raise ValueError(f'values are missing {sorted(required - set(values.columns))}')
    joined = values.merge(mapping[[source_key, target_key]], on=source_key, how='inner', validate='many_to_many')
    return joined.groupby([*time_keys, target_key], as_index=False)[list(metrics)].mean()

def merge_regional_performance(reservoir_scores: pd.DataFrame, quality_scores: pd.DataFrame, *, keys: Sequence[str]=('year', 'month', 'region')) -> pd.DataFrame:
    return reservoir_scores.merge(quality_scores, on=list(keys), how='inner', validate='one_to_one')

def period_summary(regional: pd.DataFrame, *, start_year: int, end_year: int, metrics: Sequence[str]=('water_scarcity', 'ecological_flow', 'hydropower', 'water_quality_grade')) -> pd.DataFrame:
    selected = regional[regional['year'].between(start_year, end_year)].copy()
    if selected.empty:
        raise ValueError('no records fall inside the requested period')
    missing = set(metrics) - set(selected.columns)
    if missing:
        raise ValueError(f'regional table is missing {sorted(missing)}')
    grouped = selected.groupby('region', as_index=False)[list(metrics)].mean()
    grouped['water_supply'] = 1.0 - grouped['water_scarcity'].clip(0, 1)
    grouped['water_quality_satisfaction'] = 1.0 - ((grouped['water_quality_grade'] - 1.0) / 4.0).clip(0, 1)
    if 'synergy_volume' in selected:
        volume = selected.groupby('region')['synergy_volume'].mean()
        grouped['synergy_volume'] = grouped['region'].map(volume)
    grouped.insert(0, 'period', f'{start_year}-{end_year}')
    return grouped

@dataclass(frozen=True)
class PSOSettings:
    particles: int = 80
    generations: int = 80
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    max_velocity_fraction: float = 0.2
    penalty_scale: float = 1000.0
    seed: int = 42

@dataclass
class MonthEvaluation:
    objective: float
    synergy_volume: float
    records: list[dict[str, float | str | int]]
    next_storage: dict[str, float]
    violations: dict[str, float]

    @property
    def feasible(self) -> bool:
        return sum(self.violations.values()) <= 1e-08

def ecological_satisfaction(release: float, thresholds: Sequence[float], method: str='continuous_quartiles') -> float:
    q = float(release)
    x = np.asarray(thresholds, dtype=float)
    if method != 'continuous_quartiles':
        raise ValueError('unknown ecological membership method')
    if x.shape != (4,):
        raise ValueError('exactly four ecological thresholds are required')
    if not np.isfinite(q) or not np.all(np.isfinite(x)) or x[0] <= 0:
        raise ValueError('invalid ecological membership inputs')
    if np.any(np.diff(x) <= 0):
        raise ValueError('ecological thresholds must be strictly increasing')
    if q > x[-1]:
        return 1.0
    return float(np.interp(q, np.r_[x[0] / 2, x], [0, 0.25, 0.5, 0.75, 1]))

def _interp(curve: Sequence[tuple[float, float]], x_value: float) -> float:
    points = np.asarray(curve, dtype=float)
    order = np.argsort(points[:, 0])
    return float(np.interp(x_value, points[order, 0], points[order, 1]))

def _synergy_volume(supply: float, power: float, environment: float, donation: float, factors: Mapping[str, float]) -> float:
    s = factors['water_supply'] * supply
    h = factors['hydropower'] * power
    e = factors['environment'] * environment
    base_area = math.sqrt(3.0) / 4.0 * (s * h + s * e + h * e)
    return base_area * donation / 3.0

class ReservoirSystem:

    def __init__(self, specs: Sequence[ReservoirSpec], ecological_membership: Mapping | None=None):
        self.specs = {spec.reservoir_id: spec for spec in specs}
        if not self.specs:
            raise ValueError('at least one reservoir is required')
        self.order = self._topological_order()
        self.ecological_membership = dict(ecological_membership or {})

    def _topological_order(self) -> list[str]:
        indegree = {key: 0 for key in self.specs}
        children = {key: [] for key in self.specs}
        for spec in self.specs.values():
            if spec.downstream:
                children[spec.reservoir_id].append(spec.downstream)
                indegree[spec.downstream] += 1
        queue = [key for key, value in indegree.items() if value == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.specs):
            raise ValueError('reservoir network contains a cycle')
        return order

    def initial_storage(self) -> dict[str, float]:
        return {key: spec.initial_storage for key, spec in self.specs.items()}

    def bounds(self, demand: Mapping[str, float], month: int, eco_thresholds: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
        lower: list[float] = []
        upper: list[float] = []
        for key in self.order:
            spec = self.specs[key]
            lower.extend((0.0, spec.min_release, 0.0, 0.0))
            upper.extend((max(float(demand[key]), 0.0), spec.max_release, spec.transfer_out_max, spec.transfer_in_max))
        return (np.asarray(lower), np.asarray(upper))

    def evaluate(self, decision: Sequence[float], *, year: int, month: int, storage: Mapping[str, float], natural_inflow: Mapping[str, float], demand: Mapping[str, float], eco_thresholds: Mapping[str, Sequence[float]], shape_factors: Mapping[str, float], penalty_scale: float=1000.0, water_losses: Mapping[str, float] | None=None) -> MonthEvaluation:
        vector = np.asarray(decision, dtype=float)
        expected = 4 * len(self.order)
        if vector.size != expected:
            raise ValueError(f'decision vector has {vector.size} values; expected {expected}')
        variables = vector.reshape(len(self.order), 4)
        next_storage: dict[str, float] = {}
        downstream_release: dict[str, float] = {}
        records: list[dict[str, float | str | int]] = []
        violations: dict[str, float] = {}
        days = monthrange(int(year), int(month))[1]
        seconds = days * 86400.0
        for row_index, key in enumerate(self.order):
            spec = self.specs[key]
            supply, release, transfer_out, transfer_in = variables[row_index]
            upstream = sum((downstream_release[parent.reservoir_id] for parent in self.specs.values() if parent.downstream == key))
            total_inflow = float(natural_inflow[key]) + upstream + transfer_in
            previous = float(storage[key])
            loss = float((water_losses or {}).get(key, 0.0))
            if not math.isfinite(loss) or loss < 0:
                raise ValueError('water losses must be finite non-negative monthly volumes')
            new_storage = previous + total_inflow - supply - release - transfer_out - loss
            turbine = min(release, spec.turbine_capacity)
            spill = max(release - turbine, 0.0)
            downstream_release[key] = release
            next_storage[key] = new_storage
            low, high = spec.monthly_storage[int(month)]
            span = max(spec.max_storage - spec.dead_storage, 1e-09)
            violations[f'{key}.storage_below_min'] = max(low - new_storage, 0.0) / span
            violations[f'{key}.storage_above_max'] = max(new_storage - high, 0.0) / span
            violations[f'{key}.release_below_min'] = max(spec.min_release - release, 0.0) / spec.max_release
            average_storage = (previous + new_storage) / 2.0
            storage_level_curve = tuple(((storage_value, level) for level, storage_value in spec.capacity_curve))
            upstream_level = _interp(storage_level_curve, average_storage)
            q_total = release * 1000000.0 / seconds
            tailwater_level = _interp(spec.tailwater_curve, q_total)
            head = max(upstream_level - tailwater_level, 0.0)
            q_turbine = turbine * 1000000.0 / seconds
            power_mw = 1000.0 * 9.81 * spec.efficiency * q_turbine * head / 1000000.0
            energy_mwh = power_mw * 24.0 * days
            violations[f'{key}.installed_power'] = max(power_mw - spec.installed_power_mw, 0.0) / spec.installed_power_mw
            supply_ratio = min(supply / float(demand[key]), 1.0) if demand[key] > 0 else 1.0
            power_ratio = min(energy_mwh / spec.energy_target_mwh, 1.0)
            eco_ratio = ecological_satisfaction(release, eco_thresholds[key], **self.ecological_membership)
            donation_ratio = min(transfer_out / spec.transfer_out_max, 1.0) if spec.transfer_out_max > 0 else math.nan
            receiving_ratio = min(transfer_in / spec.transfer_in_max, 1.0) if spec.transfer_in_max > 0 else math.nan
            records.append({'year': int(year), 'month': int(month), 'reservoir': key, 'storage_start': previous, 'natural_inflow': float(natural_inflow[key]), 'upstream_release': upstream, 'transfer_in': transfer_in, 'water_loss': loss, 'demand': float(demand[key]), 'supply': supply, 'release': release, 'turbine_release': turbine, 'spill': spill, 'transfer_out': transfer_out, 'net_external_transfer': transfer_in - transfer_out, 'storage_end': new_storage, 'hydropower_mwh': energy_mwh, 'supply_ratio': supply_ratio, 'power_ratio': power_ratio, 'eco_ratio': eco_ratio, 'donation_ratio': donation_ratio, 'receiving_ratio': receiving_ratio})
        supply_avg = float(np.mean([r['supply_ratio'] for r in records]))
        power_avg = float(np.mean([r['power_ratio'] for r in records]))
        eco_avg = float(np.mean([r['eco_ratio'] for r in records]))
        total_donation_capacity = sum(spec.transfer_out_max for spec in self.specs.values())
        donation_avg = sum(r['transfer_out'] for r in records) / total_donation_capacity if total_donation_capacity > 0 else 1.0
        volume = _synergy_volume(supply_avg, power_avg, eco_avg, donation_avg, shape_factors)
        for record in records:
            record['system_donation_ratio'] = donation_avg
        total_violation = float(sum(violations.values()))
        objective = -volume + penalty_scale * total_violation
        return MonthEvaluation(objective, volume, records, next_storage, violations)

def particle_swarm(objective, lower: np.ndarray, upper: np.ndarray, settings: PSOSettings) -> tuple[np.ndarray, float]:
    if settings.particles < 2 or settings.generations < 1:
        raise ValueError('PSO requires at least two particles and one generation')
    rng = np.random.default_rng(settings.seed)
    span = upper - lower
    if np.any(span < 0) or not np.all(np.isfinite(span)):
        raise ValueError('invalid optimization bounds')
    positions = lower + rng.random((settings.particles, lower.size)) * span
    velocities = rng.uniform(-1.0, 1.0, positions.shape) * span * settings.max_velocity_fraction
    scores = np.asarray([objective(row) for row in positions], dtype=float)
    personal_positions = positions.copy()
    personal_scores = scores.copy()
    best_index = int(np.argmin(scores))
    global_position = positions[best_index].copy()
    global_score = float(scores[best_index])
    max_velocity = settings.max_velocity_fraction * span
    for _ in range(settings.generations):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = settings.inertia * velocities + settings.cognitive * r1 * (personal_positions - positions) + settings.social * r2 * (global_position - positions)
        velocities = np.clip(velocities, -max_velocity, max_velocity)
        positions = np.clip(positions + velocities, lower, upper)
        scores = np.asarray([objective(row) for row in positions], dtype=float)
        improved = scores < personal_scores
        personal_positions[improved] = positions[improved]
        personal_scores[improved] = scores[improved]
        best_index = int(np.argmin(personal_scores))
        if personal_scores[best_index] < global_score:
            global_position = personal_positions[best_index].copy()
            global_score = float(personal_scores[best_index])
    return (global_position, global_score)

def optimize_month(system: ReservoirSystem, *, year: int, month: int, storage: Mapping[str, float], natural_inflow: Mapping[str, float], demand: Mapping[str, float], eco_thresholds: Mapping[str, Sequence[float]], shape_factors: Mapping[str, float], settings: PSOSettings | None=None, water_losses: Mapping[str, float] | None=None) -> MonthEvaluation:
    cfg = settings or PSOSettings()
    lower, upper = system.bounds(demand, month, eco_thresholds)

    def feasible_allocation(vector):
        values = np.clip(np.asarray(vector, dtype=float), lower, upper).reshape(-1, 4).copy()
        released = {}
        for i, key in enumerate(system.order):
            spec = system.specs[key]
            upstream = sum(released[p.reservoir_id] for p in system.specs.values() if p.downstream == key)
            available = storage[key] + natural_inflow[key] + upstream + values[i, 3] - (water_losses or {}).get(key, 0.0)
            low, high = spec.monthly_storage[month]
            minimum = lower[4 * i + 1]
            outflow = values[i, :3].sum()
            shortage = max(outflow - (available - low), 0.0)
            flexible = values[i, :3] - np.array([0.0, minimum, 0.0])
            total = flexible.sum()
            if total > 0:
                values[i, :3] -= flexible * min(shortage / total, 1.0)
            excess = max(available - values[i, :3].sum() - high, 0.0)
            for j in (1, 0, 2):
                increment = min(excess, max(upper[4 * i + j] - values[i, j], 0.0))
                values[i, j] += increment
                excess -= increment
            released[key] = values[i, 1]
        return values.ravel()

    def score(vector: np.ndarray) -> float:
        return system.evaluate(feasible_allocation(vector), year=year, month=month, storage=storage, natural_inflow=natural_inflow, demand=demand, eco_thresholds=eco_thresholds, shape_factors=shape_factors, penalty_scale=cfg.penalty_scale, water_losses=water_losses).objective
    decision, _ = particle_swarm(score, lower, upper, cfg)
    return system.evaluate(feasible_allocation(decision), year=year, month=month, storage=storage, natural_inflow=natural_inflow, demand=demand, eco_thresholds=eco_thresholds, shape_factors=shape_factors, penalty_scale=cfg.penalty_scale, water_losses=water_losses)
SECTORS = ('population', 'industrial_added_value', 'irrigation_area', 'cattle', 'pigs', 'poultry')
POLLUTANT_SOURCE_SECTORS = {'point': ('population', 'industrial_added_value'), 'nonpoint': ('irrigation_area', 'cattle', 'pigs', 'poultry')}
CONSTRAINTS = ('water_scarcity', 'eco_dissatisfaction', 'water_quality')

@dataclass(frozen=True)
class SDParameters:
    thresholds: Mapping[str, float]
    pressure_steepness: Mapping[str, float]
    sector_sensitivity: Mapping[str, Mapping[str, float]]
    suppression_k: float
    gdp_impact: Mapping[str, float]

    def validate(self) -> None:
        if self.suppression_k <= 0:
            raise ValueError('suppression_k must be positive')
        for key in CONSTRAINTS:
            if key not in self.thresholds or key not in self.pressure_steepness:
                raise ValueError(f'missing SD constraint parameter: {key}')
        for sector in SECTORS:
            if sector not in self.sector_sensitivity:
                raise ValueError(f'missing sensitivity coefficients for {sector}')
            if set(self.sector_sensitivity[sector]) != set(CONSTRAINTS):
                raise ValueError(f'{sector} sensitivities must define {CONSTRAINTS}')

def constraint_pressure(value: float, threshold: float, steepness: float) -> float:
    if steepness <= 0:
        raise ValueError('pressure steepness must be positive')
    exceedance = max(float(value) - float(threshold), 0.0)
    return math.expm1(steepness * exceedance)

def growth_suppression(total_penalty: float, k: float) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    return 1.0 - math.exp(-k * max(float(total_penalty), 0.0))

def project_state(state: pd.DataFrame, natural_growth: pd.DataFrame, performance: Mapping[str, float], parameters: SDParameters, *, regional_gdp: Mapping[str, float], national_gdp: float, dt_years: float=1.0 / 12.0, external_drivers: pd.DataFrame | None=None) -> pd.DataFrame:
    parameters.validate()
    missing = set(SECTORS) - set(state.columns)
    if missing or set(SECTORS) - set(natural_growth.columns):
        raise ValueError(f'state/growth tables are missing sector columns: {sorted(missing)}')
    if not state.index.equals(natural_growth.index):
        raise ValueError('state and natural_growth must have identical region indexes')
    constraint_values = {'water_scarcity': 1.0 - float(performance['water_supply']), 'eco_dissatisfaction': 1.0 - float(performance['ecological_flow']), 'water_quality': float(performance['water_quality_grade'])}
    pressures = {key: constraint_pressure(constraint_values[key], parameters.thresholds[key], parameters.pressure_steepness[key]) for key in CONSTRAINTS}
    result = state.astype(float).copy()
    if not math.isfinite(national_gdp) or national_gdp <= 0 or not math.isfinite(dt_years) or dt_years <= 0:
        raise ValueError('national GDP and integration time step must be positive and finite')
    if set(regional_gdp) != set(state.index) or any(not math.isfinite(float(v)) or float(v) < 0 for v in regional_gdp.values()):
        raise ValueError('regional GDP must define every region with finite nonnegative values')
    forcing = pd.DataFrame(0.0, index=state.index, columns=SECTORS) if external_drivers is None else external_drivers
    if not forcing.index.equals(state.index) or not set(SECTORS) <= set(forcing.columns):
        raise ValueError('external drivers must define every region and sector')
    for region in result.index:
        for sector in SECTORS:
            penalty = sum((pressures[key] * parameters.sector_sensitivity[sector][key] for key in CONSTRAINTS))
            suppression = growth_suppression(penalty, parameters.suppression_k)
            gap = (float(regional_gdp[region]) - national_gdp) / national_gdp
            gdp_factor = 1.0 + parameters.gdp_impact[sector] * gap
            growth = float(natural_growth.loc[region, sector]) * (1.0 - suppression) * gdp_factor
            driver = float(forcing.loc[region, sector])
            integrated_driver = dt_years if abs(growth) < 1e-14 else math.expm1(growth * dt_years) / growth
            updated = float(result.loc[region, sector]) * math.exp(growth * dt_years) + driver * integrated_driver
            if not math.isfinite(updated) or updated < 0:
                raise ValueError('SD equation produced invalid state; check growth and external drivers')
            result.loc[region, sector] = updated
    return result

def calculate_water_demand(state: pd.DataFrame, quotas: pd.DataFrame) -> pd.Series:
    common = [column for column in SECTORS if column in quotas.columns]
    if not common or not state.index.equals(quotas.index):
        raise ValueError('quotas must share regions and at least one sector with state')
    return (state[common].astype(float) * quotas[common].astype(float)).sum(axis=1)

def calculate_pollutant_load(state: pd.DataFrame, generation_coefficients: Mapping[str, Mapping[str, float]], removal_rates: Mapping[str, float]) -> pd.DataFrame:
    by_source = calculate_pollutant_load_by_source(state, generation_coefficients, removal_rates)
    return by_source['point'].add(by_source['nonpoint'], fill_value=0.0)

def calculate_pollutant_load_by_source(state: pd.DataFrame, generation_coefficients: Mapping[str, Mapping[str, float]], removal_rates: Mapping[str, float]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, dict[str, pd.Series]] = {source: {} for source in POLLUTANT_SOURCE_SECTORS}
    for pollutant, coefficients in generation_coefficients.items():
        for sector, coefficient in coefficients.items():
            if sector not in state:
                raise ValueError(f'unknown sector {sector!r} for pollutant {pollutant}')
        rates = removal_rates[pollutant]
        if not isinstance(rates, Mapping) or set(rates) != set(coefficients):
            raise ValueError('each pollutant requires a removal rate for every emission sector')
        if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in rates.values()):
            raise ValueError('removal rates must be finite fractions')
        for source, sectors in POLLUTANT_SOURCE_SECTORS.items():
            total = pd.Series(0.0, index=state.index)
            for sector in sectors:
                if sector in coefficients:
                    total = total + state[sector].astype(float) * float(coefficients[sector]) * (1.0 - float(rates[sector]))
            outputs[source][pollutant] = total
    return {source: pd.DataFrame(values) for source, values in outputs.items()}
POLLUTANTS = ('COD', 'NH3N', 'TP')
RIVER_STANDARDS = {'COD': ((15.0, 'I'), (20.0, 'III'), (30.0, 'IV'), (40.0, 'V'), (math.inf, 'Above V')), 'NH3N': ((0.15, 'I'), (0.5, 'II'), (1.0, 'III'), (1.5, 'IV'), (2.0, 'V'), (math.inf, 'Above V')), 'TP': ((0.02, 'I'), (0.1, 'II'), (0.2, 'III'), (0.3, 'IV'), (0.4, 'V'), (math.inf, 'Above V'))}
RESERVOIR_STANDARDS = {**RIVER_STANDARDS, 'TP': ((0.01, 'I'), (0.025, 'II'), (0.05, 'III'), (0.1, 'IV'), (0.2, 'V'), (math.inf, 'Above V'))}
STANDARDS = RIVER_STANDARDS

@dataclass(frozen=True)
class Reach:
    node: str
    downstream: str | None
    length_km: float
    width_m: float
    depth_m: float
    slope: float
    manning_n: float
    decay_per_day: Mapping[str, float]
    background_mg_l: Mapping[str, float]

@dataclass(frozen=True)
class ReservoirNode:
    node: str
    downstream: str | None
    volume_m3: float
    decay_per_day: Mapping[str, float]

def classify_water_quality(pollutant: str, concentration: float, *, waterbody: str='river') -> str:
    if pollutant not in POLLUTANTS:
        raise ValueError(f'unknown pollutant {pollutant!r}')
    value = float(concentration)
    if math.isnan(value) or value < 0.0:
        raise ValueError('water-quality concentration must be non-negative')
    if waterbody not in {'river', 'reservoir'}:
        raise ValueError("waterbody must be 'river' or 'reservoir'")
    standards = RESERVOIR_STANDARDS if waterbody == 'reservoir' else RIVER_STANDARDS
    for limit, grade in standards[pollutant]:
        if value <= limit:
            return grade
    return 'Above V'

def kg_month_to_mg_second(load_kg_month: float, year: int, month: int) -> float:
    seconds = monthrange(int(year), int(month))[1] * 86400.0
    return float(load_kg_month) * 1000000.0 / seconds

class WaterQualityNetwork:

    def __init__(self, reaches: Sequence[Reach], reservoirs: Sequence[ReservoirNode]=()):
        for item in reaches:
            if item.length_km <= 0:
                raise ValueError('river-reach lengths must be positive')
            if set(item.decay_per_day) != set(POLLUTANTS):
                raise ValueError(f'{item.node}: decay_per_day must define {POLLUTANTS}')
            if set(item.background_mg_l) != set(POLLUTANTS):
                raise ValueError(f'{item.node}: background_mg_l must define {POLLUTANTS}')
            if any((float(value) < 0.0 for value in item.decay_per_day.values())):
                raise ValueError(f'{item.node}: decay coefficients cannot be negative')
            if any((float(value) < 0.0 for value in item.background_mg_l.values())):
                raise ValueError(f'{item.node}: background concentrations cannot be negative')
        if any((item.volume_m3 <= 0 for item in reservoirs)):
            raise ValueError('reservoir water-quality volumes must be positive')
        for item in reservoirs:
            if set(item.decay_per_day) != set(POLLUTANTS):
                raise ValueError(f'{item.node}: decay_per_day must define {POLLUTANTS}')
            if any((float(value) < 0.0 for value in item.decay_per_day.values())):
                raise ValueError(f'{item.node}: decay coefficients cannot be negative')
        self.reaches = {item.node: item for item in reaches}
        self.reservoirs = {item.node: item for item in reservoirs}
        if set(self.reaches) & set(self.reservoirs):
            raise ValueError('a node cannot be both a reach and a reservoir')
        self.nodes = {**self.reaches, **self.reservoirs}
        self.upstream: dict[str, list[str]] = defaultdict(list)
        for node, item in self.nodes.items():
            if item.downstream is not None:
                if item.downstream not in self.nodes:
                    raise ValueError(f'{node}: unknown downstream node {item.downstream}')
                self.upstream[item.downstream].append(node)
        self.order = self._topological_order()
        self.state: dict[str, dict[str, float]] = defaultdict(dict)

    def _topological_order(self) -> list[str]:
        indegree = {node: len(self.upstream[node]) for node in self.nodes}
        queue = deque((node for node, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            downstream = self.nodes[node].downstream
            if downstream is not None:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    queue.append(downstream)
        if len(order) != len(self.nodes):
            raise ValueError('water-quality network contains a cycle')
        return order

    def _upstream_concentration(self, node: str, pollutant: str, flows_m3_s: Mapping[str, float]) -> float:
        parents = self.upstream[node]
        if not parents:
            return 0.0
        weights = [max(float(flows_m3_s.get(parent, 0.0)), 0.0) for parent in parents]
        total = sum(weights)
        if total <= 0:
            return 0.0
        return sum((flow * self.state[parent].get(pollutant, 0.0) for parent, flow in zip(parents, weights))) / total

    @staticmethod
    def _velocity(reach: Reach) -> float:
        if reach.width_m <= 0 or reach.depth_m <= 0 or reach.slope <= 0 or (reach.manning_n <= 0):
            raise ValueError(f'invalid hydraulic parameters for {reach.node}')
        area = reach.width_m * reach.depth_m
        radius = area / (reach.width_m + 2.0 * reach.depth_m)
        return 1.0 / reach.manning_n * radius ** (2.0 / 3.0) * math.sqrt(reach.slope)

    def simulate_month(self, *, year: int, month: int, flows_m3_s: Mapping[str, float], point_loads: pd.DataFrame, nonpoint_loads: pd.DataFrame, boundary_concentrations_mg_l: Mapping[str, Mapping[str, float]] | None=None, reservoir_volumes_m3: Mapping[str, float] | None=None, external_inflows: Mapping | None=None) -> pd.DataFrame:
        required_point = {'node', 'pollutant', 'position_km', 'load_kg_month'}
        required_nonpoint = {'node', 'pollutant', 'load_kg_month'}
        if not point_loads.empty and (not required_point <= set(point_loads.columns)):
            raise ValueError(f'point_loads must contain {sorted(required_point)}')
        if not nonpoint_loads.empty and (not required_nonpoint <= set(nonpoint_loads.columns)):
            raise ValueError(f'nonpoint_loads must contain {sorted(required_nonpoint)}')
        for label, frame in (('point', point_loads), ('non-point', nonpoint_loads)):
            if frame.empty:
                continue
            unknown = set(frame['pollutant']) - set(POLLUTANTS)
            if unknown:
                raise ValueError(f'unknown {label}-load pollutants: {sorted(unknown)}')
            if (frame['load_kg_month'].astype(float) < 0.0).any():
                raise ValueError(f'{label} loads cannot be negative')
            if not all((math.isfinite(float(v)) for v in frame['load_kg_month'])):
                raise ValueError(f'{label} loads must be finite')
            if set(frame['node']) - set(self.reaches):
                raise ValueError('local loads must map to river reaches; direct reservoir loads are not implemented')
        days = monthrange(int(year), int(month))[1]
        seconds = days * 86400.0
        boundaries = boundary_concentrations_mg_l or {}
        results: list[dict[str, float | str | int]] = []
        for node in self.order:
            flow = float(flows_m3_s.get(node, 0.0))
            if flow < 0.0:
                raise ValueError(f'{node}: flow cannot be negative')
            for pollutant in POLLUTANTS:
                previous = math.nan
                mixing_fraction = math.nan
                mixing_volume = math.nan
                if node in self.reaches:
                    if flow <= 0.0:
                        raise ValueError(f'year {year}, month {month}, {node}: river mixing flow must be positive; stagnant/dry reaches require another transport model')
                    reach = self.reaches[node]
                    if self.upstream[node]:
                        upstream = self._upstream_concentration(node, pollutant, flows_m3_s)
                        parent_flow = sum(float(flows_m3_s[p]) for p in self.upstream[node])
                        lateral_flow = flow - parent_flow
                        if lateral_flow < -1e-9:
                            raise ValueError(f'{node}: river flow is below its upstream inflow')
                        lateral_c = float(boundaries.get(node, {}).get(pollutant, reach.background_mg_l[pollutant]))
                        upstream = (upstream * parent_flow + max(0.0, lateral_flow) * lateral_c) / flow
                    else:
                        upstream = float(boundaries.get(node, {}).get(pollutant, reach.background_mg_l[pollutant]))
                        if upstream < 0.0:
                            raise ValueError(f'{node}: boundary concentrations cannot be negative')
                    lo, hi = 1e-8, max(1.0, reach.depth_m)
                    def discharge(depth):
                        area = reach.width_m * depth
                        return area / reach.manning_n * (area / (reach.width_m + 2 * depth)) ** (2 / 3) * math.sqrt(reach.slope)
                    while discharge(hi) < flow:
                        hi *= 2
                    for _ in range(45):
                        mid = (lo + hi) / 2
                        if discharge(mid) < flow:
                            lo = mid
                        else:
                            hi = mid
                    velocity = flow / (reach.width_m * (lo + hi) / 2)
                    decay_second = float(reach.decay_per_day[pollutant]) / 86400.0
                    concentration = upstream
                    position = 0.0
                    selected = point_loads[(point_loads['node'] == node) & (point_loads['pollutant'] == pollutant)] if not point_loads.empty else point_loads
                    ordered_loads = selected.sort_values('position_km').itertuples(index=False) if not selected.empty else ()
                    for load in ordered_loads:
                        load_position = float(load.position_km)
                        if not position <= load_position <= reach.length_km:
                            raise ValueError(f'point load position outside reach {node}')
                        travel_seconds = (load_position - position) * 1000.0 / velocity
                        concentration *= math.exp(-decay_second * travel_seconds)
                        concentration += kg_month_to_mg_second(float(load.load_kg_month), year, month) / (flow * 1000.0)
                        position = load_position
                    final_seconds = (reach.length_km - position) * 1000.0 / velocity
                    concentration *= math.exp(-decay_second * final_seconds)
                    selected_nonpoint = nonpoint_loads[(nonpoint_loads['node'] == node) & (nonpoint_loads['pollutant'] == pollutant)] if not nonpoint_loads.empty else nonpoint_loads
                    total_nonpoint = float(selected_nonpoint['load_kg_month'].sum()) if not selected_nonpoint.empty else 0.0
                    if total_nonpoint > 0:
                        travel_total = reach.length_km * 1000.0 / velocity
                        inlet_equivalent = kg_month_to_mg_second(total_nonpoint, year, month) / (flow * 1000.0)
                        attenuation = (1.0 - math.exp(-decay_second * travel_total)) / (decay_second * travel_total) if decay_second > 0 and travel_total > 0 else 1.0
                        concentration += inlet_equivalent * attenuation
                else:
                    reservoir = self.reservoirs[node]
                    upstream = self._upstream_concentration(node, pollutant, flows_m3_s)
                    if not self.upstream[node]:
                        upstream = float(boundaries.get(node, {}).get(pollutant, upstream))
                        if upstream < 0.0:
                            raise ValueError(f'{node}: boundary concentrations cannot be negative')
                    previous = self.state[node].get(pollutant, upstream)
                    incoming_flow = sum((max(float(flows_m3_s.get(parent, 0.0)), 0.0) for parent in self.upstream[node])) if self.upstream[node] else flow
                    extra = (external_inflows or {}).get(node)
                    if extra and extra['flow_m3_s'] > 0:
                        extra_flow = float(extra['flow_m3_s'])
                        upstream = (upstream * incoming_flow + extra_flow * float(extra['concentration_mg_l'][pollutant])) / (incoming_flow + extra_flow)
                        incoming_flow += extra_flow
                    previous = self.state[node].get(pollutant, upstream)
                    mixing_volume = float((reservoir_volumes_m3 or {}).get(node, reservoir.volume_m3))
                    if not math.isfinite(mixing_volume) or mixing_volume <= 0:
                        raise ValueError(f'{node}: mixing volume must be finite and positive')
                    mixing_fraction = min(1.0, incoming_flow * seconds / mixing_volume)
                    mixed = mixing_fraction * upstream + (1.0 - mixing_fraction) * previous
                    concentration = mixed * math.exp(-float(reservoir.decay_per_day[pollutant]) * days)
                self.state[node][pollutant] = concentration
                results.append({'year': int(year), 'month': int(month), 'node': node, 'pollutant': pollutant, 'concentration_mg_l': concentration, 'previous_concentration_mg_l': previous, 'mixing_fraction': mixing_fraction, 'mixing_volume_m3': mixing_volume, 'grade': classify_water_quality(pollutant, concentration, waterbody='reservoir' if node in self.reservoirs else 'river')})
        return pd.DataFrame(results)

def empirical_exceedance(values: pd.Series) -> pd.DataFrame:
    clean = pd.to_numeric(values, errors='coerce').dropna().sort_values(ascending=False)
    if clean.empty:
        raise ValueError('flow series contains no numeric values')
    rank = np.arange(1, len(clean) + 1, dtype=float)
    return pd.DataFrame({'flow': clean.to_numpy(), 'exceedance_probability': rank / (len(clean) + 1.0)})

def classify_hydrological_years(daily_flow: pd.DataFrame, *, date_column: str='date', flow_column: str='flow') -> pd.Series:
    frame = daily_flow.copy()
    if 'year' not in frame:
        frame['year'] = pd.to_datetime(frame[date_column]).dt.year
    frame[flow_column] = pd.to_numeric(frame[flow_column], errors='raise')
    annual = frame.groupby('year')[flow_column].sum()
    frequency = empirical_exceedance(annual)
    frequency['year'] = annual.sort_values(ascending=False).index.to_numpy()
    probability = frequency['exceedance_probability']
    frequency['class'] = np.where(probability < 0.25, 'wet', np.where(probability <= 0.75, 'normal', 'dry'))
    return frequency.set_index('year')['class']

def flow_at_exceedance(values: pd.Series, probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError('probability must be in [0, 1]')
    curve = empirical_exceedance(values)
    return float(np.interp(probability, curve['exceedance_probability'], curve['flow']))

def multilevel_thresholds(values: pd.Series) -> tuple[float, float, float, float]:
    mef = (flow_at_exceedance(values, 0.9) + flow_at_exceedance(values, 0.95)) / 2.0
    oefmax = flow_at_exceedance(values, 0.5)
    if oefmax <= mef:
        raise ValueError('Q50 must exceed the derived minimum ecological flow')
    oefmin = mef + 5.0 / 9.0 * (oefmax - mef)
    ef2 = mef + (oefmin - mef) / 2.0
    return (mef, ef2, oefmin, oefmax)

def monthly_standards(daily_flow: pd.DataFrame, *, site_column: str='site', date_column: str='date', flow_column: str='flow') -> pd.DataFrame:
    frame = daily_flow.copy()
    if not {'year', 'month'} <= set(frame):
        dates = pd.to_datetime(frame[date_column])
        frame = frame.assign(year=dates.dt.year, month=dates.dt.month)
    outputs: list[dict[str, float | str | int]] = []
    for site, site_frame in frame.groupby(site_column):
        classes = classify_hydrological_years(site_frame, date_column=date_column, flow_column=flow_column)
        site_frame['hydrological_class'] = site_frame['year'].map(classes)
        for (hydro_class, month), group in site_frame.groupby(['hydrological_class', 'month']):
            thresholds = multilevel_thresholds(group[flow_column])
            outputs.append({'site': site, 'hydrological_class': hydro_class, 'month': int(month), 'MEF': thresholds[0], 'EF2': thresholds[1], 'OEFmin': thresholds[2], 'OEFmax': thresholds[3]})
    return pd.DataFrame(outputs).sort_values(['site', 'hydrological_class', 'month'])

def _unique_value(frame: pd.DataFrame, column: str, label: str) -> float:
    if len(frame) != 1:
        raise ValueError(f'expected one {label} row, found {len(frame)}')
    return float(frame.iloc[0][column])

def run_reservoir_period(config: ModelConfig, monthly_inputs: pd.DataFrame, eco_thresholds: pd.DataFrame, *, pso: PSOSettings | None=None, shape_factors: Mapping[str, float] | None=None, initial_storage: Mapping[str, float] | None=None) -> pd.DataFrame:
    required_input = {'year', 'month', 'reservoir', 'natural_inflow', 'demand'}
    required_eco = {'month', 'reservoir', 'MEF', 'EF2', 'OEFmin', 'OEFmax'}
    if not required_input <= set(monthly_inputs.columns):
        raise ValueError(f'monthly_inputs are missing {sorted(required_input - set(monthly_inputs.columns))}')
    if not required_eco <= set(eco_thresholds.columns):
        raise ValueError(f'eco_thresholds are missing {sorted(required_eco - set(eco_thresholds.columns))}')
    system = ReservoirSystem(config.reservoirs, config.ecological_membership)
    storage = system.initial_storage() if initial_storage is None else {key: float(value) for key, value in initial_storage.items()}
    if set(storage) != set(system.order):
        raise ValueError('initial_storage must define every configured reservoir exactly once')
    for key, value in storage.items():
        spec = system.specs[key]
        if not spec.dead_storage - 1e-10 <= value <= spec.max_storage + 1e-10:
            raise ValueError(f'initial storage for {key} is outside physical bounds')
    factors = dict(shape_factors or config.initial_shape_factors)
    settings = pso or PSOSettings()
    outputs: list[dict] = []
    dates = monthly_inputs[['year', 'month']].drop_duplicates().sort_values(['year', 'month'])
    for step, date in enumerate(dates.itertuples(index=False)):
        selected = monthly_inputs[(monthly_inputs['year'] == date.year) & (monthly_inputs['month'] == date.month)]
        inflow: dict[str, float] = {}
        demand: dict[str, float] = {}
        losses: dict[str, float] = {}
        eco: dict[str, tuple[float, float, float, float]] = {}
        for reservoir in system.order:
            row = selected[selected['reservoir'] == reservoir]
            inflow[reservoir] = _unique_value(row, 'natural_inflow', f'{date.year}-{date.month} {reservoir}')
            demand[reservoir] = _unique_value(row, 'demand', f'{date.year}-{date.month} {reservoir}')
            losses[reservoir] = _unique_value(row, 'water_loss', reservoir) if 'water_loss' in row else 0.0
            threshold_row = eco_thresholds[(eco_thresholds['month'] == date.month) & (eco_thresholds['reservoir'] == reservoir)]
            if 'hydrological_class' in eco_thresholds:
                if 'hydrological_class' not in row:
                    raise ValueError('user-supplied hydrological_class is required')
                threshold_row = threshold_row[threshold_row['hydrological_class'] == row.iloc[0]['hydrological_class']]
            if len(threshold_row) != 1:
                raise ValueError(f'missing/duplicate ecological thresholds for month {date.month}, {reservoir}')
            eco[reservoir] = tuple((float(threshold_row.iloc[0][name]) for name in ('MEF', 'EF2', 'OEFmin', 'OEFmax')))
            unit = str(threshold_row.iloc[0].get('unit', 'million_m3'))
            if unit == 'm3/s':
                scale = monthrange(int(date.year), int(date.month))[1] * 86400 / 1000000.0
                eco[reservoir] = tuple((value * scale for value in eco[reservoir]))
            elif unit != 'million_m3':
                raise ValueError('ecological threshold unit must be m3/s or million_m3')
        step_settings = PSOSettings(**{**settings.__dict__, 'seed': settings.seed + step})
        result = optimize_month(system, year=int(date.year), month=int(date.month), storage=storage, natural_inflow=inflow, demand=demand, eco_thresholds=eco, shape_factors=factors, settings=step_settings, water_losses=losses)
        if not result.feasible:
            total = sum(result.violations.values())
            raise RuntimeError(f'PSO returned an infeasible solution at {date.year}-{date.month}: {total:.3e}')
        storage = result.next_storage
        for record in result.records:
            source = selected[selected['reservoir'] == record['reservoir']].iloc[0]
            record['hydrological_class'] = source.get('hydrological_class', 'user_defined')
            record.update(dict(zip(('MEF', 'EF2', 'OEFmin', 'OEFmax'), eco[record['reservoir']])))
            record['synergy_volume'] = result.synergy_volume
            record.update({f'shape_{key}': value for key, value in factors.items()})
            outputs.append(record)
    return pd.DataFrame(outputs)

def _load_wq_network(path: Path) -> WaterQualityNetwork:
    raw = dict(path) if isinstance(path, Mapping) else json.loads(Path(path).read_text(encoding='utf-8'))
    reaches = [Reach(**row) for row in raw['reaches']]
    reservoirs = [ReservoirNode(**row) for row in raw.get('reservoirs', [])]
    return WaterQualityNetwork(reaches, reservoirs)

def _decision_periods(raw: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    rows = raw.get('periods')
    if not isinstance(rows, list) or not rows:
        raise ValueError('closed-loop configuration must contain decision periods')
    periods: list[tuple[str, int, int]] = []
    for index, row in enumerate(rows, start=1):
        label = str(row.get('label', f'P{index}'))
        start = int(row['start_year'])
        end = int(row['end_year'])
        if start > end:
            raise ValueError(f'{label}: start_year exceeds end_year')
        periods.append((label, start, end))
    for previous, current in zip(periods, periods[1:]):
        if current[1] != previous[2] + 1:
            raise ValueError('decision periods must be consecutive and non-overlapping')
    expected_count = raw.get('expected_period_count')
    if expected_count is not None and len(periods) != int(expected_count):
        raise ValueError('decision-period count does not match expected_period_count')
    expected_length = raw.get('expected_years_per_period')
    if expected_length is not None:
        if any((end - start + 1 != int(expected_length) for _, start, end in periods)):
            raise ValueError('a decision period has an unexpected length')
    return periods

def _year_inputs(year: int, config: ModelConfig, hydrology_template: pd.DataFrame, regional_demand: pd.Series, demand_mapping: pd.DataFrame, seasonality: Mapping[str, float], demand_basis: str='monthly_mean') -> pd.DataFrame:
    required = {'month', 'reservoir', 'natural_inflow'}
    if not required <= set(hydrology_template.columns):
        raise ValueError(f'hydrology template is missing {sorted(required - set(hydrology_template.columns))}')
    template = hydrology_template.copy()
    if 'year' in template:
        template = template[template['year'] == year].copy()
    if template.duplicated(['month', 'reservoir']).any():
        raise ValueError('hydrology template contains duplicate month-reservoir rows')
    expected = {(month, spec.reservoir_id) for month in range(1, 13) for spec in config.reservoirs}
    found = set(template[['month', 'reservoir']].itertuples(index=False, name=None))
    if found != expected:
        raise ValueError('hydrology template must define every month-reservoir pair exactly once')
    reservoir_demand = allocate_series(regional_demand, demand_mapping, source_key='region', target_key='reservoir')
    reservoir_ids = {spec.reservoir_id for spec in config.reservoirs}
    if set(reservoir_demand.index) != reservoir_ids:
        raise ValueError('demand mapping must allocate to every configured reservoir')
    monthly_factor = validate_seasonality(seasonality)
    scale = demand_scale(demand_basis)
    template['year'] = int(year)
    template['demand'] = template.apply(lambda row: float(reservoir_demand.loc[row['reservoir']]) * scale * monthly_factor[int(row['month'])], axis=1)
    return template

def _ending_storage(results: pd.DataFrame) -> dict[str, float]:
    ordered = results.sort_values(['year', 'month'])
    final_rows = ordered.groupby('reservoir', as_index=False).tail(1)
    return {str(row.reservoir): float(row.storage_end) for row in final_rows.itertuples(index=False)}

def _performance(reservoir: pd.DataFrame, quality_scores: pd.DataFrame) -> dict[str, float]:
    return {'water_supply': float(reservoir['supply_ratio'].mean()), 'hydropower': float(reservoir['power_ratio'].mean()), 'ecological_flow': float(reservoir['eco_ratio'].mean()), 'water_quality_grade': float(quality_scores['water_quality_grade'].mean())}

def _quality_for_year(network: WaterQualityNetwork, reservoir_results: pd.DataFrame, flow_mapping: list[Mapping[str, str]], point_template: pd.DataFrame, nonpoint_template: pd.DataFrame, dynamic_point_node_loads: pd.DataFrame, dynamic_nonpoint_node_loads: pd.DataFrame, dynamic_point_positions_km: Mapping[str, float], boundary_concentrations_mg_l: Mapping | None=None, storage_mapping: Mapping[str, str] | None=None, transfer_concentrations: Mapping | None=None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dynamic_point_long = dynamic_point_node_loads.rename_axis('node').reset_index().melt(id_vars='node', var_name='pollutant', value_name='load_kg_month')
    dynamic_point_long = dynamic_point_long[dynamic_point_long['load_kg_month'] > 0.0].copy()
    dynamic_point_long['position_km'] = dynamic_point_long['node'].map({str(key): float(value) for key, value in dynamic_point_positions_km.items()})
    if dynamic_point_long['position_km'].isna().any():
        missing = sorted(dynamic_point_long.loc[dynamic_point_long['position_km'].isna(), 'node'].unique())
        raise ValueError(f'missing dynamic point-source positions for nodes {missing}')
    dynamic_point_long = dynamic_point_long[['node', 'pollutant', 'position_km', 'load_kg_month']]
    dynamic_nonpoint_long = dynamic_nonpoint_node_loads.rename_axis('node').reset_index().melt(id_vars='node', var_name='pollutant', value_name='load_kg_month')
    for (year, month), group in reservoir_results.groupby(['year', 'month'], sort=True):
        by_reservoir = group.set_index('reservoir')
        seconds = monthrange(int(year), int(month))[1] * 86400.0
        flows: dict[str, float] = {}
        for row in flow_mapping:
            node = str(row['node'])
            reservoir = str(row['reservoir'])
            fields = row['field'] if isinstance(row['field'], list) else [row['field']]
            if reservoir not in by_reservoir.index or any(f not in by_reservoir.columns for f in fields):
                raise ValueError(f'invalid water-quality flow mapping for {node}')
            flows[node] = float(by_reservoir.loc[reservoir, fields].sum()) * 1000000.0 / seconds
        external = {}
        for node, reservoir in (storage_mapping or {}).items():
            imported = float(by_reservoir.loc[reservoir, 'transfer_in'])
            if imported > 0:
                if reservoir not in (transfer_concentrations or {}):
                    raise ValueError('transfer-in requires external water-quality concentrations')
                external[node] = {'flow_m3_s': imported * 1e6 / seconds, 'concentration_mg_l': transfer_concentrations[reservoir]}
        baseline_point = point_template[point_template['month'] == month].drop(columns='month')
        point = pd.concat([baseline_point, dynamic_point_long], ignore_index=True)
        point = point.groupby(['node', 'pollutant', 'position_km'], as_index=False)['load_kg_month'].sum()
        baseline = nonpoint_template[nonpoint_template['month'] == month].drop(columns='month')
        nonpoint = pd.concat([baseline, dynamic_nonpoint_long], ignore_index=True)
        nonpoint = nonpoint.groupby(['node', 'pollutant'], as_index=False)['load_kg_month'].sum()
        frames.append(network.simulate_month(year=int(year), month=int(month), flows_m3_s=flows, point_loads=point, nonpoint_loads=nonpoint, boundary_concentrations_mg_l=boundary_concentrations_mg_l, reservoir_volumes_m3={node: float(by_reservoir.loc[reservoir, ['storage_start', 'storage_end']].mean()) * 1000000.0 for node, reservoir in (storage_mapping or {}).items()}, external_inflows=external))
    return pd.concat(frames, ignore_index=True)

def _state_rows(state: pd.DataFrame, *, period: str, year: int, state_time: str) -> pd.DataFrame:
    frame = state.rename_axis('region').reset_index()
    frame.insert(0, 'state_time', state_time)
    frame.insert(0, 'year', int(year))
    frame.insert(0, 'period', period)
    return frame

def _time_value(value, year, month):
    if isinstance(value, Mapping):
        key = f'{year}-{month:02d}'
        value = value[key] if key in value else value[str(year)]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('time-varying inputs must be finite')
    return value

def run_closed_loop(input_file: Path, output_dir: Path, *, pso: PSOSettings | None=None, quality_scenario: str | None=None, feedback_mode: str='full') -> dict[str, Path]:
    input_file = Path(input_file)
    data = json.loads(input_file.read_text(encoding='utf-8'))
    loop_raw = data['schedule']
    periods = _decision_periods(loop_raw)
    warmup = loop_raw.get('warmup')
    run_periods = list(periods)
    if warmup:
        first, last = (int(warmup['start_year']), int(warmup['end_year']))
        if first < 1 or first > last or last + 1 != periods[0][1]:
            raise ValueError('warmup must immediately precede evaluation periods')
        run_periods.insert(0, ('WARMUP', first, last))
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f'unknown feedback mode {feedback_mode!r}')
    demand_basis = loop_raw['demand_basis']
    demand_scale(demand_basis)
    validate_seasonality(loop_raw['demand_seasonality'])
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(data['model'])
    hydrology_config = data['hydrology']
    if hydrology_config['mode'] == 'generated':
        hydrology, eco, _, _ = generate_hydrology(hydrology_config['settings'])
    elif hydrology_config['mode'] == 'provided':
        hydrology = pd.DataFrame(hydrology_config['monthly_inputs'])
        eco = pd.DataFrame(hydrology_config['ecological_thresholds'])
    else:
        raise ValueError('hydrology mode must be generated or provided')
    demand_mapping = pd.DataFrame(data['demand_mapping'])
    pollution_mapping = pd.DataFrame(data['pollution_mapping'])
    point_template = pd.DataFrame(data['point_loads'])
    nonpoint_template = pd.DataFrame(data['nonpoint_loads'])
    state = pd.DataFrame(data['socioeconomic_state']).set_index('region')
    growth = pd.DataFrame(data['natural_growth']).set_index('region')
    quotas = pd.DataFrame(data['water_use_quotas']).set_index('region')
    sd_parameters = SDParameters(**data['sd_parameters'])
    if set(data['gdp']['regional']) != set(state.index):
        raise ValueError('GDP inputs must explicitly define every region')
    drivers = data['external_drivers']
    if set(drivers) != set(state.index) or any(set(drivers[r]) != set(SECTORS) for r in state.index):
        raise ValueError('external_drivers must define Gamma for every region and sector')
    pollution_raw = data['pollution_parameters']
    network = _load_wq_network(data['water_quality_network'])
    quality_raw = data['quality_scenarios']
    scenario_name = quality_scenario or loop_raw['quality_scenario']
    boundaries_by_period = quality_schedule(quality_raw, scenario_name, periods, network)
    if warmup:
        boundaries_by_period['WARMUP'] = boundaries_by_period[periods[0][0]]
    storage_mapping = loop_raw.get('water_quality_storage_mapping', {})
    if set(storage_mapping) != {node for node in network.reservoirs}:
        raise ValueError('water_quality_storage_mapping must map every water-quality reservoir')
    flow_mapping = loop_raw.get('water_quality_flow_mapping', [])
    if not isinstance(flow_mapping, list) or not flow_mapping:
        raise ValueError('water_quality_flow_mapping must be a non-empty list')
    point_positions = loop_raw.get('dynamic_point_source_positions_km', {})
    if not isinstance(point_positions, Mapping):
        raise ValueError('dynamic_point_source_positions_km must be an object')
    settings = pso or PSOSettings(particles=32, generations=40)
    factors = dict(config.initial_shape_factors)
    storage: dict[str, float] | None = None
    year_offset = 0
    reservoir_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    quality_score_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    demand_frames: list[pd.DataFrame] = []
    pollution_frames: list[pd.DataFrame] = []
    node_pollution_frames: list[pd.DataFrame] = []
    monthly_performance_rows: list[dict[str, float | int | str]] = []
    period_rows: list[dict[str, float | int | str]] = []
    for period_label, start_year, end_year in run_periods:
        period_reservoir: list[pd.DataFrame] = []
        period_quality_scores: list[pd.DataFrame] = []
        factors_used = dict(factors)
        for year, month in ((y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)):
            state_frames.append(_state_rows(state, period=period_label, year=year, state_time='start_of_month').assign(month=month))
            regional_demand = calculate_water_demand(state, quotas).rename('water_demand')
            demand_frame = regional_demand.rename_axis('region').reset_index()
            demand_frame.insert(0, 'year', year)
            demand_frame.insert(0, 'period', period_label)
            demand_frame['month'] = month
            demand_frames.append(demand_frame)
            regional_pollution_by_source = calculate_pollutant_load_by_source(state, pollution_raw['generation_coefficients'], pollution_raw['removal_rates'])
            regional_point_pollution = regional_pollution_by_source['point']
            regional_nonpoint_pollution = regional_pollution_by_source['nonpoint']
            regional_pollution = regional_point_pollution.add(regional_nonpoint_pollution, fill_value=0.0)
            pollution_frame = regional_pollution.rename_axis('region').reset_index()
            pollution_frame.insert(0, 'year', year)
            pollution_frame.insert(0, 'period', period_label)
            pollution_frame['month'] = month
            pollution_frames.append(pollution_frame)
            node_point_pollution = allocate_frame(regional_point_pollution, pollution_mapping, source_key='region', target_key='node')
            node_nonpoint_pollution = allocate_frame(regional_nonpoint_pollution, pollution_mapping, source_key='region', target_key='node')
            node_pollution = node_point_pollution.add(node_nonpoint_pollution, fill_value=0.0)
            node_pollution_frame = node_pollution.rename_axis('node').reset_index()
            node_pollution_frame.insert(0, 'year', year)
            node_pollution_frame.insert(0, 'period', period_label)
            node_pollution_frame['month'] = month
            node_pollution_frames.append(node_pollution_frame)
            year_inputs = _year_inputs(year, config, hydrology, regional_demand, demand_mapping, loop_raw['demand_seasonality'], demand_basis)
            year_inputs = year_inputs[year_inputs['month'] == month]
            year_settings = PSOSettings(**{**asdict(settings), 'seed': settings.seed + year_offset})
            reservoir_year = run_reservoir_period(config, year_inputs, eco, pso=year_settings, shape_factors=factors_used, initial_storage=storage)
            storage = _ending_storage(reservoir_year)
            reservoir_year.insert(0, 'period', period_label)
            days = monthrange(year, month)[1]
            quality_year = _quality_for_year(network, reservoir_year, flow_mapping, point_template, nonpoint_template, node_point_pollution * days, node_nonpoint_pollution * days, point_positions, boundaries_by_period[period_label], storage_mapping, loop_raw.get('transfer_in_concentrations_mg_l'))
            quality_year.insert(0, 'period', period_label)
            quality_scores_year = water_quality_score(quality_year)
            quality_scores_year.insert(0, 'period', period_label)
            performance = _performance(reservoir_year, quality_scores_year)
            monthly_performance_rows.append({'period': period_label, 'year': year, 'month': month, **performance, **{f'deficit_{k}': v for k, v in performance_deficits(performance, config.satisfaction_thresholds).items()}, 'quality_boundary_condition': quality_raw['scenarios'][scenario_name].get(period_label, 'warmup_first_boundary'), 'sd_feedback_enabled': int(feedback_mode in {'full', 'sd_only'}), 'quality_pressure': constraint_pressure(performance['water_quality_grade'], sd_parameters.thresholds['water_quality'], sd_parameters.pressure_steepness['water_quality'])})
            reservoir_frames.append(reservoir_year)
            quality_frames.append(quality_year)
            quality_score_frames.append(quality_scores_year)
            period_reservoir.append(reservoir_year)
            period_quality_scores.append(quality_scores_year)
            regional_gdp = {region: _time_value(value, year, month) for region, value in data['gdp']['regional'].items()}
            national_gdp = _time_value(data['gdp']['national'], year, month)
            forcing = pd.DataFrame({region: {sector: _time_value(value, year, month) for sector, value in values.items()} for region, values in drivers.items()}).T.reindex(state.index)
            state = project_state(state, growth, performance if feedback_mode in {'full', 'sd_only'} else {'water_supply': 1.0, 'hydropower': 1.0, 'ecological_flow': 1.0, 'water_quality_grade': 1.0}, sd_parameters, regional_gdp=regional_gdp, national_gdp=national_gdp, dt_years=1 / 12, external_drivers=forcing)
            year_offset += 1
        if period_label == 'WARMUP':
            continue
        combined_reservoir = pd.concat(period_reservoir, ignore_index=True)
        combined_quality_scores = pd.concat(period_quality_scores, ignore_index=True)
        period_performance = _performance(combined_reservoir, combined_quality_scores)
        period_performance['DoSR'] = float(combined_reservoir.groupby(['year', 'month'])['system_donation_ratio'].first().mean())
        next_factors = update_shape_factors(period_performance, factors_used, config.satisfaction_thresholds, maximum_adjustment=config.feedback.get('maximum_adjustment', 0.2), steepness=config.feedback.get('steepness', 5.0)) if feedback_mode in {'full', 'weights_only'} else dict(factors_used)
        period_rows.append({'period': period_label, 'start_year': start_year, 'end_year': end_year, **period_performance, **feedback_diagnostics(period_performance, factors_used, config.satisfaction_thresholds, maximum_adjustment=config.feedback.get('maximum_adjustment', 0.2), steepness=config.feedback.get('steepness', 5.0), enabled=feedback_mode in {'full', 'weights_only'}), 'quality_boundary_condition': quality_raw['scenarios'][scenario_name][period_label], **{f'shape_used_{key}': value for key, value in factors_used.items()}, **{f'shape_next_{key}': value for key, value in next_factors.items()}})
        factors = next_factors
    state_frames.append(_state_rows(state, period='terminal', year=periods[-1][2] + 1, state_time='terminal_after_last_month').assign(month=1))
    path = output_dir / 'period_performance_and_feedback.csv'
    pd.DataFrame(period_rows).to_csv(path, index=False)
    return {'period_feedback': path}

def generate_hydrology(settings):
    if settings['ecological_management_grade_count'] != 3:
        raise ValueError('four ecological thresholds require N=3 (one intermediate EFr)')
    rng = np.random.default_rng(settings['seed'])
    seasonal = np.asarray(settings['seasonal_factors'], dtype=float)
    seasonal /= seasonal.mean()
    records, daily = ([], [])
    for year in range(1, settings['years'] + 1):
        common = rng.uniform(*settings['annual_factor_range'])
        annual = {r: 0.5 * common + 0.5 * rng.uniform(*settings['annual_factor_range']) for r in settings['mean_monthly_incremental_inflow_million_m3']}
        for month, factor in enumerate(seasonal, 1):
            days = monthrange(year, month)[1]
            noise = rng.lognormal(0, settings['daily_lognormal_sigma'], days)
            noise /= noise.mean()
            upstream = np.zeros(days)
            for reservoir, mean in settings['mean_monthly_incremental_inflow_million_m3'].items():
                local_noise = rng.lognormal(0, settings['daily_lognormal_sigma'], days)
                local_noise /= local_noise.mean()
                mixed_noise = 0.5 * noise + 0.5 * local_noise
                volume = mean * factor * annual[reservoir]
                incremental = volume * 1000000.0 / (days * 86400) * mixed_noise
                cumulative = upstream + incremental
                daily.extend((dict(year=year, month=month, day=d + 1, site=reservoir, flow=float(flow)) for d, flow in enumerate(cumulative)))
                upstream = cumulative
                records.append(dict(year=year, month=month, reservoir=reservoir, natural_inflow=volume, water_loss=settings['mean_monthly_loss_million_m3'][reservoir] * factor))
    inputs = pd.DataFrame(records)
    daily_frame = pd.DataFrame(daily)
    annual = daily_frame.groupby(['site', 'year'], as_index=False)['flow'].sum()
    annual['annual_runoff_million_m3'] = annual.pop('flow') * 86400 / 1000000.0
    annual['exceedance_probability'] = annual.groupby('site')['annual_runoff_million_m3'].rank(ascending=False, method='first') / (settings['years'] + 1)
    probability = annual['exceedance_probability']
    annual['hydrological_class'] = np.where(probability < 0.25, 'wet', np.where(probability <= 0.75, 'normal', 'dry'))
    annual = annual.rename(columns={'site': 'reservoir'})
    inputs = inputs.merge(annual[['reservoir', 'year', 'hydrological_class']], on=['reservoir', 'year'])
    thresholds = monthly_standards(daily_frame).rename(columns={'site': 'reservoir'})
    thresholds['unit'] = 'm3/s'
    frames = (inputs, thresholds, daily_frame, annual)
    return tuple((pd.read_csv(StringIO(frame.to_csv(index=False))) for frame in frames))

def main(argv=None):
    parser = argparse.ArgumentParser(description='SHE-Nexus 1.2.0')
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('example_data.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('outputs'))
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quality-scenario')
    parser.add_argument('--feedback-mode', choices=sorted(FEEDBACK_MODES), default='full')
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.input.resolve() or args.output_dir.resolve() == Path(__file__).resolve():
        parser.error('output directory must not be an input or source file')
    results = run_closed_loop(args.input, args.output_dir, pso=PSOSettings(particles=args.particles, generations=args.generations, seed=args.seed), quality_scenario=args.quality_scenario, feedback_mode=args.feedback_mode)
    for name, path in results.items():
        print(f'{name}: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
