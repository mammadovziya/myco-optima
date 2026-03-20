"""Deterministic, sensitivity-guided design-of-experiments utilities.

The intended workflow starts with four controllable factors at three candidate
levels (3**4 = 81 conditions), retains the three most influential factors from
the model sensitivity analysis, and emits a 15-run Box–Behnken follow-up.  This
is a prioritisation tool; it does not replace biological replicates or wet-lab
validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Factor:
    """A bounded process or media factor used in the follow-up design."""

    name: str
    low: float
    high: float
    unit: str = ""

    @property
    def centre(self) -> float:
        return (self.low + self.high) / 2

    def decode(self, coded_level: float) -> float:
        return self.centre + coded_level * (self.high - self.low) / 2


@dataclass(frozen=True)
class ScreeningDesign:
    """Selected runs and enough provenance to explain the reduction."""

    runs: pd.DataFrame
    candidate_count: int
    selected_count: int
    retained_factors: tuple[str, ...]
    criterion: str = "sensitivity-ranked three-factor Box–Behnken"

    @property
    def reduction_percent(self) -> float:
        return 100 * (1 - self.selected_count / self.candidate_count)


def _validate_factors(factors: Sequence[Factor]) -> None:
    if len(factors) < 3:
        raise ValueError("At least three candidate factors are required.")
    if len({factor.name for factor in factors}) != len(factors):
        raise ValueError("Factor names must be unique.")
    if any(factor.low >= factor.high for factor in factors):
        raise ValueError("Every factor must have low < high.")


def _rank_names(
    factors: Sequence[Factor],
    sensitivities: Mapping[str, float] | Sequence[str],
) -> tuple[str, ...]:
    known = {factor.name for factor in factors}
    if isinstance(sensitivities, Mapping):
        # Absolute effect matters for screening: a strong negative effect is
        # still experimentally informative. Factor name is a stable tie-break.
        names = tuple(
            name
            for name, _ in sorted(
                sensitivities.items(), key=lambda item: (-abs(float(item[1])), item[0])
            )
        )
    else:
        names = tuple(sensitivities)

    if len(set(names)) != len(names):
        raise ValueError("Sensitivity ranking must not contain duplicate factors.")
    unknown = set(names) - known
    if unknown:
        raise ValueError(f"Unknown factor(s) in sensitivity ranking: {sorted(unknown)}")

    # Unranked factors stay eligible but come last in their declared order.
    names += tuple(factor.name for factor in factors if factor.name not in names)
    return names


def _box_behnken_points(centre_replicates: int) -> np.ndarray:
    if centre_replicates < 1:
        raise ValueError("At least one centre replicate is required.")

    points: list[tuple[float, float, float]] = []
    for fixed_axis in range(3):
        active_axes = [axis for axis in range(3) if axis != fixed_axis]
        for first, second in product((-1.0, 1.0), repeat=2):
            point = [0.0, 0.0, 0.0]
            point[active_axes[0]] = first
            point[active_axes[1]] = second
            points.append(tuple(point))
    points.extend([(0.0, 0.0, 0.0)] * centre_replicates)
    return np.asarray(points, dtype=float)


def build_sensitivity_guided_design(
    factors: Sequence[Factor],
    sensitivities: Mapping[str, float] | Sequence[str],
    *,
    centre_replicates: int = 3,
) -> ScreeningDesign:
    """Reduce a full three-level grid to a three-factor Box–Behnken design.

    ``sensitivities`` can be either ``{factor: effect}`` or an already-ranked
    sequence of names. The three highest-ranked candidate factors are retained.
    With the default four factors and three centre replicates this turns 81
    candidate conditions into 15 planned runs.
    """

    factors = tuple(factors)
    _validate_factors(factors)
    ranked_names = _rank_names(factors, sensitivities)
    retained_names = ranked_names[:3]
    factor_by_name = {factor.name: factor for factor in factors}
    retained = tuple(factor_by_name[name] for name in retained_names)

    coded = _box_behnken_points(centre_replicates)
    decoded = {
        factor.name: [round(factor.decode(value), 6) for value in coded[:, index]]
        for index, factor in enumerate(retained)
    }
    runs = pd.DataFrame(decoded)
    runs.insert(0, "run", np.arange(1, len(runs) + 1))
    runs["design_role"] = [
        "centre replicate" if np.allclose(point, 0) else "interaction edge" for point in coded
    ]

    return ScreeningDesign(
        runs=runs,
        candidate_count=3 ** len(factors),
        selected_count=len(runs),
        retained_factors=retained_names,
    )
