"""Typed, JSON-friendly records used by the scientific core.

The records in this module describe results from curated, reduced-order teaching
models.  They must not be interpreted as measurements or as outputs from
validated genome-scale reconstructions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class DictRecord:
    """Mixin providing a predictable representation for downloads and APIs."""

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively converted dictionary."""

        return asdict(self)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Nutrient(DictRecord):
    """A modelled medium component.

    ``amount`` values used elsewhere are maximum-availability constraints in
    model flux units.  They are not concentrations unless a user has fitted a
    concentration-to-uptake relationship for their own process.
    """

    id: str
    name: str
    exchange_id: str
    metabolite_id: str
    category: str
    unit: str
    cost_per_unit: float
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class FungusProfile(DictRecord):
    """Inspectable configuration for one reduced-order fungal surrogate."""

    id: str
    name: str
    short_name: str
    industrial_use: str
    signature_product: str
    signature_reaction: str
    temperature: float
    ph: float
    accent: str
    default_medium: dict[str, float]
    carbon_efficiencies: dict[str, float]
    uptake_capacities: dict[str, float]
    biomass_coefficients: dict[str, float]
    notes: tuple[str, ...]

    @property
    def scientific_name(self) -> str:
        """Alias used by catalogue consumers."""

        return self.name


@dataclass(frozen=True)
class FluxBalanceResult(DictRecord):
    """A deterministic FBA result from a reduced-order surrogate."""

    fungus_id: str
    status: str
    objective_reaction: str
    objective_value: float
    growth_rate: float
    biomass_flux: float
    objective_flux: float
    biomass_yield: float
    yield_coefficient: float
    fluxes: dict[str, float]
    medium: dict[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FluxRange(DictRecord):
    """Minimum and maximum feasible flux for one reaction."""

    reaction_id: str
    reaction_name: str
    minimum: float
    maximum: float

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class FluxVariabilityResult(DictRecord):
    """FVA ranges while retaining a requested fraction of optimum growth."""

    fungus_id: str
    status: str
    objective_value: float
    growth_rate: float
    fraction_of_optimum: float
    ranges: dict[str, FluxRange]
    medium: dict[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SensitivityEntry(DictRecord):
    """Central finite-difference sensitivity for one medium factor."""

    rank: int
    parameter: str
    nutrient_id: str
    baseline_amount: float
    baseline_growth: float
    lower_amount: float
    lower_growth: float
    upper_amount: float
    upper_growth: float
    elasticity: float
    impact: float
    direction: str
    priority: str


@dataclass(frozen=True)
class SensitivityAnalysis(DictRecord):
    """Full sensitivity calculation plus reproducibility metadata."""

    fungus_id: str
    baseline_growth: float
    perturbation_fraction: float
    rankings: tuple[SensitivityEntry, ...]
    evaluations: int
    medium: dict[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MediumCandidate(DictRecord):
    """One medium evaluated during deterministic optimisation."""

    rank: int
    medium: dict[str, float]
    growth_rate: float
    estimated_cost: float
    score: float
    changed_nutrients: tuple[str, ...]


@dataclass(frozen=True)
class MediumOptimizationResult(DictRecord):
    """Best feasible candidate and the candidates used to select it."""

    fungus_id: str
    status: str
    baseline_medium: dict[str, float]
    optimized_medium: dict[str, float]
    baseline_growth: float
    growth_rate: float
    objective_value: float
    baseline_cost: float
    estimated_cost: float
    budget: float
    target_fraction: float
    candidates: tuple[MediumCandidate, ...]
    evaluations: int
    warnings: tuple[str, ...]

    @property
    def optimized_growth(self) -> float:
        return self.growth_rate


@dataclass(frozen=True)
class GeneInteraction(DictRecord):
    """One transparent contribution to a qualitative morphology score."""

    rule_id: str
    gene: str
    state: str
    condition: str
    effects: dict[str, float]
    explanation: str
    evidence_url: str | None
    confidence: str
    applied: bool


@dataclass(frozen=True)
class MorphologyPrediction(DictRecord):
    """Qualitative morphology hypothesis from explicit gene/media rules.

    ``support_scores`` are relative rule scores, not calibrated probabilities.
    The result is intended to prioritise follow-up experiments only.
    """

    fungus_id: str
    predicted_morphology: str
    confidence: str
    support_scores: dict[str, float]
    latent_scores: dict[str, float]
    drivers: tuple[str, ...]
    interaction_trace: tuple[GeneInteraction, ...]
    insufficient_evidence: tuple[str, ...]
    warnings: tuple[str, ...]
