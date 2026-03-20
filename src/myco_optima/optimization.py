"""Deterministic FBA, FVA, sensitivity and medium-search helpers.

All calculations operate on the curated reduced-order teaching models in this
package.  The values are internally consistent flux-equivalent scores, not
validated growth rates from genome-scale reconstructions.  Use them to compare
scenarios and design follow-up work, not as cultivation set-points.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product
from numbers import Real
from typing import Any

from cobra import Model
from cobra.flux_analysis import flux_variability_analysis

from .catalog import (
    MODEL_DISCLAIMER,
    NUTRIENTS,
    FungusProfile,
    get_fungus,
    get_nutrient,
    normalise_medium,
)
from .models import (
    ASSIMILATION_REACTIONS,
    BIOMASS_REACTION,
    CARBON_EQUIVALENTS,
    apply_medium,
    build_model,
    effective_medium,
    get_model_fungus,
)
from .types import (
    FluxBalanceResult,
    FluxRange,
    FluxVariabilityResult,
    MediumCandidate,
    MediumOptimizationResult,
    SensitivityAnalysis,
    SensitivityEntry,
)

_EPSILON = 1e-9
_RESULT_WARNINGS = (
    MODEL_DISCLAIMER,
    "Medium amounts are maximum-availability constraints, not fitted concentrations.",
)


def _clean(value: Any) -> float:
    """Convert solver numerics to stable, finite floats."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or abs(number) < _EPSILON:
        return 0.0
    return number


def _prepared_model(
    model_or_fungus: Model | str | FungusProfile,
    medium: Mapping[str, float] | None,
) -> tuple[Model, FungusProfile]:
    if isinstance(model_or_fungus, Model):
        working = model_or_fungus.copy()
        profile = get_model_fungus(working)
        if medium is not None:
            working = apply_medium(working, medium, inplace=True)
    else:
        profile = get_fungus(model_or_fungus)
        working = build_model(profile, medium)
    return working, profile


def _scenario_warnings(
    profile: FungusProfile,
    medium: Mapping[str, float] | None,
) -> tuple[str, ...]:
    warnings = list(_RESULT_WARNINGS)
    if medium:
        requested = normalise_medium(medium)
        capped = [
            nutrient_id
            for nutrient_id, amount in requested.items()
            if amount > profile.uptake_capacities[nutrient_id] + _EPSILON
        ]
        if capped:
            warnings.append(
                "Requested availability exceeded the model capacity and was capped for: "
                + ", ".join(capped)
                + "."
            )
    return tuple(warnings)


def run_fba(
    model_or_fungus: Model | str | FungusProfile,
    medium: Mapping[str, float] | None = None,
    *,
    objective: str = BIOMASS_REACTION,
) -> FluxBalanceResult:
    """Run flux balance analysis without mutating the caller's model.

    ``objective`` may select the illustrative product sink for scenario
    exploration.  ``growth_rate`` and ``biomass_flux`` always report the
    biomass pseudo-reaction, even when another objective is selected.
    """

    working, profile = _prepared_model(model_or_fungus, medium)
    if objective not in working.reactions:
        raise KeyError(f"Unknown objective reaction {objective!r}.")
    working.objective = objective
    working.objective_direction = "max"
    solution = working.optimize()
    status = str(solution.status)
    scenario = effective_medium(working)
    warnings = list(_scenario_warnings(profile, medium))

    if status != "optimal":
        warnings.append(f"Solver status was {status!r}; all reported fluxes were set to zero.")
        return FluxBalanceResult(
            fungus_id=profile.id,
            status=status,
            objective_reaction=objective,
            objective_value=0.0,
            growth_rate=0.0,
            biomass_flux=0.0,
            objective_flux=0.0,
            biomass_yield=0.0,
            yield_coefficient=0.0,
            fluxes={},
            medium=scenario,
            warnings=tuple(warnings),
        )

    fluxes = {reaction.id: _clean(solution.fluxes[reaction.id]) for reaction in working.reactions}
    objective_value = _clean(solution.objective_value)
    biomass_flux = _clean(solution.fluxes[BIOMASS_REACTION])
    objective_flux = _clean(solution.fluxes[objective])

    carbon_uptake = 0.0
    for nutrient_id, equivalents in CARBON_EQUIVALENTS.items():
        exchange_flux = _clean(solution.fluxes[NUTRIENTS[nutrient_id].exchange_id])
        carbon_uptake += (
            max(0.0, -exchange_flux) * equivalents * profile.carbon_efficiencies[nutrient_id]
        )
    biomass_yield = biomass_flux / carbon_uptake if carbon_uptake > _EPSILON else 0.0

    return FluxBalanceResult(
        fungus_id=profile.id,
        status=status,
        objective_reaction=objective,
        objective_value=objective_value,
        growth_rate=biomass_flux,
        biomass_flux=biomass_flux,
        objective_flux=objective_flux,
        biomass_yield=biomass_yield,
        yield_coefficient=biomass_yield,
        fluxes=fluxes,
        medium=scenario,
        warnings=tuple(warnings),
    )


def _default_fva_reactions(model: Model, profile: FungusProfile) -> list[str]:
    medium = effective_medium(model)
    reaction_ids = [BIOMASS_REACTION, "RESP", "ATPM", profile.signature_reaction]
    for nutrient_id, amount in medium.items():
        if amount <= _EPSILON:
            continue
        reaction_ids.append(NUTRIENTS[nutrient_id].exchange_id)
        assimilation = ASSIMILATION_REACTIONS.get(nutrient_id)
        if assimilation:
            reaction_ids.append(assimilation)
    return list(dict.fromkeys(item for item in reaction_ids if item in model.reactions))


def run_fva(
    model_or_fungus: Model | str | FungusProfile,
    medium: Mapping[str, float] | None = None,
    *,
    fraction_of_optimum: float = 0.95,
    reaction_ids: Sequence[str] | None = None,
) -> FluxVariabilityResult:
    """Run single-process FVA while retaining a fraction of maximum biomass."""

    if not 0 < fraction_of_optimum <= 1:
        raise ValueError("fraction_of_optimum must be in the interval (0, 1].")

    working, profile = _prepared_model(model_or_fungus, medium)
    working.objective = BIOMASS_REACTION
    working.objective_direction = "max"
    baseline = working.optimize()
    status = str(baseline.status)
    warnings = list(_scenario_warnings(profile, medium))
    scenario = effective_medium(working)
    if status != "optimal":
        warnings.append(f"FVA was skipped because the baseline solver status was {status!r}.")
        return FluxVariabilityResult(
            fungus_id=profile.id,
            status=status,
            objective_value=0.0,
            growth_rate=0.0,
            fraction_of_optimum=fraction_of_optimum,
            ranges={},
            medium=scenario,
            warnings=tuple(warnings),
        )

    selected = (
        list(reaction_ids) if reaction_ids is not None else _default_fva_reactions(working, profile)
    )
    missing = [reaction_id for reaction_id in selected if reaction_id not in working.reactions]
    if missing:
        raise KeyError("Unknown FVA reaction(s): " + ", ".join(missing))
    selected = list(dict.fromkeys(selected))
    frame = flux_variability_analysis(
        working,
        reaction_list=selected,
        fraction_of_optimum=fraction_of_optimum,
        processes=1,
    )
    ranges: dict[str, FluxRange] = {}
    for reaction_id in selected:
        minimum = _clean(frame.loc[reaction_id, "minimum"])
        maximum = _clean(frame.loc[reaction_id, "maximum"])
        if minimum > maximum and minimum - maximum < 1e-7:
            minimum = maximum
        reaction = working.reactions.get_by_id(reaction_id)
        ranges[reaction_id] = FluxRange(
            reaction_id=reaction_id,
            reaction_name=reaction.name,
            minimum=minimum,
            maximum=maximum,
        )

    objective_value = _clean(baseline.objective_value)
    return FluxVariabilityResult(
        fungus_id=profile.id,
        status="optimal",
        objective_value=objective_value,
        growth_rate=objective_value,
        fraction_of_optimum=fraction_of_optimum,
        ranges=ranges,
        medium=scenario,
        warnings=tuple(warnings),
    )


def analyse_sensitivities(
    model_or_fungus: Model | str | FungusProfile,
    medium: Mapping[str, float] | None = None,
    *,
    perturbation_fraction: float = 0.10,
    nutrients: Sequence[str] | None = None,
) -> SensitivityAnalysis:
    """Calculate central finite-difference elasticities in a stable order."""

    if not 0 < perturbation_fraction < 1:
        raise ValueError("perturbation_fraction must be in the interval (0, 1).")

    working, profile = _prepared_model(model_or_fungus, medium)
    scenario = effective_medium(working)
    baseline = run_fba(working)
    if nutrients is None:
        nutrient_ids = [
            nutrient_id for nutrient_id in NUTRIENTS if scenario[nutrient_id] > _EPSILON
        ]
    else:
        nutrient_ids = [get_nutrient(item).id for item in nutrients]
        nutrient_ids = list(dict.fromkeys(nutrient_ids))

    provisional: list[SensitivityEntry] = []
    for nutrient_id in nutrient_ids:
        amount = scenario[nutrient_id]
        if amount > _EPSILON:
            lower_amount = amount * (1.0 - perturbation_fraction)
            upper_amount = min(
                amount * (1.0 + perturbation_fraction),
                profile.uptake_capacities[nutrient_id],
            )
        else:
            lower_amount = 0.0
            upper_amount = min(1.0, profile.uptake_capacities[nutrient_id])

        lower_scenario = dict(scenario)
        upper_scenario = dict(scenario)
        lower_scenario[nutrient_id] = lower_amount
        upper_scenario[nutrient_id] = upper_amount
        lower_growth = run_fba(working, lower_scenario).growth_rate
        upper_growth = run_fba(working, upper_scenario).growth_rate

        if baseline.growth_rate > _EPSILON and amount > _EPSILON:
            relative_step = (upper_amount - lower_amount) / (2.0 * amount)
            elasticity = (
                (upper_growth - lower_growth) / (2.0 * relative_step * baseline.growth_rate)
                if relative_step > _EPSILON
                else 0.0
            )
        elif baseline.growth_rate > _EPSILON:
            elasticity = (upper_growth - lower_growth) / baseline.growth_rate
        else:
            elasticity = 0.0
        elasticity = _clean(elasticity)
        impact = abs(elasticity)
        direction = (
            "Positive"
            if elasticity > _EPSILON
            else "Negative"
            if elasticity < -_EPSILON
            else "Neutral"
        )
        priority = "Critical" if impact >= 0.25 else "Explore" if impact >= 0.05 else "Monitor"
        provisional.append(
            SensitivityEntry(
                rank=0,
                parameter=NUTRIENTS[nutrient_id].name,
                nutrient_id=nutrient_id,
                baseline_amount=amount,
                baseline_growth=baseline.growth_rate,
                lower_amount=lower_amount,
                lower_growth=lower_growth,
                upper_amount=upper_amount,
                upper_growth=upper_growth,
                elasticity=elasticity,
                impact=impact,
                direction=direction,
                priority=priority,
            )
        )

    order = {nutrient_id: index for index, nutrient_id in enumerate(NUTRIENTS)}
    provisional.sort(key=lambda row: (-row.impact, order[row.nutrient_id]))
    rankings = tuple(replace(row, rank=index) for index, row in enumerate(provisional, start=1))
    warnings = list(_RESULT_WARNINGS)
    if baseline.growth_rate <= _EPSILON:
        warnings.append(
            "Baseline biomass was zero, so relative elasticities are undefined and reported as zero."
        )
    return SensitivityAnalysis(
        fungus_id=profile.id,
        baseline_growth=baseline.growth_rate,
        perturbation_fraction=perturbation_fraction,
        rankings=rankings,
        evaluations=1 + 2 * len(rankings),
        medium=scenario,
        warnings=tuple(warnings),
    )


def rank_sensitivities(
    model_or_fungus: Model | str | FungusProfile,
    medium: Mapping[str, float] | None = None,
    *,
    perturbation_fraction: float = 0.10,
    nutrients: Sequence[str] | None = None,
) -> tuple[SensitivityEntry, ...]:
    """Return ranked sensitivity rows (a table-friendly public API)."""

    return analyse_sensitivities(
        model_or_fungus,
        medium,
        perturbation_fraction=perturbation_fraction,
        nutrients=nutrients,
    ).rankings


def medium_cost(medium: Mapping[str, float]) -> float:
    """Return a relative ingredient-cost index for a canonical or aliased medium."""

    values = normalise_medium(medium)
    return sum(
        amount * NUTRIENTS[nutrient_id].cost_per_unit for nutrient_id, amount in values.items()
    )


def _maximum_budget(budgets: Mapping[str, float] | Real | None, fallback: float) -> float:
    if budgets is None:
        return fallback
    if isinstance(budgets, Mapping):
        raw_budget = budgets.get("maximum_cost", budgets.get("budget", fallback))
    else:
        raw_budget = budgets
    try:
        budget = float(raw_budget)
    except (TypeError, ValueError) as exc:
        raise TypeError("budget must be a finite non-negative number") from exc
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    return budget


def _candidate_key(medium: Mapping[str, float]) -> tuple[float, ...]:
    return tuple(round(float(medium.get(nutrient_id, 0.0)), 9) for nutrient_id in NUTRIENTS)


def optimise_medium(
    model_or_fungus: Model | str | FungusProfile,
    candidates: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    budgets: Mapping[str, float] | Real | None = None,
    *,
    medium: Mapping[str, float] | None = None,
    target_fraction: float = 0.98,
) -> MediumOptimizationResult:
    """Search a small sensitivity-guided medium grid under a cost budget.

    A mapping passed as ``candidates`` is treated as the starting medium for
    compatibility with the Streamlit adapter.  A sequence is treated as an
    explicit candidate set.  With no explicit set, the three most sensitive
    active factors are searched at 0.75x, 1.0x and 1.25x.  Among candidates
    reaching ``target_fraction`` of the best feasible growth, the least costly
    one is selected.
    """

    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must be in the interval (0, 1].")
    if medium is not None and isinstance(candidates, Mapping):
        raise TypeError("Pass the starting scenario as either candidates or medium, not both.")

    starting_override = medium
    explicit_candidates: Sequence[Mapping[str, float]] | None = None
    if isinstance(candidates, Mapping):
        starting_override = candidates
    elif candidates is not None:
        explicit_candidates = candidates

    working, profile = _prepared_model(model_or_fungus, starting_override)
    baseline_scenario = effective_medium(working)
    baseline_result = run_fba(working)
    baseline_cost = medium_cost(baseline_scenario)
    maximum_cost = _maximum_budget(budgets, baseline_cost)

    scenarios: list[dict[str, float]] = [dict(baseline_scenario)]
    if explicit_candidates is not None:
        for candidate in explicit_candidates:
            candidate_model, _ = _prepared_model(working, candidate)
            scenarios.append(effective_medium(candidate_model))
    else:
        sensitivities = rank_sensitivities(working)
        factors = [row.nutrient_id for row in sensitivities if row.baseline_amount > _EPSILON][:3]
        for levels in product((0.75, 1.0, 1.25), repeat=len(factors)):
            scenario = dict(baseline_scenario)
            for nutrient_id, level in zip(factors, levels, strict=True):
                scenario[nutrient_id] = min(
                    baseline_scenario[nutrient_id] * level,
                    profile.uptake_capacities[nutrient_id],
                )
            scenarios.append(scenario)

    # Always include a proportional fallback when the requested budget is
    # below the starting cost; this guarantees at least one feasible scenario.
    if baseline_cost > maximum_cost + _EPSILON and baseline_cost > _EPSILON:
        scale = maximum_cost / baseline_cost
        scenarios.append(
            {nutrient_id: amount * scale for nutrient_id, amount in baseline_scenario.items()}
        )

    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for scenario in scenarios:
        unique.setdefault(_candidate_key(scenario), scenario)

    evaluated: list[tuple[dict[str, float], float, float]] = []
    for scenario in unique.values():
        cost = medium_cost(scenario)
        growth = run_fba(working, scenario).growth_rate
        evaluated.append((scenario, growth, cost))

    feasible = [row for row in evaluated if row[2] <= maximum_cost + 1e-7]
    warnings = list(_RESULT_WARNINGS)
    if not feasible:
        feasible = [min(evaluated, key=lambda row: row[2])]
        warnings.append(
            "No candidate met the supplied budget; the least-cost candidate is reported."
        )
        status = "budget_infeasible"
    else:
        status = "optimal"

    best_growth = max(row[1] for row in feasible)
    near_optimal = [row for row in feasible if row[1] + _EPSILON >= target_fraction * best_growth]
    selected = min(near_optimal, key=lambda row: (row[2], -row[1], _candidate_key(row[0])))

    ranked_raw = sorted(feasible, key=lambda row: (-row[1], row[2], _candidate_key(row[0])))
    candidate_rows: list[MediumCandidate] = []
    for rank, (scenario, growth, cost) in enumerate(ranked_raw, start=1):
        score = 100.0 * growth / best_growth if best_growth > _EPSILON else 0.0
        changed = tuple(
            nutrient_id
            for nutrient_id in NUTRIENTS
            if abs(scenario[nutrient_id] - baseline_scenario[nutrient_id]) > _EPSILON
        )
        candidate_rows.append(
            MediumCandidate(
                rank=rank,
                medium=dict(scenario),
                growth_rate=growth,
                estimated_cost=cost,
                score=score,
                changed_nutrients=changed,
            )
        )

    if maximum_cost < baseline_cost - _EPSILON:
        warnings.append("The supplied budget was below the starting-medium cost index.")
    selected_medium, selected_growth, selected_cost = selected
    return MediumOptimizationResult(
        fungus_id=profile.id,
        status=status,
        baseline_medium=dict(baseline_scenario),
        optimized_medium=dict(selected_medium),
        baseline_growth=baseline_result.growth_rate,
        growth_rate=selected_growth,
        objective_value=selected_growth,
        baseline_cost=baseline_cost,
        estimated_cost=selected_cost,
        budget=maximum_cost,
        target_fraction=target_fraction,
        candidates=tuple(candidate_rows),
        evaluations=len(evaluated),
        warnings=tuple(warnings),
    )


def optimize_medium(*args: Any, **kwargs: Any) -> MediumOptimizationResult:
    """US-English alias for :func:`optimise_medium`."""

    return optimise_medium(*args, **kwargs)
