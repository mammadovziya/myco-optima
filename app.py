"""myco-optima Streamlit application.

An engineer-facing interface for exploring fermentation media with constraint-based
modelling.  The UI can use the scientific package when installed, but it remains
fully explorable through a deterministic, explicitly labelled reduced-order demo.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Scientific-core adapter
# -----------------------------------------------------------------------------
# Keep package-specific imports and result-shape translation in this section.
# The Streamlit views below consume plain dicts/dataframes only. If the core API
# changes, this is the only section that should need editing.

CORE_AVAILABLE = False
CORE_IMPORT_NOTE = "Scientific package not installed"
AI_HELPER_AVAILABLE = False

try:
    from myco_optima.catalog import list_fungi as _core_list_fungi
    from myco_optima.catalog import list_nutrients as _core_list_nutrients
    from myco_optima.doe import Factor as _CoreFactor
    from myco_optima.doe import build_sensitivity_guided_design as _core_build_design
    from myco_optima.gene_media import predict_morphology as _core_predict_morphology
    from myco_optima.models import build_model as _core_build_model
    from myco_optima.optimization import optimise_medium as _core_optimise_medium
    from myco_optima.optimization import rank_sensitivities as _core_rank_sensitivities
    from myco_optima.optimization import run_fba as _core_run_fba
    from myco_optima.optimization import run_fva as _core_run_fva

    CORE_AVAILABLE = True
    CORE_IMPORT_NOTE = "Scientific core connected"
except Exception as exc:
    _core_list_fungi = None
    _core_list_nutrients = None
    _CoreFactor = None
    _core_build_design = None
    _core_predict_morphology = None
    _core_build_model = None
    _core_optimise_medium = None
    _core_rank_sensitivities = None
    _core_run_fba = None
    _core_run_fva = None
    CORE_IMPORT_NOTE = f"Demo mode ({type(exc).__name__})"

try:
    from myco_optima.ai import AIUnavailable as _CoreAIUnavailable
    from myco_optima.ai import generate_ai_insight as _core_generate_ai_insight

    AI_HELPER_AVAILABLE = True
except Exception:
    _CoreAIUnavailable = RuntimeError
    _core_generate_ai_insight = None


def _plain(value: Any) -> Any:
    """Convert typed core results into JSON-friendly Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _try_core(call: Callable[[], Any]) -> tuple[Any | None, str | None]:
    """Run a core call without allowing an evolving API to break the UI."""
    if not CORE_AVAILABLE:
        return None, "Scientific core is unavailable; using the demo reduced-order model."
    try:
        return _plain(call()), None
    except Exception as exc:  # core errors are surfaced as a safe, concise notice
        return (
            None,
            f"Core result could not be adapted ({type(exc).__name__}); showing demo output.",
        )


# -----------------------------------------------------------------------------
# Demo reduced-order model and common data
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "myco-optima-wordmark.svg"

FUNGI: dict[str, dict[str, Any]] = {
    "aspergillus_niger": {
        "name": "Aspergillus niger",
        "short": "A. niger",
        "role": "Organic acids & enzymes",
        "temperature": 30.0,
        "ph": 5.0,
        "accent": "#1B8F6B",
        "genes": [
            ("racA", "Rho-family morphology regulator", "Branching / dispersion"),
            ("arfA", "Small GTPase regulator", "Pellet size / secretion"),
            ("creA", "Carbon catabolite repression", "Substrate utilisation"),
            ("pacC", "Ambient pH response", "Branching pattern"),
            ("chsB", "Chitin synthase", "Tip extension"),
        ],
    },
    "aspergillus_oryzae": {
        "name": "Aspergillus oryzae",
        "short": "A. oryzae",
        "role": "Food fermentation & enzymes",
        "temperature": 31.0,
        "ph": 5.5,
        "accent": "#D89A39",
        "genes": [
            ("agsA", "α-1,3-glucan synthase", "Pellet aggregation"),
            ("agsB", "α-1,3-glucan synthase", "Pellet aggregation"),
            ("agsC", "α-1,3-glucan synthase", "Pellet aggregation"),
            ("sphZ", "Galactosaminogalactan pathway", "Hyphal aggregation"),
            ("ugeZ", "Galactosaminogalactan pathway", "Hyphal aggregation"),
            ("nsdC", "Developmental regulator", "Branching / dispersion"),
        ],
    },
    "trichoderma_reesei": {
        "name": "Trichoderma reesei",
        "short": "T. reesei",
        "role": "Cellulases & biorefining",
        "temperature": 28.0,
        "ph": 4.8,
        "accent": "#397D65",
        "genes": [
            ("rac1", "Rho-family morphology regulator", "Branching / dispersion"),
            ("gul1", "Morphogenesis regulator", "Lateral branching"),
            ("xyr1", "Cellulase master regulator", "Carbon response"),
            ("cre1", "Carbon catabolite repression", "Substrate utilisation"),
            ("tmk3", "Stress MAP kinase", "Stress / branching"),
        ],
    },
    "fusarium_venenatum": {
        "name": "Fusarium venenatum",
        "short": "F. venenatum",
        "role": "Mycoprotein & biomass",
        "temperature": 29.0,
        "ph": 6.0,
        "accent": "#8367A8",
        "genes": [
            ("areA", "Candidate nitrogen regulator", "Evidence gap"),
            ("pacC", "Candidate pH-response regulator", "Evidence gap"),
            ("stuA", "Candidate developmental regulator", "Evidence gap"),
            ("chsV", "Candidate chitin synthase", "Evidence gap"),
            ("fmk1", "Candidate MAP kinase", "Evidence gap"),
        ],
    },
}

OBJECTIVES = {
    "Biomass productivity": (
        "Maximise the reduced-order biomass flux under the selected availability constraints."
    ),
}

CARBON_SOURCES = ["Glucose", "Xylose", "Glycerol", "Sucrose", "Maltose"]
NITROGEN_SOURCES = ["Ammonium sulfate", "Urea", "Nitrate"]

NUTRIENT_COST = {
    "Glucose": 0.72,
    "Xylose": 0.86,
    "Glycerol": 0.94,
    "Sucrose": 0.81,
    "Maltose": 1.12,
    "Ammonium sulfate": 0.38,
    "Urea": 0.52,
    "Nitrate": 0.68,
    "KH₂PO₄": 1.05,
    "MgSO₄": 0.74,
    "Trace elements": 3.80,
}

PATHWAY_LABELS = [
    "Biomass synthesis",
    "Glycolysis",
    "TCA cycle",
    "Pentose phosphate",
    "ATP maintenance",
    "Protein secretion",
]


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(_seed(*parts))


def _fungus_catalog() -> tuple[dict[str, dict[str, Any]], str | None]:
    raw, note = _try_core(lambda: _core_list_fungi())
    if not raw:
        return FUNGI, note

    adapted: dict[str, dict[str, Any]] = {}
    records = raw.values() if isinstance(raw, dict) else raw
    try:
        for record in records:
            item = _plain(record)
            fungus_id = str(item.get("id") or item.get("fungus_id") or item.get("slug"))
            if not fungus_id or fungus_id == "None":
                continue
            adapted[fungus_id] = {
                "name": item.get("name")
                or item.get("scientific_name")
                or fungus_id.replace("_", " ").title(),
                "short": item.get("short_name") or item.get("name") or fungus_id,
                "role": item.get("industrial_use")
                or item.get("description")
                or "Industrial fermentation",
                "temperature": float(
                    item.get("temperature") or item.get("optimal_temperature") or 30
                ),
                "ph": float(item.get("ph") or item.get("optimal_ph") or 5.5),
                "accent": item.get("accent") or "#1B8F6B",
                "genes": FUNGI.get(fungus_id, next(iter(FUNGI.values())))["genes"],
            }
    except (TypeError, ValueError, AttributeError):
        return FUNGI, "Core catalogue shape is not recognised; showing the bundled catalogue."
    return (adapted or FUNGI), note


def _demo_media_result(settings: dict[str, Any]) -> dict[str, Any]:
    fungus_id = settings["fungus_id"]
    objective = settings["objective"]
    rng = _rng("media", *settings.values())
    organism_factor = list(FUNGI).index(fungus_id) * 0.025 if fungus_id in FUNGI else 0.04
    objective_factor = list(OBJECTIVES).index(objective) * 0.018

    carbon = float(settings["carbon_g_l"])
    nitrogen = float(settings["nitrogen_g_l"])
    oxygen = float(settings["oxygen_mmol"])
    phosphate = float(settings["phosphate_g_l"])
    target_ratio = 5.6 + list(FUNGI).index(fungus_id) * 0.55 if fungus_id in FUNGI else 6.2
    ratio = carbon / max(nitrogen, 0.1)
    ratio_score = np.exp(-((ratio - target_ratio) ** 2) / 19)
    oxygen_score = 1 - np.exp(-oxygen / 7.5)
    mineral_score = 1 - np.exp(-phosphate / 0.45)
    growth = 0.20 + 0.33 * ratio_score + 0.17 * oxygen_score + 0.07 * mineral_score
    growth += organism_factor - objective_factor + float(rng.normal(0, 0.006))
    growth = float(np.clip(growth, 0.18, 0.94))
    yield_coeff = float(np.clip(0.43 + 0.13 * ratio_score - objective_factor, 0.28, 0.68))
    objective_flux = growth * (1.0 + list(OBJECTIVES).index(objective) * 0.16)

    recommended_carbon = float(np.clip(carbon * (0.92 + 0.12 * ratio_score), 8, 45))
    recommended_nitrogen = float(np.clip(recommended_carbon / target_ratio, 1.5, 8))
    composition = pd.DataFrame(
        [
            (settings["carbon_source"], "Carbon", recommended_carbon, "relative max uptake"),
            (settings["nitrogen_source"], "Nitrogen", recommended_nitrogen, "relative max uptake"),
            ("KH₂PO₄", "Buffer / phosphorus", max(0.45, phosphate * 0.96), "relative max uptake"),
            ("MgSO₄", "Cofactor", 0.42 + 0.06 * ratio_score, "relative max uptake"),
            (
                "Trace elements",
                "Micronutrients",
                0.012 + 0.004 * mineral_score,
                "relative max uptake",
            ),
        ],
        columns=["Ingredient", "Role", "Recommended", "Unit"],
    )
    composition["Estimated cost"] = composition.apply(
        lambda row: row["Recommended"] * NUTRIENT_COST.get(row["Ingredient"], 0.8), axis=1
    )
    total_cost = float(composition["Estimated cost"].sum())
    budget_scale = min(1.0, float(settings["budget"]) / max(total_cost, 0.01))
    if budget_scale < 1.0:
        composition["Recommended"] *= budget_scale
        composition["Estimated cost"] = composition.apply(
            lambda row: row["Recommended"] * NUTRIENT_COST.get(row["Ingredient"], 0.8), axis=1
        )
        total_cost = float(composition["Estimated cost"].sum())
        growth *= 0.72 + 0.28 * budget_scale
        objective_flux = growth * (1.0 + list(OBJECTIVES).index(objective) * 0.16)

    fva_rows = []
    base = [
        objective_flux,
        growth * 1.8,
        growth * 1.35,
        growth * 0.82,
        growth * 0.54,
        growth * 0.45,
    ]
    for pathway, centre, spread in zip(
        PATHWAY_LABELS, base, [0.07, 0.34, 0.27, 0.20, 0.12, 0.18], strict=True
    ):
        jitter = float(rng.uniform(0.92, 1.08))
        low = max(0.0, centre - spread * jitter)
        high = centre + spread * jitter
        fva_rows.append((pathway, low, high, centre))
    fva = pd.DataFrame(fva_rows, columns=["Pathway", "Minimum", "Maximum", "FBA flux"])

    variants = []
    for rank in range(1, 7):
        c_scale = 1 + rng.normal(0, 0.08)
        n_scale = 1 + rng.normal(0, 0.07)
        score = float(np.clip(100 - (rank - 1) * rng.uniform(2.6, 5.8), 63, 100))
        variants.append(
            {
                "Rank": rank,
                "Carbon availability": round(recommended_carbon * c_scale, 2),
                "Nitrogen availability": round(recommended_nitrogen * n_scale, 2),
                "O₂ availability": round(oxygen * (1 + rng.normal(0, 0.06)), 2),
                "Predicted flux": round(objective_flux * score / 100, 3),
                "Relative score": round(score, 1),
            }
        )

    return {
        "source": "Demo reduced-order model",
        "growth_rate": growth,
        "objective_flux": float(objective_flux),
        "yield_coefficient": yield_coeff,
        "estimated_cost": total_cost,
        "composition": composition,
        "fva": fva,
        "alternatives": pd.DataFrame(variants),
        "limiting_nutrient": "Oxygen"
        if oxygen < 8
        else ("Nitrogen" if ratio > target_ratio + 1.2 else "Phosphate"),
    }


def _run_media(settings: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Run and fully adapt the scientific core, or return a labelled demo result."""
    if not CORE_AVAILABLE:
        return _demo_media_result(settings), "Results use the demo reduced-order model."

    medium = {
        settings["carbon_source"]: settings["carbon_g_l"],
        settings["nitrogen_source"]: settings["nitrogen_g_l"],
        "oxygen": settings["oxygen_mmol"],
        "phosphate": settings["phosphate_g_l"],
    }

    def core_call() -> dict[str, Any]:
        optimum = _plain(
            _core_optimise_medium(
                settings["fungus_id"],
                candidates=medium,
                budgets={"maximum_cost": settings["budget"]},
            )
        )
        optimized_medium = optimum["optimized_medium"]
        model = _core_build_model(settings["fungus_id"], optimized_medium)
        fba = _plain(_core_run_fba(model))
        fva = _plain(_core_run_fva(model, fraction_of_optimum=0.95))
        sensitivities = _plain(_core_rank_sensitivities(model))
        nutrients = _plain(_core_list_nutrients())
        return {
            "fba": fba,
            "fva": fva,
            "optimum": optimum,
            "sensitivities": sensitivities,
            "nutrients": nutrients,
        }

    raw, note = _try_core(core_call)
    if not raw:
        return _demo_media_result(settings), note

    fba = raw["fba"]
    fva = raw["fva"]
    optimum = raw["optimum"]
    nutrient_by_id = {item["id"]: item for item in raw["nutrients"]}
    optimized_medium = optimum["optimized_medium"]

    composition_rows = []
    for nutrient_id, amount in optimized_medium.items():
        if float(amount) <= 0:
            continue
        nutrient = nutrient_by_id[nutrient_id]
        composition_rows.append(
            {
                "Ingredient": nutrient["name"],
                "Role": nutrient["category"].title(),
                "Recommended": float(amount),
                "Unit": nutrient["unit"],
                "Estimated cost": float(amount) * float(nutrient["cost_per_unit"]),
            }
        )
    composition = pd.DataFrame(composition_rows)

    fva_rows = []
    for reaction_id, flux_range in fva["ranges"].items():
        minimum = float(flux_range["minimum"])
        maximum = float(flux_range["maximum"])
        fba_flux = float(fba["fluxes"].get(reaction_id, 0.0))
        label = flux_range["reaction_name"]
        if reaction_id.startswith("EX_"):
            minimum, maximum = max(0.0, -maximum), max(0.0, -minimum)
            fba_flux = max(0.0, -fba_flux)
            label = label.replace("exchange", "uptake")
        fva_rows.append(
            {
                "Pathway": label,
                "Reaction": reaction_id,
                "Minimum": minimum,
                "Maximum": maximum,
                "FBA flux": fba_flux,
            }
        )
    fva_frame = pd.DataFrame(fva_rows)

    alternative_rows = []
    for candidate in optimum["candidates"][:6]:
        nonzero = [
            f"{nutrient_by_id[key]['name']} {float(value):.2f}"
            for key, value in candidate["medium"].items()
            if float(value) > 0
        ]
        alternative_rows.append(
            {
                "Rank": int(candidate["rank"]),
                "Growth flux": float(candidate["growth_rate"]),
                "Estimated cost": float(candidate["estimated_cost"]),
                "Relative score": float(candidate["score"]),
                "Changed factors": ", ".join(candidate["changed_nutrients"]) or "Baseline",
                "Medium (max availability)": "; ".join(nonzero),
            }
        )
    alternatives = pd.DataFrame(alternative_rows)

    sensitivities = raw["sensitivities"]
    limiting_nutrient = sensitivities[0]["parameter"] if sensitivities else "No active factor"
    warnings = tuple(dict.fromkeys((*fba.get("warnings", []), *optimum.get("warnings", []))))
    result = {
        "source": "COBRApy reduced-order scientific core",
        "growth_rate": float(fba["growth_rate"]),
        "objective_flux": float(fba["objective_flux"]),
        "yield_coefficient": float(fba["yield_coefficient"]),
        "estimated_cost": float(optimum["estimated_cost"]),
        "composition": composition,
        "fva": fva_frame,
        "alternatives": alternatives,
        "limiting_nutrient": limiting_nutrient,
        "warnings": warnings,
    }
    return result, (
        "Core-connected result. Medium values are relative maximum-availability bounds, "
        "not measured concentrations or cultivation set-points."
    )


def _demo_sensitivity(
    settings: dict[str, Any],
    run_count: int,
    perturbation: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng("sensitivity", *settings.values(), run_count, perturbation)
    parameters = [
        settings["carbon_source"],
        settings["nitrogen_source"],
        "Oxygen uptake",
        "pH",
        "Temperature",
        "KH₂PO₄",
        "MgSO₄",
        "Agitation",
    ]
    baseline = np.array([0.48, 0.35, 0.62, -0.18, 0.21, 0.14, 0.08, 0.27])
    elasticity = baseline + rng.normal(0, 0.045, len(parameters))
    sensitivity = pd.DataFrame(
        {
            "Parameter": parameters,
            "Elasticity": elasticity,
            "Impact": np.abs(elasticity),
            "Direction": np.where(elasticity >= 0, "Positive", "Negative"),
            "Priority": pd.qcut(np.abs(elasticity), 3, labels=["Monitor", "Explore", "Critical"]),
        }
    ).sort_values("Impact", ascending=False, ignore_index=True)

    if run_count == 15:
        # Exact three-factor Box–Behnken follow-up: 12 edge runs + 3 centres.
        coded = np.array(
            [
                (-1, -1, 0),
                (-1, 1, 0),
                (1, -1, 0),
                (1, 1, 0),
                (-1, 0, -1),
                (-1, 0, 1),
                (1, 0, -1),
                (1, 0, 1),
                (0, -1, -1),
                (0, -1, 1),
                (0, 1, -1),
                (0, 1, 1),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
            ],
            dtype=float,
        )
        carbon = float(settings["carbon_g_l"]) * (1 + perturbation * coded[:, 0])
        nitrogen = float(settings["nitrogen_g_l"]) * (1 + perturbation * coded[:, 1])
        oxygen = float(settings["oxygen_mmol"]) * (1 + perturbation * coded[:, 2])
        design_type = ["Edge"] * 12 + [
            "Centre replicate 1",
            "Centre replicate 2",
            "Centre replicate 3",
        ]
    else:
        # Deterministic stratified design for a user-requested non-standard size.
        coded = np.column_stack([rng.permutation(np.linspace(-1, 1, run_count)) for _ in range(3)])
        carbon = float(settings["carbon_g_l"]) * (1 + perturbation * coded[:, 0])
        nitrogen = float(settings["nitrogen_g_l"]) * (1 + perturbation * coded[:, 1])
        oxygen = float(settings["oxygen_mmol"]) * (1 + perturbation * coded[:, 2])
        design_type = ["Stratified"] * run_count

    reference = FUNGI.get(settings["fungus_id"], FUNGI["aspergillus_niger"])
    ph = np.full(run_count, reference["ph"])
    temperature = np.full(run_count, reference["temperature"])
    ratio = carbon / nitrogen
    predicted = 0.42 + 0.19 * np.exp(-((ratio - 6.0) ** 2) / 12) + 0.012 * oxygen
    predicted -= 0.018 * np.abs(temperature - reference["temperature"])
    doe = pd.DataFrame(
        {
            "Run": np.arange(1, run_count + 1),
            "Design point": design_type,
            "Carbon availability": carbon.round(2),
            "Nitrogen availability": nitrogen.round(2),
            "O₂ availability": oxygen.round(2),
            "Phosphate availability": np.full(run_count, settings["phosphate_g_l"]).round(2),
            "pH": ph.round(2),
            "Temperature (°C)": temperature.round(1),
            "Predicted response": predicted.round(3),
        }
    ).sort_values("Predicted response", ascending=False, ignore_index=True)
    doe.insert(0, "Priority", np.arange(1, run_count + 1))
    return sensitivity, doe


def _run_sensitivity(
    settings: dict[str, Any],
    run_count: int,
    perturbation: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    demo_sensitivity, demo_doe = _demo_sensitivity(settings, run_count, perturbation)
    if not CORE_AVAILABLE:
        return demo_sensitivity, demo_doe, "Sensitivity rankings are illustrative demo outputs."

    carbon_id = settings["carbon_source"].lower()
    nitrogen_id = {
        "Ammonium sulfate": "ammonium",
        "Urea": "urea",
        "Nitrate": "nitrate",
    }[settings["nitrogen_source"]]
    baselines = {
        carbon_id: settings["carbon_g_l"],
        nitrogen_id: settings["nitrogen_g_l"],
        "oxygen": settings["oxygen_mmol"],
        "phosphate": settings["phosphate_g_l"],
    }

    def core_call() -> dict[str, Any]:
        rankings = _core_rank_sensitivities(
            settings["fungus_id"],
            baselines,
            perturbation_fraction=perturbation,
            nutrients=list(baselines),
        )
        sensitivity_by_id = {row.nutrient_id: row.elasticity for row in rankings}
        factors = [
            _CoreFactor(
                nutrient_id,
                max(0.0, amount * (1.0 - perturbation)),
                amount * (1.0 + perturbation),
                "relative maximum uptake",
            )
            for nutrient_id, amount in baselines.items()
        ]
        design = _core_build_design(factors, sensitivity_by_id)
        design_frame = design.runs.copy()
        for nutrient_id, amount in baselines.items():
            if nutrient_id not in design_frame:
                design_frame[nutrient_id] = amount

        responses = []
        for row in design_frame.to_dict(orient="records"):
            scenario = {nutrient_id: float(row[nutrient_id]) for nutrient_id in baselines}
            responses.append(_core_run_fba(settings["fungus_id"], scenario).growth_rate)
        design_frame["Predicted response"] = responses
        design_frame = design_frame.rename(
            columns={
                "run": "Run",
                "design_role": "Design point",
                carbon_id: "Carbon availability",
                nitrogen_id: "Nitrogen availability",
                "oxygen": "O₂ availability",
                "phosphate": "Phosphate availability",
            }
        ).sort_values("Predicted response", ascending=False, ignore_index=True)
        design_frame.insert(0, "Priority", np.arange(1, len(design_frame) + 1))
        return {
            "rankings": rankings,
            "design": design_frame.to_dict(orient="records"),
            "retained_factors": design.retained_factors,
        }

    raw, note = _try_core(core_call)
    if not raw:
        return demo_sensitivity, demo_doe, note
    try:
        frame = pd.DataFrame(raw["rankings"])
        rename = {
            "parameter": "Parameter",
            "elasticity": "Elasticity",
            "impact": "Impact",
            "direction": "Direction",
            "priority": "Priority",
        }
        frame = frame.rename(columns=rename)
        if "Impact" not in frame and "Elasticity" in frame:
            frame["Impact"] = frame["Elasticity"].abs()
        if {"Parameter", "Elasticity", "Impact"}.issubset(frame.columns):
            frame["Direction"] = np.where(frame["Elasticity"] >= 0, "Positive", "Negative")
            if "Priority" not in frame:
                frame["Priority"] = "Core ranked"
            doe = pd.DataFrame(raw["design"])
            retained = ", ".join(raw["retained_factors"])
            return (
                frame.sort_values("Impact", ascending=False),
                doe,
                f"Core-connected sensitivity at ±{perturbation:.0%}; retained factors: {retained}.",
            )
    except (TypeError, ValueError, KeyError):
        pass
    return (
        demo_sensitivity,
        demo_doe,
        "Core sensitivities could not be charted; showing demo rankings.",
    )


def _demo_gene_prediction(
    fungus_id: str,
    gene_states: dict[str, str],
    carbon: float,
    nitrogen: float,
    oxygen: float,
    carbon_source: str = "Glucose",
) -> dict[str, Any]:
    """Apply a small, inspectable rule set to produce qualitative support scores."""
    classes = ["Compact pellets", "Loose pellets", "Dispersed hyphae", "Dense clumps"]
    scores = np.array([0.58, 0.54, 0.50, 0.40], dtype=float)
    cn_ratio = carbon / max(nitrogen, 0.1)
    # Process/media terms remain intentionally modest; they are context flags,
    # not experimentally fitted morphology relationships.
    scores[1] += max(0.0, oxygen - 10.0) * 0.012
    scores[2] += max(0.0, oxygen - 10.0) * 0.015
    scores[3] += max(0.0, cn_ratio - 7.0) * 0.022

    rule_notes: dict[str, tuple[str, str, float]] = {}

    def apply_rule(
        gene: str, effects: dict[int, float], tendency: str, scope: str, strength: float
    ) -> None:
        state = gene_states.get(gene, "Native")
        if state == "Native":
            return
        direction = 1.0 if state == "Knock-down" else -0.55
        for class_index, effect in effects.items():
            scores[class_index] += effect * direction
        rule_notes[gene] = (tendency, scope, strength)

    if fungus_id == "aspergillus_niger":
        apply_rule(
            "racA",
            {0: -0.28, 2: 0.62},
            "Hyperbranching / dispersion tendency",
            "A. niger-specific rule",
            0.90,
        )
        apply_rule(
            "arfA",
            {0: -0.12, 1: 0.32, 2: 0.18},
            "Smaller, less aggregated growth tendency",
            "Strongest under glucose",
            0.72,
        )
    elif fungus_id == "trichoderma_reesei":
        apply_rule(
            "rac1",
            {0: -0.22, 2: 0.55},
            "Hyperbranching / dispersion tendency",
            "T. reesei-specific rule",
            0.88,
        )
        apply_rule(
            "gul1",
            {1: 0.26, 2: 0.40, 3: -0.15},
            "More lateral branching; lower viscosity tendency",
            "T. reesei-specific rule",
            0.78,
        )
    elif fungus_id == "aspergillus_oryzae":
        for gene in ("agsA", "agsB", "agsC"):
            apply_rule(
                gene,
                {0: -0.13, 1: 0.22},
                "Smaller pellet tendency",
                "A. oryzae-specific rule",
                0.76,
            )
        apply_rule(
            "sphZ",
            {0: -0.12, 2: 0.30},
            "Lower aggregation tendency",
            "A. oryzae-specific rule",
            0.72,
        )
        apply_rule(
            "nsdC",
            {0: -0.18, 2: 0.42},
            "Hyperbranching / dispersed-clump tendency",
            "A. oryzae-specific rule",
            0.70,
        )
        if all(gene_states.get(gene) == "Knock-down" for gene in ("agsA", "agsB", "agsC", "sphZ")):
            scores[0] -= 0.25
            scores[2] += 0.55
    # F. venenatum deliberately has no species-specific gene rule: candidate
    # regulators are displayed only to make the evidence gap explicit.

    scores = np.clip(scores, 0.05, None)
    support = scores / scores.max()
    morphology = classes[int(support.argmax())]
    ordered = np.sort(support)
    separation = float(ordered[-1] - ordered[-2])

    genes = FUNGI.get(fungus_id, FUNGI["aspergillus_niger"])["genes"]
    rows = []
    for index, (gene, function, _phenotype) in enumerate(genes):
        state = gene_states.get(gene, "Native")
        media_driver = ["C:N ratio", "Dissolved oxygen", "pH", "Nitrogen source", "Carbon source"][
            index
        ]
        if gene in rule_notes:
            direction, scope, interaction = rule_notes[gene]
            evidence = "Moderate" if interaction < 0.8 else "Higher"
        elif fungus_id == "fusarium_venenatum":
            direction = "Insufficient species-specific evidence"
            scope = "Candidate only; no deterministic gene effect applied"
            interaction = 0.12
            evidence = "Low"
        elif state == "Native":
            direction = "Native baseline; no perturbation rule applied"
            scope = "Perturb a supported regulator to explore"
            interaction = 0.20
            evidence = "Not scored"
        else:
            direction = "No morphology effect encoded"
            scope = "Regulatory context only"
            interaction = 0.18
            evidence = "Exploratory"
        rows.append((gene, state, function, media_driver, direction, scope, evidence, interaction))
    interactions = pd.DataFrame(
        rows,
        columns=[
            "Gene",
            "State",
            "Function",
            "Strongest media link",
            "Predicted tendency",
            "Scope",
            "Evidence",
            "Interaction score",
        ],
    ).sort_values("Interaction score", ascending=False)

    return {
        "source": "Demo reduced-order model",
        "morphology": morphology,
        "confidence": "low",
        "separation": separation,
        "score_table": pd.DataFrame({"Morphology": classes, "Relative support": support}),
        "interactions": interactions,
        "trace": {
            "carbon_source": carbon_source,
            "c_n_ratio": cn_ratio,
            "oxygen_context": oxygen,
            "applied_gene_rules": sorted(rule_notes),
        },
    }


def _run_gene_prediction(
    fungus_id: str,
    gene_states: dict[str, str],
    carbon: float,
    nitrogen: float,
    oxygen: float,
    carbon_source: str = "Glucose",
) -> tuple[dict[str, Any], str | None]:
    demo = _demo_gene_prediction(fungus_id, gene_states, carbon, nitrogen, oxygen, carbon_source)
    if not CORE_AVAILABLE:
        return demo, "Morphology is a demo hypothesis, not a validated phenotype prediction."
    medium = {carbon_source: carbon, "nitrogen": nitrogen, "oxygen": oxygen}
    raw, note = _try_core(lambda: _core_predict_morphology(fungus_id, medium, gene_states))
    if not raw or not isinstance(raw, dict):
        return demo, note

    support_scores = {str(key): float(value) for key, value in raw["support_scores"].items()}
    maximum_support = max(support_scores.values(), default=1.0) or 1.0
    relative_support = {key: value / maximum_support for key, value in support_scores.items()}
    ordered = sorted(relative_support.values())
    separation = ordered[-1] - ordered[-2] if len(ordered) > 1 else 0.0

    interaction_rows = []
    for interaction in raw["interaction_trace"]:
        effects = interaction.get("effects", {})
        effect_text = (
            ", ".join(f"{name} {float(value):+g}" for name, value in effects.items())
            or "No encoded effect"
        )
        interaction_rows.append(
            {
                "Gene": interaction["gene"],
                "State": interaction["state"],
                "Function": interaction["explanation"],
                "Strongest media link": interaction["condition"],
                "Predicted tendency": effect_text,
                "Scope": interaction["rule_id"],
                "Evidence": str(interaction["confidence"]).title(),
                "Applied": bool(interaction["applied"]),
                "Interaction score": sum(abs(float(value)) for value in effects.values()),
            }
        )
    interaction_columns = [
        "Gene",
        "State",
        "Function",
        "Strongest media link",
        "Predicted tendency",
        "Scope",
        "Evidence",
        "Applied",
        "Interaction score",
    ]
    interactions = pd.DataFrame(interaction_rows, columns=interaction_columns)
    if not interactions.empty:
        interactions = interactions.sort_values(
            ["Applied", "Interaction score"], ascending=[False, False]
        )

    result = {
        "source": "Deterministic gene–media scientific core",
        "morphology": str(raw["predicted_morphology"]),
        "confidence": str(raw["confidence"]),
        "separation": float(separation),
        "score_table": pd.DataFrame(
            {
                "Morphology": list(relative_support),
                "Relative support": list(relative_support.values()),
            }
        ),
        "interactions": interactions,
        "trace": {
            "carbon_source": carbon_source,
            "relative_c_n": carbon / max(nitrogen, 0.1),
            "oxygen_context": oxygen,
            "confidence": raw["confidence"],
            "drivers": raw["drivers"],
            "insufficient_evidence": raw["insufficient_evidence"],
            "rules": raw["interaction_trace"],
            "warnings": raw["warnings"],
        },
    }
    return result, (
        "Core-connected qualitative rule result. Support scores are relative rule scores, "
        "not probabilities or validated phenotypes."
    )


# -----------------------------------------------------------------------------
# Presentation helpers
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="myco-optima · Fermentation optimisation",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "myco-optima · Edinburgh BioHackathon 2026"},
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #10251f;
            --muted: #5c7169;
            --forest: #082a24;
            --mint: #70e0b3;
            --line: rgba(15, 65, 52, 0.12);
        }
        .stApp { background: #f6f8f4; }
        [data-testid="stSidebar"] { background: #eaf1ec; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stImage"] img { max-width: 230px; }
        [data-testid="stHeader"] { background: rgba(246, 248, 244, 0.82); backdrop-filter: blur(12px); }
        .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1450px; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
        h1 { font-size: clamp(2.15rem, 4vw, 4rem) !important; line-height: 1.02 !important; }
        h2 { margin-top: 0.4rem !important; }
        p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
        .hero {
            position: relative; overflow: hidden; min-height: 285px; padding: 42px 46px;
            border-radius: 30px; color: #f5fbf7;
            background: radial-gradient(circle at 89% 18%, rgba(112,224,179,.28), transparent 27%),
                        linear-gradient(135deg, #071f1b 0%, #0b3c31 72%, #155c48 100%);
            box-shadow: 0 22px 70px rgba(8,42,36,.16); margin-bottom: 1.3rem;
        }
        .hero:after { content:""; position:absolute; width:190px; height:190px; right:7%; bottom:-120px;
            border:1px solid rgba(185,245,212,.24); border-radius:50%; box-shadow: 0 0 0 28px rgba(185,245,212,.05), 0 0 0 60px rgba(185,245,212,.035); }
        .eyebrow { display:inline-flex; align-items:center; gap:8px; text-transform:uppercase; letter-spacing:.13em;
            font-weight:750; font-size:.72rem; color:#b9f5d4; margin-bottom:16px; }
        .eyebrow:before { content:""; width:7px; height:7px; border-radius:50%; background:#f2c66d; }
        .hero h1 { color:#f7fbf8 !important; max-width:800px; margin:.1rem 0 .8rem !important; }
        .hero p { color:#cae4da; font-size:1.06rem; max-width:720px; line-height:1.65; margin:0; }
        .hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:24px; }
        .hero-pill { border:1px solid rgba(185,245,212,.24); background:rgba(255,255,255,.06); color:#e4f7ec;
            border-radius:999px; padding:7px 12px; font-size:.78rem; font-weight:650; }
        .section-kicker { color:#1b8f6b; font-weight:750; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; }
        .soft-card, .kpi-card, .fungus-card {
            background:rgba(255,255,255,.76); border:1px solid var(--line); border-radius:20px;
            padding:20px; box-shadow:0 8px 30px rgba(8,42,36,.045); height:100%;
        }
        .kpi-card .label { color:#61766d; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
        .kpi-card .value { color:#10251f; font-size:1.75rem; font-weight:780; letter-spacing:-.035em; margin:7px 0 2px; }
        .kpi-card .delta { color:#1b8f6b; font-size:.8rem; font-weight:650; }
        .fungus-card .monogram { width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
            background:#e0f5e9; color:#11644d; font-size:1.15rem; margin-bottom:14px; }
        .fungus-card h3 { font-size:1.03rem; margin:0 0 6px; }
        .fungus-card p { font-size:.84rem; margin:0; line-height:1.45; }
        .step { display:flex; gap:14px; align-items:flex-start; padding:14px 0; border-bottom:1px solid var(--line); }
        .step:last-child { border-bottom:0; }
        .step-number { flex:none; width:28px; height:28px; display:grid; place-items:center; border-radius:9px;
            background:#dff4e8; color:#126247; font-size:.77rem; font-weight:800; }
        .step strong { color:#173229; display:block; margin-bottom:2px; }
        .step span { color:#687d74; font-size:.84rem; }
        .status-strip { display:flex; align-items:flex-start; gap:12px; border:1px solid #bcdccb; background:#edf8f1;
            border-radius:15px; padding:13px 15px; margin:.5rem 0 1.2rem; }
        .status-dot { width:9px; height:9px; border-radius:50%; background:#22a879; margin-top:6px; flex:none; box-shadow:0 0 0 5px rgba(34,168,121,.10); }
        .status-strip strong { color:#174c3c; font-size:.86rem; }
        .status-strip span { color:#587167; font-size:.8rem; display:block; margin-top:2px; }
        .warning-strip { border-left:4px solid #dda443; background:#fff8e9; color:#65502d; padding:13px 16px;
            border-radius:0 13px 13px 0; margin:1rem 0; font-size:.86rem; line-height:1.5; }
        .tag { display:inline-block; padding:5px 9px; border-radius:999px; background:#e4f4ea; color:#17684f;
            font-size:.7rem; font-weight:750; letter-spacing:.03em; margin-right:5px; }
        .model-card { border-radius:20px; padding:22px; color:#e7f6ee; background:linear-gradient(145deg,#0b3028,#124b3d); }
        .model-card .big { font-size:1.6rem; font-weight:760; margin:.35rem 0; color:#fff; }
        .model-card p { color:#bad8cc; margin:.2rem 0; }
        [data-testid="stMetric"] { border:1px solid var(--line); background:rgba(255,255,255,.75); padding:15px 17px; border-radius:17px; }
        [data-testid="stMetricValue"] { color:#10251f; font-weight:760; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius:12px; min-height:42px; font-weight:680; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background:#176f56; border-color:#176f56; color:#fff !important;
            box-shadow:0 8px 20px rgba(27,143,107,.18);
        }
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
            background:#105c47; border-color:#105c47; color:#fff !important;
        }
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stTextArea textarea {
            border-color:rgba(15,65,52,.15); border-radius:11px;
        }
        hr { border-color:var(--line) !important; }
        @media (max-width: 780px) {
            .block-container { padding-top:1rem; }
            .hero { padding:30px 25px; min-height:auto; border-radius:22px; }
            .hero p { font-size:.95rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(label: str, value: str, delta: str) -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="label">{label}</div><div class="value">{value}</div><div class="delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def _page_intro(kicker: str, title: str, body: str) -> None:
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(body)


def _download_pair(
    frame: pd.DataFrame, filename: str, json_payload: dict[str, Any] | None = None
) -> None:
    col_csv, col_json, spacer = st.columns([1, 1, 4])
    with col_csv:
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename}.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_json:
        payload = json_payload or {"records": frame.to_dict(orient="records")}
        st.download_button(
            "Download JSON",
            json.dumps(payload, indent=2, default=str),
            file_name=f"{filename}.json",
            mime="application/json",
            width="stretch",
        )


def _model_notice(source: str, note: str | None = None) -> None:
    is_demo = "demo" in source.lower()
    dot = "#D89A39" if is_demo else "#22A879"
    detail = note or (
        "Illustrative, deterministic outputs for interface exploration."
        if is_demo
        else "Results were returned by the installed scientific core."
    )
    st.markdown(
        f'<div class="status-strip"><span class="status-dot" style="background:{dot}"></span><div><strong>{source}</strong><span>{detail}</span></div></div>',
        unsafe_allow_html=True,
    )


def _plot_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui", color="#405a50", size=12),
        hoverlabel=dict(bgcolor="#082a24", font_color="white", bordercolor="#082a24"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="rgba(15,65,52,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(15,65,52,.08)", zeroline=False)
    return fig


def _shared_settings() -> dict[str, Any]:
    return {
        "fungus_id": st.session_state.fungus_id,
        "objective": st.session_state.objective,
        "carbon_source": st.session_state.get("carbon_source", "Glucose"),
        "nitrogen_source": st.session_state.get("nitrogen_source", "Ammonium sulfate"),
        "carbon_g_l": float(st.session_state.get("carbon_g_l", 24.0)),
        "nitrogen_g_l": float(st.session_state.get("nitrogen_g_l", 4.0)),
        "oxygen_mmol": float(st.session_state.get("oxygen_mmol", 10.0)),
        "phosphate_g_l": float(st.session_state.get("phosphate_g_l", 1.1)),
        "budget": float(st.session_state.get("budget", 30.0)),
    }


def _serialise_media(result: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "tool": "myco-optima",
            "model_source": result["source"],
            "disclaimer": "Decision-support hypothesis; not a validated cultivation recipe.",
        },
        "settings": {
            "fungus_id": settings["fungus_id"],
            "objective": settings["objective"],
            "carbon_source": settings["carbon_source"],
            "nitrogen_source": settings["nitrogen_source"],
            "carbon_max_availability": settings["carbon_g_l"],
            "nitrogen_max_availability": settings["nitrogen_g_l"],
            "oxygen_max_availability": settings["oxygen_mmol"],
            "phosphate_max_availability": settings["phosphate_g_l"],
            "maximum_cost_index": settings["budget"],
        },
        "summary": {
            key: result[key]
            for key in [
                "growth_rate",
                "objective_flux",
                "yield_coefficient",
                "estimated_cost",
                "limiting_nutrient",
            ]
        },
        "composition": result["composition"].to_dict(orient="records"),
        "fva": result["fva"].to_dict(orient="records"),
        "warnings": list(result.get("warnings", [])),
    }
    return payload


_inject_css()

catalog, catalog_note = _fungus_catalog()
if "fungus_id" not in st.session_state or st.session_state.fungus_id not in catalog:
    st.session_state.fungus_id = next(iter(catalog))
if "objective" not in st.session_state:
    st.session_state.objective = "Biomass productivity"


# Sidebar ---------------------------------------------------------------------

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width="stretch")
    else:
        st.markdown("## 🍄 myco-optima")
    st.caption("Fungal Fermentation Optimisation Tool · 2026")
    st.markdown("---")

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Media Optimizer",
            "Sensitivity & DoE",
            "Gene–Media Explorer",
            "AI Interpretation & About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### Shared scenario")
    fungus_name_to_id = {item["name"]: fungus_id for fungus_id, item in catalog.items()}
    chosen_name = st.selectbox(
        "Organism",
        list(fungus_name_to_id),
        index=list(fungus_name_to_id).index(catalog[st.session_state.fungus_id]["name"]),
    )
    st.session_state.fungus_id = fungus_name_to_id[chosen_name]
    st.session_state.objective = "Biomass productivity"
    st.markdown("**Optimisation objective**  ")
    st.caption("Biomass productivity · reduced-order biomass flux")
    process_mode = "Aerobic steady-state surrogate"
    st.caption("Process scope · aerobic steady-state surrogate")

    st.markdown("---")
    badge_label = "CORE CONNECTED" if CORE_AVAILABLE else "DEMO MODEL"
    badge_colour = "#17684f" if CORE_AVAILABLE else "#9b6822"
    badge_bg = "#dff4e8" if CORE_AVAILABLE else "#fff0cd"
    st.markdown(
        f'<span class="tag" style="color:{badge_colour};background:{badge_bg}">{badge_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(CORE_IMPORT_NOTE)
    st.caption("v0.1 · Edinburgh BioHackathon 2026")


# Pages -----------------------------------------------------------------------

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Engineer-first decision support</div>
          <h1>Better fungal media, fewer experimental rounds.</h1>
          <p>Explore how carbon, nitrogen, oxygen and genotype may shape fermentation performance—then turn the most informative regions into a focused wet-lab plan.</p>
          <div class="hero-pills">
            <span class="hero-pill">Flux Balance Analysis</span>
            <span class="hero-pill">Flux Variability Analysis</span>
            <span class="hero-pill">Sensitivity-led DoE</span>
            <span class="hero-pill">Gene–media hypotheses</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _card("Fungal chassis", "4", "Industrial reference organisms")
    with col2:
        _card("Suggested DoE", "~15", "From an ~80-run broad screen")
    with col3:
        _card("Analysis modes", "FBA + FVA", "Growth and solution flexibility")
    with col4:
        _card("Workflow", "Minutes", "From medium to ranked hypotheses")

    st.markdown("### Choose a chassis, keep the workflow consistent")
    fungus_cols = st.columns(4)
    icons = ["◒", "◎", "⌁", "◉"]
    for column, icon, (_, fungus) in zip(fungus_cols, icons, catalog.items(), strict=True):
        with column:
            st.markdown(
                f'<div class="fungus-card"><div class="monogram">{icon}</div><h3><i>{fungus["name"]}</i></h3><p>{fungus["role"]}<br><br>Reference: {fungus["temperature"]:.0f} °C · pH {fungus["ph"]:.1f}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    workflow_col, context_col = st.columns([1.35, 1])
    with workflow_col:
        st.markdown("### One traceable path from model to bench")
        st.markdown(
            """
            <div class="soft-card">
              <div class="step"><div class="step-number">01</div><div><strong>Define the operating envelope</strong><span>Select an organism, objective, substrates, oxygen ceiling and cost boundary.</span></div></div>
              <div class="step"><div class="step-number">02</div><div><strong>Inspect feasible flux</strong><span>Compare a point FBA solution with FVA ranges before trusting a single optimum.</span></div></div>
              <div class="step"><div class="step-number">03</div><div><strong>Prioritise uncertainty</strong><span>Rank influential variables and generate a compact, information-rich DoE.</span></div></div>
              <div class="step"><div class="step-number">04</div><div><strong>Form a morphology hypothesis</strong><span>Explore gene–media interactions as testable hypotheses, never as measured phenotypes.</span></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with context_col:
        st.markdown("### Current scenario")
        current = catalog[st.session_state.fungus_id]
        st.markdown(
            f"""
            <div class="model-card">
              <span class="tag">{process_mode}</span>
              <div class="big"><i>{current["name"]}</i></div>
              <p>{st.session_state.objective}</p>
              <p>Reference operating point: {current["temperature"]:.0f} °C · pH {current["ph"]:.1f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="warning-strip"><strong>Scope:</strong> myco-optima is a reduced-order decision-support tool. Outputs are model-derived hypotheses and require strain-specific calibration, safety review and wet-lab validation.</div>',
            unsafe_allow_html=True,
        )

elif page == "Media Optimizer":
    _page_intro(
        "Constraint-based design",
        "Media Optimizer",
        "Set the practical limits. The tool scores a feasible medium, exposes alternate optima, and shows which fluxes remain flexible.",
    )
    _model_notice(
        "COBRApy scientific core" if CORE_AVAILABLE else "Demo reduced-order model", catalog_note
    )

    form_col, explainer_col = st.columns([1.6, 1])
    with form_col, st.form("media_form"):
        source_a, source_b = st.columns(2)
        with source_a:
            carbon_source = st.selectbox("Carbon source", CARBON_SOURCES, key="carbon_source")
            carbon_g_l = st.slider(
                "Carbon maximum availability (model units)",
                5.0,
                50.0,
                24.0,
                0.5,
                key="carbon_g_l",
                help="Maximum relative uptake bound used by the reduced-order model; not a flask concentration.",
            )
            oxygen_mmol = st.slider(
                "O₂ maximum availability (model units)", 2.0, 20.0, 10.0, 0.5, key="oxygen_mmol"
            )
        with source_b:
            nitrogen_source = st.selectbox(
                "Nitrogen source", NITROGEN_SOURCES, key="nitrogen_source"
            )
            nitrogen_g_l = st.slider(
                "Nitrogen maximum availability (model units)",
                0.5,
                12.0,
                4.0,
                0.25,
                key="nitrogen_g_l",
                help="Maximum relative uptake bound; not a measured nitrogen concentration.",
            )
            phosphate_g_l = st.slider(
                "Phosphate maximum availability (model units)",
                0.1,
                3.0,
                1.1,
                0.1,
                key="phosphate_g_l",
            )
        budget = st.slider("Maximum medium cost index", 8.0, 60.0, 30.0, 1.0, key="budget")
        st.caption(
            "The optimiser chooses the least-cost candidate within 98% of the best feasible biomass flux."
        )
        submitted = st.form_submit_button("Run media optimisation", type="primary", width="stretch")
    with explainer_col:
        st.markdown("#### What the model is balancing")
        st.markdown(
            """
            <div class="soft-card">
              <div class="step"><div class="step-number">C</div><div><strong>Carbon economy</strong><span>Enough substrate for the objective without assuming infinite uptake.</span></div></div>
              <div class="step"><div class="step-number">N</div><div><strong>Nitrogen sufficiency</strong><span>Avoids a formulation that looks cheap but caps biomass or protein.</span></div></div>
              <div class="step"><div class="step-number">O₂</div><div><strong>Transfer realism</strong><span>Treats oxygen as an explicit constraint in aerobic operation.</span></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Every slider is a relative maximum-availability bound. Converting a real recipe into these values requires measured uptake data or fitted kinetics."
        )
        st.caption(
            f"Optimising for: {st.session_state.objective}. {OBJECTIVES[st.session_state.objective]}"
        )

    settings = _shared_settings()
    if (
        submitted
        or "media_result" not in st.session_state
        or st.session_state.get("media_signature") != settings
    ):
        result, result_note = _run_media(settings)
        st.session_state.media_result = result
        st.session_state.media_note = result_note
        st.session_state.media_signature = settings.copy()
    result = st.session_state.media_result

    st.markdown("### Recommended operating point")
    _model_notice(result["source"], st.session_state.get("media_note"))
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(
            "Predicted growth flux",
            f"{result['growth_rate']:.3f} model units",
            help="Pseudo-biomass flux from the reduced-order model; not a calibrated h⁻¹ growth rate.",
        )
    with metric_cols[1]:
        st.metric(
            "Objective flux",
            f"{result['objective_flux']:.3f}",
            help="Relative flux through the biomass pseudo-reaction.",
        )
    with metric_cols[2]:
        st.metric(
            "Model yield ratio",
            f"{result['yield_coefficient']:.2f}",
            help="Illustrative biomass-to-carbon-equivalent ratio; not a measured g/g yield.",
        )
    with metric_cols[3]:
        st.metric(
            "Estimated medium cost index",
            f"{result['estimated_cost']:.2f}",
            help="Uses bundled relative ingredient costs.",
        )

    comp_col, flux_col = st.columns([1, 1.15])
    with comp_col:
        st.markdown("#### Optimized availability bounds")
        display_composition = result["composition"].copy()
        display_composition["Recommended"] = display_composition["Recommended"].round(3)
        display_composition["Estimated cost"] = display_composition["Estimated cost"].round(2)
        st.dataframe(display_composition, hide_index=True, width="stretch")
        st.caption(f"Highest local sensitivity: **{result['limiting_nutrient']}**")
    with flux_col:
        st.markdown("#### FVA solution envelope")
        fva = result["fva"].sort_values("FBA flux")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=fva["Pathway"],
                x=fva["Maximum"] - fva["Minimum"],
                base=fva["Minimum"],
                orientation="h",
                name="Feasible range",
                marker_color="#bfe7d2",
                customdata=fva["Maximum"],
                hovertemplate="%{y}<br>range: %{base:.3f}–%{customdata:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=fva["Pathway"],
                x=fva["FBA flux"],
                mode="markers",
                marker=dict(color="#0d6d52", size=10, line=dict(color="white", width=2)),
                name="FBA point",
                hovertemplate="%{y}<br>FBA: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title="Relative flux", yaxis_title=None)
        st.plotly_chart(_plot_layout(fig, 375), width="stretch", config={"displayModeBar": False})

    with st.expander("Compare the six highest-scoring alternatives", expanded=True):
        alternatives = result["alternatives"].copy()
        st.dataframe(
            alternatives,
            hide_index=True,
            width="stretch",
        )

    _download_pair(
        result["composition"],
        f"myco-optima_{settings['fungus_id']}_medium",
        _serialise_media(result, settings),
    )
    st.markdown(
        '<div class="warning-strip"><strong>Before cultivation:</strong> these bounds are not recipe concentrations. Calibrate them to uptake data, then confirm units, solubility, osmolarity, oxygen-transfer capacity, strain auxotrophies and local biosafety requirements.</div>',
        unsafe_allow_html=True,
    )

elif page == "Sensitivity & DoE":
    _page_intro(
        "Experiment prioritisation",
        "Sensitivity & Design of Experiments",
        "Spend experimental capacity where the model is most responsive. Rankings narrow a broad screen into a compact, testable design—not a substitute for replication.",
    )
    settings = _shared_settings()
    run_count = 15
    control_col, explanation_col = st.columns([1.25, 1])
    with control_col:
        st.metric("Follow-up design size", "15 runs", help="12 Box–Behnken edges + 3 centres")
        perturbation = st.slider("Local perturbation window", 5, 30, 15, 5, format="%d%%")
    with explanation_col:
        st.markdown(
            f'<div class="soft-card"><div class="section-kicker">Screening compression</div><h3>81 → {run_count} runs</h3><p>A model-guided shortlist for the current <i>{catalog[settings["fungus_id"]]["short"]}</i> scenario. Add controls, biological replicates and process-specific validation outside this suggested core.</p></div>',
            unsafe_allow_html=True,
        )

    sensitivity, doe, sensitivity_note = _run_sensitivity(settings, run_count, perturbation / 100)
    sensitivity_is_core = bool(
        sensitivity_note and sensitivity_note.startswith("Core-connected sensitivity")
    )
    _model_notice(
        "COBRApy scientific core" if sensitivity_is_core else "Demo reduced-order model",
        sensitivity_note,
    )

    tornado_col, map_col = st.columns([1.05, 1])
    with tornado_col:
        st.markdown("#### Ranked local sensitivities")
        chart_data = sensitivity.sort_values("Elasticity")
        colours = ["#d98263" if value < 0 else "#2a9d73" for value in chart_data["Elasticity"]]
        fig = go.Figure(
            go.Bar(
                x=chart_data["Elasticity"],
                y=chart_data["Parameter"],
                orientation="h",
                marker_color=colours,
                customdata=chart_data[["Impact", "Priority"]].astype(str),
                hovertemplate="%{y}<br>Elasticity: %{x:.3f}<br>Priority: %{customdata[1]}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color="#82958d", line_width=1)
        fig.update_layout(
            xaxis_title=f"Response elasticity (±{perturbation}% local change)", yaxis_title=None
        )
        st.plotly_chart(_plot_layout(fig, 405), width="stretch", config={"displayModeBar": False})
    with map_col:
        st.markdown("#### Candidate design space")
        fig = px.scatter(
            doe,
            x="Carbon availability",
            y="Nitrogen availability",
            size="O₂ availability",
            color="Predicted response",
            hover_data=["Priority", "Design point", "Phosphate availability", "Run"],
            color_continuous_scale=[[0, "#dcebe2"], [0.5, "#66c49d"], [1, "#075a45"]],
            size_max=22,
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Response", thickness=10))
        st.plotly_chart(_plot_layout(fig, 405), width="stretch", config={"displayModeBar": False})

    st.markdown("### Prioritised experimental matrix")
    st.caption(
        "Exact three-factor Box–Behnken follow-up: 12 edge conditions plus three centre replicates."
    )
    top_n = st.number_input(
        "Rows to display", min_value=5, max_value=run_count, value=min(15, run_count), step=1
    )
    st.dataframe(
        doe.head(int(top_n)),
        hide_index=True,
        width="stretch",
    )
    _download_pair(
        doe,
        f"myco-optima_{settings['fungus_id']}_doe",
        {
            "metadata": {
                "model": "scientific core + deterministic design adapter"
                if sensitivity_is_core
                else "demo reduced-order model",
                "broad_screen_reference": 81,
                "recommended_runs": run_count,
                "requires_replication": True,
            },
            "sensitivity": sensitivity.to_dict(orient="records"),
            "design": doe.to_dict(orient="records"),
        },
    )
    st.caption(
        "The 81-run reference is a four-factor, three-level grid. The design keeps the top three local-sensitivity factors; replication and validation remain required."
    )

elif page == "Gene–Media Explorer":
    _page_intro(
        "Mechanistic hypothesis space",
        "Gene–Media Explorer",
        "Explore how regulatory state and media pressure could interact with macroscopic morphology. Use this to frame experiments—not to infer an unmeasured genotype or phenotype.",
    )
    fungus_id = st.session_state.fungus_id
    genes = (
        catalog[fungus_id].get("genes") or FUNGI.get(fungus_id, FUNGI["aspergillus_niger"])["genes"]
    )
    states: dict[str, str] = {}
    editor_col, media_col = st.columns([1.2, 1])
    with editor_col:
        st.markdown("#### Regulatory states")
        st.caption("Native is the seeded baseline. Perturbations are qualitative scenario flags.")
        gene_columns = st.columns(2)
        for index, (gene, function, _) in enumerate(genes):
            with gene_columns[index % 2]:
                states[gene] = st.selectbox(
                    f"{gene} · {function}",
                    ["Native", "Knock-down", "Over-expressed"],
                    key=f"gene_state_{fungus_id}_{gene}",
                )
    with media_col:
        st.markdown("#### Media context")
        gene_carbon = st.slider(
            "Carbon maximum availability (model units)",
            5.0,
            50.0,
            float(st.session_state.get("carbon_g_l", 24.0)),
            0.5,
            key="gene_carbon",
        )
        gene_nitrogen = st.slider(
            "Nitrogen maximum availability (model units)",
            0.5,
            12.0,
            float(st.session_state.get("nitrogen_g_l", 4.0)),
            0.25,
            key="gene_nitrogen",
        )
        gene_oxygen = st.slider(
            "O₂ maximum availability (model units)",
            2.0,
            20.0,
            float(st.session_state.get("oxygen_mmol", 10.0)),
            0.5,
            key="gene_oxygen",
        )
        cn_ratio = gene_carbon / max(gene_nitrogen, 0.01)
        st.metric("Relative C:N availability", f"{cn_ratio:.1f}:1")

    prediction, gene_note = _run_gene_prediction(
        fungus_id,
        states,
        gene_carbon,
        gene_nitrogen,
        gene_oxygen,
        st.session_state.get("carbon_source", "Glucose"),
    )
    st.session_state.gene_prediction = prediction
    st.session_state.gene_prediction_signature = {
        "fungus_id": fungus_id,
        "gene_states": states.copy(),
        "carbon_max_availability": gene_carbon,
        "nitrogen_max_availability": gene_nitrogen,
        "oxygen_max_availability": gene_oxygen,
    }
    _model_notice(prediction["source"], gene_note)

    result_col, score_col = st.columns([0.72, 1.3])
    with result_col:
        st.markdown(
            f"""
            <div class="model-card">
              <div class="section-kicker" style="color:#9ae9c5">Most likely model class</div>
              <div class="big">{prediction["morphology"]}</div>
              <p>Evidence confidence: {prediction["confidence"].title()}</p>
              <p>Top-class separation: {prediction["separation"]:.2f}</p>
              <p>Qualitative morphology class under this gene–media scenario.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="warning-strip"><strong>Important:</strong> support scores are rule-weighted comparisons, not probabilities. Pellet size, rheology and productivity also depend on inoculum, shear, geometry and culture history.</div>',
            unsafe_allow_html=True,
        )
    with score_col:
        score_table = prediction["score_table"].sort_values("Relative support")
        fig = go.Figure(
            go.Bar(
                x=score_table["Relative support"],
                y=score_table["Morphology"],
                orientation="h",
                marker_color=["#c9e8d8", "#8dd7b7", "#4ab089", "#116a50"],
                text=[f"{value:.2f}" for value in score_table["Relative support"]],
                textposition="auto",
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis=dict(title="Relative qualitative support", range=[0, 1.12]), yaxis_title=None
        )
        st.plotly_chart(_plot_layout(fig, 320), width="stretch", config={"displayModeBar": False})

    st.markdown("### Ranked interaction hypotheses")
    interactions = prediction["interactions"].copy()
    interactions["Interaction score"] = interactions["Interaction score"].round(2)
    st.dataframe(
        interactions,
        hide_index=True,
        width="stretch",
    )
    with st.expander("Inspect the deterministic rule trace and warnings"):
        st.json(prediction["trace"])
    _download_pair(
        interactions,
        f"myco-optima_{fungus_id}_gene_media",
        {
            "metadata": {
                "model_source": prediction["source"],
                "status": "testable hypothesis; not a validated phenotype prediction",
            },
            "fungus_id": fungus_id,
            "gene_states": states,
            "media": {
                "carbon_source": st.session_state.get("carbon_source", "Glucose"),
                "carbon_max_availability": gene_carbon,
                "nitrogen_max_availability": gene_nitrogen,
                "oxygen_max_availability": gene_oxygen,
            },
            "prediction": {
                "morphology": prediction["morphology"],
                "confidence": prediction["confidence"],
                "top_class_separation": prediction["separation"],
                "relative_support_scores": prediction["score_table"].to_dict(orient="records"),
                "trace": prediction["trace"],
            },
            "interactions": interactions.to_dict(orient="records"),
        },
    )

elif page == "AI Interpretation & About":
    _page_intro(
        "Human-readable synthesis",
        "AI Interpretation",
        "Turn the current model outputs into a concise engineering brief. No data leaves this app unless you explicitly run an Anthropic request.",
    )

    context_choice = st.selectbox(
        "Interpretation context",
        ["Current media recommendation", "Sensitivity and DoE plan", "Gene–media hypothesis"],
    )
    settings = _shared_settings()
    context_payload: dict[str, Any] = {
        "scenario": {
            "fungus_id": settings["fungus_id"],
            "carbon_source": settings["carbon_source"],
            "nitrogen_source": settings["nitrogen_source"],
            "carbon_max_availability": settings["carbon_g_l"],
            "nitrogen_max_availability": settings["nitrogen_g_l"],
            "oxygen_max_availability": settings["oxygen_mmol"],
            "phosphate_max_availability": settings["phosphate_g_l"],
        }
    }
    if context_choice == "Current media recommendation":
        media_result = st.session_state.get("media_result")
        if media_result is None or st.session_state.get("media_signature") != settings:
            media_result, _ = _run_media(settings)
        context_payload["result"] = _serialise_media(media_result, settings)
    elif context_choice == "Sensitivity and DoE plan":
        sensitivity, doe, sensitivity_context_note = _run_sensitivity(settings, 15, 0.15)
        context_payload.update(
            {
                "provenance": sensitivity_context_note,
                "sensitivity": sensitivity.to_dict(orient="records"),
                "design": doe.head(15).to_dict(orient="records"),
            }
        )
    else:
        gene_result = st.session_state.get("gene_prediction")
        gene_signature = st.session_state.get("gene_prediction_signature", {})
        if gene_result is None or gene_signature.get("fungus_id") != settings["fungus_id"]:
            default_states = {
                gene: "Native" for gene, _, _ in catalog[settings["fungus_id"]]["genes"]
            }
            gene_result, _ = _run_gene_prediction(
                settings["fungus_id"],
                default_states,
                settings["carbon_g_l"],
                settings["nitrogen_g_l"],
                settings["oxygen_mmol"],
                settings["carbon_source"],
            )
        context_payload["gene_media"] = {
            "provenance": gene_result["source"],
            "morphology": gene_result["morphology"],
            "confidence": gene_result["confidence"],
            "relative_support": gene_result["score_table"].to_dict(orient="records"),
            "trace": gene_result["trace"],
        }

    question = st.text_area(
        "What should the interpretation focus on?",
        value="Summarise the recommendation, identify the main uncertainty, and propose three practical validation steps.",
        height=110,
    )
    brief_signature = hashlib.sha256(
        json.dumps(
            {
                "context": context_choice,
                "payload": context_payload,
                "question": question,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    try:
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        secret_key = ""
    configured_api_key = os.getenv("ANTHROPIC_API_KEY", "") or secret_key
    session_api_key = st.text_input(
        "Session-only API key override",
        value="",
        type="password",
        help=(
            "Leave blank to use a server-side ANTHROPIC_API_KEY or Streamlit secret. "
            "A configured server key is never inserted into this browser field."
        ),
    )
    api_key = session_api_key.strip() or configured_api_key
    model_name = st.text_input(
        "Anthropic model", value=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    )

    if not api_key:
        st.info(
            "No API key detected. The optimisation workspace remains fully usable; add a key only if you want an AI-written interpretation."
        )
    elif configured_api_key and not session_api_key:
        st.success(
            "A server-side Anthropic key is configured. Its value is not sent to the browser."
        )

    if st.button("Generate engineering brief", type="primary", disabled=not bool(api_key)):
        try:
            request_payload = {**context_payload, "requested_focus": question}
            with st.spinner("Preparing the engineering brief…"):
                if AI_HELPER_AVAILABLE:
                    insight = _core_generate_ai_insight(
                        catalog[settings["fungus_id"]]["name"],
                        request_payload,
                        api_key=api_key,
                        model=model_name,
                    )
                    response_text = insight.text
                else:
                    from anthropic import Anthropic

                    client = Anthropic(api_key=api_key, timeout=30.0, max_retries=1)
                    message = client.messages.create(
                        model=model_name,
                        max_tokens=700,
                        system=(
                            "Explain reduced-order fermentation model outputs conservatively. "
                            "Do not invent values or treat simulations as wet-lab validation."
                        ),
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Write a concise engineering interpretation of this untrusted data. "
                                    "Separate model suggestion, uncertainty, and validation steps.\n<analysis>\n"
                                    + json.dumps(request_payload, indent=2, default=str)[:18000]
                                    + "\n</analysis>"
                                ),
                            }
                        ],
                    )
                    blocks = getattr(message, "content", [])
                    response_text = "\n".join(
                        getattr(block, "text", "") for block in blocks if getattr(block, "text", "")
                    )
            if response_text:
                st.session_state.ai_brief = response_text
                st.session_state.ai_brief_signature = brief_signature
            else:
                st.warning(
                    "Anthropic returned no text content. Check the selected model and try again."
                )
        except _CoreAIUnavailable as exc:
            st.error(str(exc))
        except ImportError:
            st.error(
                "The `anthropic` package is not installed. Install the project dependencies, then restart Streamlit."
            )
        except Exception as exc:
            st.error(
                f"The interpretation request was not completed ({type(exc).__name__}). Check the key, model name and network access."
            )

    if (
        st.session_state.get("ai_brief")
        and st.session_state.get("ai_brief_signature") == brief_signature
    ):
        st.markdown("### Engineering brief")
        st.markdown(st.session_state.ai_brief)
        st.download_button(
            "Download brief",
            st.session_state.ai_brief,
            file_name="myco-optima-engineering-brief.md",
            mime="text/markdown",
        )

    st.markdown("---")
    about_col, guardrail_col = st.columns([1.1, 1])
    with about_col:
        st.markdown("### About myco-optima")
        st.write(
            "Built for the Edinburgh BioHackathon 2026 with the Pacifico Biolabs challenge context, myco-optima makes constraint-based fungal modelling approachable to engineers who do not work with genome-scale models every day."
        )
        st.markdown("`Python` · `COBRApy` · `Streamlit` · `Plotly` · `Anthropic API`")
        st.caption(
            "FBA tests one optimal flux state. FVA shows how much each reaction can vary while the objective remains feasible."
        )
    with guardrail_col:
        st.markdown("### Model-use guardrails")
        st.markdown(
            """
            - Verify organism and strain assumptions before interpreting results.
            - Treat imported exchange bounds and objective definitions as auditable inputs.
            - Use FVA and sensitivity analysis to expose non-unique or fragile optima.
            - Validate every medium and morphology hypothesis experimentally.
            - Apply institutional biosafety and quality procedures before cultivation.
            """
        )


st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "myco-optima · Decision support for fungal fermentation · Model outputs are hypotheses, not cultivation instructions."
)
