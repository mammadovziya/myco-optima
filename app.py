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
from datetime import UTC, datetime
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
MODEL_UPLOAD_AVAILABLE = False
SEQUENCE_UPLOAD_AVAILABLE = False

try:
    from myco_optima import __version__ as APP_VERSION
except Exception:
    APP_VERSION = "0.3.0"

try:
    from cobra import __version__ as COBRA_VERSION
except Exception:
    COBRA_VERSION = "unavailable"

try:
    from myco_optima.exports import dataframe_to_safe_csv as _safe_csv_bytes
    from myco_optima.exports import strict_json_dumps as _strict_json_dumps
except Exception:

    def _safe_csv_bytes(frame: pd.DataFrame) -> bytes:
        safe = frame.copy(deep=True)
        for column in safe.columns:
            if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(
                safe[column]
            ):
                safe[column] = safe[column].map(
                    lambda value: (
                        "'" + value
                        if isinstance(value, str)
                        and value
                        and (
                            value[0] in {"\t", "\r", "\n"}
                            or value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@"))
                        )
                        else value
                    )
                )
        return safe.to_csv(index=False).encode("utf-8")

    def _strict_json_dumps(payload: Any) -> str:
        return json.dumps(payload, indent=2, default=str, allow_nan=False)


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

try:
    from myco_optima.model_io import DEFAULT_MAX_UPLOAD_BYTES as _CUSTOM_MODEL_MAX_BYTES
    from myco_optima.model_io import MAX_FVA_REACTIONS as _CUSTOM_MODEL_MAX_FVA
    from myco_optima.model_io import CustomModelAnalysisError as _CustomModelAnalysisError
    from myco_optima.model_io import ModelUploadError as _ModelUploadError
    from myco_optima.model_io import analyse_custom_model as _core_analyse_custom_model
    from myco_optima.model_io import load_sbml_upload as _core_load_sbml_upload

    MODEL_UPLOAD_AVAILABLE = True
except Exception:
    _CUSTOM_MODEL_MAX_BYTES = 5_000_000
    _CUSTOM_MODEL_MAX_FVA = 50
    _CustomModelAnalysisError = RuntimeError
    _ModelUploadError = ValueError
    _core_analyse_custom_model = None
    _core_load_sbml_upload = None

try:
    from myco_optima.sequence_io import (
        DEFAULT_MAX_COMBINED_SEQUENCE_BYTES as _SEQUENCE_SESSION_MAX_BYTES,
    )
    from myco_optima.sequence_io import DEFAULT_MAX_SEQUENCE_BYTES as _SEQUENCE_MAX_BYTES
    from myco_optima.sequence_io import SequenceIntakeError as _SequenceUploadError
    from myco_optima.sequence_io import (
        build_reconstruction_handoff as _core_build_reconstruction_handoff,
    )
    from myco_optima.sequence_io import load_fasta_upload as _core_load_fasta_upload
    from myco_optima.sequence_io import pair_fasta_inspections as _core_pair_fasta_inspections

    SEQUENCE_UPLOAD_AVAILABLE = True
except Exception:
    _SEQUENCE_MAX_BYTES = 100_000_000
    _SEQUENCE_SESSION_MAX_BYTES = 150_000_000
    _SequenceUploadError = ValueError
    _core_build_reconstruction_handoff = None
    _core_load_fasta_upload = None
    _core_pair_fasta_inspections = None


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


def _display_text(value: Any, max_chars: int = 180) -> str:
    """Bound untrusted display text while leaving source/export values unchanged."""

    text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


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


def _load_custom_model(filename: str, payload: bytes) -> Any:
    """Parse one validated SBML payload without sharing it across user sessions."""

    if not MODEL_UPLOAD_AVAILABLE or _core_load_sbml_upload is None:
        raise RuntimeError("The custom model loader is unavailable.")
    return _core_load_sbml_upload(payload, filename)


def _load_sequence(filename: str, payload: bytes | bytearray | memoryview) -> Any:
    """Inspect one FASTA payload without sharing it across user sessions."""

    if not SEQUENCE_UPLOAD_AVAILABLE or _core_load_fasta_upload is None:
        raise RuntimeError("The FASTA sequence loader is unavailable.")
    return _core_load_fasta_upload(payload, filename)


_CUSTOM_ANALYSIS_STATE_KEYS = (
    "custom_analysis",
    "custom_analysis_signature",
    "custom_analysis_metadata",
    "custom_analysis_generated_at",
)
_CUSTOM_INSPECTION_STATE_KEYS = (
    "custom_inspection",
    "custom_inspection_identity",
    "custom_active_upload_identity",
)
_CUSTOM_WIDGET_PREFIXES = (
    "custom_objective_",
    "custom_direction_",
    "custom_include_fva_",
    "custom_fva_reactions_",
    "custom_fva_fraction_",
)
_CUSTOM_UPLOAD_WIDGET_PREFIXES = ("custom_sbml_",)
_SEQUENCE_STATE_KEYS = (
    "sequence_inspections",
    "sequence_active_upload_identity",
    "sequence_inspection_identity",
    "sequence_inspection_identities",
)
_SEQUENCE_WIDGET_PREFIXES = ("sequence_fna_", "sequence_faa_")


def _clear_custom_analysis_state() -> None:
    for key in _CUSTOM_ANALYSIS_STATE_KEYS:
        st.session_state.pop(key, None)


def _clear_custom_model_state(*, clear_widgets: bool = False) -> None:
    _clear_custom_analysis_state()
    for key in _CUSTOM_INSPECTION_STATE_KEYS:
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if key.startswith(_CUSTOM_WIDGET_PREFIXES) or (
            clear_widgets and key.startswith(_CUSTOM_UPLOAD_WIDGET_PREFIXES)
        ):
            st.session_state.pop(key, None)


def _clear_sequence_state(*, clear_widgets: bool = False) -> None:
    for key in _SEQUENCE_STATE_KEYS:
        st.session_state.pop(key, None)
    if clear_widgets:
        for key in list(st.session_state):
            if key.startswith(_SEQUENCE_WIDGET_PREFIXES):
                st.session_state.pop(key, None)


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
    page_icon=str(APP_DIR / "assets" / "myco-optima-mark.svg"),
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "myco-optima · Edinburgh BioHackathon 2026"},
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8f6; }
        [data-testid="stSidebar"] { background: #eef2ef; border-right: 1px solid #d8e0db; }
        [data-testid="stSidebar"] [data-testid="stImage"] img { max-width: 230px; }
        [data-testid="stHeader"] { background: rgba(247, 248, 246, 0.92); }
        .block-container { padding-top: 1.8rem; padding-bottom: 3.5rem; max-width: 1280px; }
        h1, h2, h3 { color: #10251f; letter-spacing: -0.02em; }
        h1 { font-size: clamp(2rem, 3vw, 3.1rem) !important; line-height: 1.08 !important; }
        p, label, [data-testid="stCaptionContainer"] { color: #52665e; }
        div[data-testid="stDataFrame"] { border: 1px solid #d8e0db; border-radius: 8px; overflow: hidden; }
        [data-testid="stFileUploaderDropzone"] { background: #f2f6f3; border-color: #aebdb4; }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { min-height: 40px; }
        @media (max-width: 780px) {
            .block-container { padding-top: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(label: str, value: str, delta: str) -> None:
    with st.container(border=True):
        st.metric(label, value)
        st.caption(delta)


def _page_intro(kicker: str, title: str, body: str) -> None:
    st.caption(kicker.upper())
    st.title(title)
    st.markdown(body)
    st.divider()


def _download_pair(
    frame: pd.DataFrame, filename: str, json_payload: dict[str, Any] | None = None
) -> None:
    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            "Download CSV",
            _safe_csv_bytes(frame),
            file_name=f"{filename}.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_json:
        payload = json_payload or {"records": frame.to_dict(orient="records")}
        st.download_button(
            "Download JSON",
            _strict_json_dumps(payload),
            file_name=f"{filename}.json",
            mime="application/json",
            width="stretch",
        )


def _sequence_metadata(inspection: Any) -> dict[str, Any]:
    """Return the bounded, JSON-safe metadata supplied by the sequence core."""

    metadata = inspection.metadata() if hasattr(inspection, "metadata") else inspection
    plain = _plain(metadata)
    return plain if isinstance(plain, dict) else {}


def _sequence_field(metadata: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return default


def _sequence_preview(inspection: Any) -> pd.DataFrame:
    metadata = _sequence_metadata(inspection)
    raw_preview = _sequence_field(metadata, "preview", "record_preview", default=[])
    if hasattr(inspection, "preview"):
        raw_preview = _plain(inspection.preview(limit=20))
    rows = raw_preview if isinstance(raw_preview, list) else []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Record ID", "Description", "Length"])
    return frame.rename(
        columns={
            "id": "Record ID",
            "record_id": "Record ID",
            "identifier": "Record ID",
            "description": "Description",
            "length": "Length",
            "gc_percent": "GC (%)",
        }
    )


def _render_sequence_workspace() -> None:
    """Render session-local FNA/FAA intake without exposing solver controls."""

    st.subheader("Nucleotide / protein FASTA intake")
    st.markdown(
        "Bring nucleotide sequences (`.fna`), protein sequences (`.faa`), or both "
        "files together. This route validates and inventories sequences for "
        "a reconstruction handoff; it does **not** annotate them or build a metabolic model."
    )
    st.warning(
        "FBA and FVA need reaction stoichiometry, bounds and an objective. Use an "
        "external annotation and GEM-reconstruction workflow, curate its result, then "
        "return here and upload the resulting SBML model."
    )

    if not SEQUENCE_UPLOAD_AVAILABLE:
        st.error(
            "The FASTA intake module is unavailable in this installation. Install the "
            "project dependencies and restart Streamlit."
        )
        return

    upload_key = st.session_state.get("sequence_upload_key", 0)
    with st.container(border=True):
        st.markdown("#### Nucleotide input")
        uploaded_fna = st.file_uploader(
            "Nucleotide FASTA (.fna)",
            type=["fna"],
            accept_multiple_files=False,
            max_upload_size=int(_SEQUENCE_MAX_BYTES / 1_000_000),
            key=f"sequence_fna_{upload_key}",
            help=(f"One UTF-8 FASTA file, up to {_SEQUENCE_MAX_BYTES / 1_000_000:.0f} MB."),
        )
        fna_content = st.radio(
            "FNA content declaration",
            ["Genome assembly", "Coding sequences (CDS)"],
            horizontal=True,
            key=f"sequence_fna_content_{upload_key}",
            help=(
                "Choose what the records represent. Assembly contig identifiers are not "
                "treated as protein identifiers."
            ),
        )
    with st.container(border=True):
        st.markdown("#### Protein input")
        uploaded_faa = st.file_uploader(
            "Protein FASTA (.faa)",
            type=["faa"],
            accept_multiple_files=False,
            max_upload_size=int(_SEQUENCE_MAX_BYTES / 1_000_000),
            key=f"sequence_faa_{upload_key}",
            help=(f"One UTF-8 FASTA file, up to {_SEQUENCE_MAX_BYTES / 1_000_000:.0f} MB."),
        )
        st.caption(
            "If this came from an annotation workflow, keep that provenance with the "
            "handoff. Co-uploading files "
            "does not prove that a nucleotide record encodes a protein record."
        )

    st.caption(
        f"Session intake budget: one FNA and one FAA, each up to "
        f"{_SEQUENCE_MAX_BYTES / 1_000_000:.0f} MB "
        f"({_SEQUENCE_SESSION_MAX_BYTES / 1_000_000:.0f} MB combined). Files remain in this "
        "Streamlit session and are never sent to Anthropic."
    )

    uploads = {"fna": uploaded_fna, "faa": uploaded_faa}
    active_uploads = {kind: upload for kind, upload in uploads.items() if upload is not None}
    if not active_uploads:
        if st.session_state.get("sequence_active_upload_identity") is not None:
            _clear_sequence_state()
        st.info("Upload an `.fna`, an `.faa`, or both to create a sequence inventory.")
        return

    oversized = []
    for kind, upload in active_uploads.items():
        upload_size = getattr(upload, "size", None)
        if upload_size is not None and upload_size > _SEQUENCE_MAX_BYTES:
            oversized.append(f".{kind} ({upload_size / 1_000_000:.1f} MB)")
    if oversized:
        _clear_sequence_state()
        st.error(
            "Upload exceeds the per-file sequence limit: "
            + ", ".join(oversized)
            + f". Maximum: {_SEQUENCE_MAX_BYTES / 1_000_000:.0f} MB each."
        )
        return

    payloads = {kind: upload.getbuffer() for kind, upload in active_uploads.items()}
    if sum(len(payload) for payload in payloads.values()) > _SEQUENCE_SESSION_MAX_BYTES:
        _clear_sequence_state()
        st.error("The combined FNA/FAA payload exceeds the 150 MB session intake budget.")
        return

    file_identities = {
        kind: {
            "filename": active_uploads[kind].name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for kind, payload in payloads.items()
    }
    upload_identity = {
        "fna_content": fna_content if "fna" in active_uploads else None,
        "files": file_identities,
    }
    st.session_state.sequence_active_upload_identity = upload_identity

    cached_inspections = st.session_state.get("sequence_inspections", {})
    cached_identities = st.session_state.get("sequence_inspection_identities", {})
    if not isinstance(cached_inspections, dict):
        cached_inspections = {}
    if not isinstance(cached_identities, dict):
        cached_identities = {}
    inspections: dict[str, Any] = {}
    successful_identities: dict[str, Any] = {}
    changed_kinds = [
        kind for kind in active_uploads if cached_identities.get(kind) != file_identities[kind]
    ]
    for kind in active_uploads:
        if kind not in changed_kinds and kind in cached_inspections:
            inspections[kind] = cached_inspections[kind]
            successful_identities[kind] = file_identities[kind]

    if changed_kinds:
        with st.spinner("Validating FASTA records…"):
            for kind in changed_kinds:
                try:
                    inspections[kind] = _load_sequence(active_uploads[kind].name, payloads[kind])
                    successful_identities[kind] = file_identities[kind]
                except _SequenceUploadError as exc:
                    st.error(_display_text(str(exc), max_chars=300))
                except Exception as exc:
                    st.error(
                        f"The .{kind} upload could not be inspected safely "
                        f"({type(exc).__name__}). Check the FASTA structure."
                    )
    st.session_state.pop("sequence_inspection_identity", None)
    if inspections:
        st.session_state.sequence_inspections = inspections
        st.session_state.sequence_inspection_identities = successful_identities
    else:
        st.session_state.pop("sequence_inspections", None)
        st.session_state.pop("sequence_inspection_identities", None)

    if not inspections:
        return

    pair_status = "Co-uploaded FNA + FAA" if len(inspections) == 2 else "Single sequence input"
    _model_notice(
        "Validated sequence intake",
        f"{pair_status}. This is an inventory and reconstruction handoff, not a GEM.",
    )
    st.subheader("Sequence inventory")

    inventory_rows: list[dict[str, Any]] = []
    preview_frames: list[pd.DataFrame] = []
    for kind in ("fna", "faa"):
        if kind not in inspections:
            continue
        inspection = inspections[kind]
        metadata = _sequence_metadata(inspection)
        record_count = int(
            _sequence_field(
                metadata,
                "sequence_count" if kind == "fna" else "protein_count",
                "record_count",
                default=0,
            )
        )
        total_residues = int(
            _sequence_field(
                metadata,
                "total_bp" if kind == "fna" else "total_aa",
                "total_residues",
                default=0,
            )
        )
        minimum_length = int(_sequence_field(metadata, "minimum_length", "min_length", default=0))
        maximum_length = int(_sequence_field(metadata, "maximum_length", "max_length", default=0))
        median_length = float(_sequence_field(metadata, "median_length", default=0.0))
        gc_percent = _sequence_field(metadata, "gc_percent", "gc_content_percent")
        n_percent = _sequence_field(metadata, "n_percent")
        n50 = _sequence_field(metadata, "n50")
        ambiguous_fraction = _sequence_field(metadata, "ambiguous_residue_fraction")
        terminal_stops = _sequence_field(
            metadata, "terminal_stop_marker_count", "terminal_stop_count"
        )
        filename = str(_sequence_field(metadata, "filename", default=active_uploads[kind].name))
        sha256 = str(
            _sequence_field(
                metadata,
                "sha256",
                default=upload_identity["files"][kind]["sha256"],
            )
        )
        role = fna_content if kind == "fna" else "Protein sequences"
        role_label = f"User-declared {role.lower()}" if kind == "fna" else role

        with st.container(border=True):
            st.markdown(f"#### {'.fna' if kind == 'fna' else '.faa'} · {role_label}")
            st.text(f"File: {_display_text(filename)}")
            st.text(f"SHA-256: {sha256}")
            metric_row_one = st.columns(2)
            metric_row_two = st.columns(2)
            metric_row_one[0].metric(
                "Nucleotide records" if kind == "fna" else "Protein records",
                f"{record_count:,}",
            )
            metric_row_one[1].metric(
                "Total bases" if kind == "fna" else "Amino acids",
                f"{total_residues:,}",
            )
            metric_row_two[0].metric("Shortest record", f"{minimum_length:,}")
            metric_row_two[1].metric("Longest record", f"{maximum_length:,}")
            if kind == "fna":
                gc_label = f"{float(gc_percent):.2f}%" if gc_percent is not None else "n/a"
                length_statistic = (
                    f"Assembly N50: {int(n50 or 0):,} · "
                    if fna_content == "Genome assembly"
                    else ""
                )
                st.caption(
                    f"Median length: {median_length:,.1f} bases · "
                    f"{length_statistic}GC: {gc_label} · N: {float(n_percent or 0):.2f}%"
                )
                if fna_content == "Genome assembly":
                    st.caption(
                        "Assembly N50 is a sequence-length contiguity statistic, not completeness."
                    )
            else:
                st.caption(
                    f"Median length: {median_length:,.1f} amino acids · "
                    f"Ambiguous residues: {float(ambiguous_fraction or 0):.2%} · "
                    f"Terminal stop markers: {int(terminal_stops or 0):,}"
                )
            for warning in metadata.get("warnings", []):
                st.warning(_display_text(warning, max_chars=300))

        row = {
            "Input": kind.upper(),
            "Declared role": role,
            "Filename": filename,
            "SHA-256": sha256,
            "Bytes": upload_identity["files"][kind]["bytes"],
            "Records": record_count,
            "Total residues": total_residues,
            "Minimum length": minimum_length,
            "Maximum length": maximum_length,
            "Median length": median_length,
            "GC (%)": float(gc_percent) if gc_percent is not None else None,
            "N (%)": float(n_percent) if n_percent is not None else None,
            "Assembly N50": (
                int(n50)
                if kind == "fna" and fna_content == "Genome assembly" and n50 is not None
                else None
            ),
            "Ambiguous residue fraction": (
                float(ambiguous_fraction) if ambiguous_fraction is not None else None
            ),
            "Terminal stop markers": int(terminal_stops) if terminal_stops is not None else None,
        }
        inventory_rows.append(row)
        preview = _sequence_preview(inspection)
        if not preview.empty:
            preview.insert(0, "Input", kind.upper())
            preview_frames.append(preview)

    with st.expander("Bounded record preview", expanded=True):
        if preview_frames:
            st.dataframe(
                pd.concat(preview_frames, ignore_index=True),
                hide_index=True,
                width="stretch",
            )
            st.caption("Only a bounded preview is shown; sequence residues are not displayed.")
        else:
            st.info("The validated files did not provide preview rows.")

    relationship: dict[str, Any] = {
        "status": "co-uploaded" if len(inspections) == 2 else "single-input",
        "claim": "No sequence linkage, annotation or gene–protein mapping is inferred.",
    }
    pair_inspection = None
    handoff_error = False
    if len(inspections) == 2 and _core_pair_fasta_inspections is not None:
        try:
            pair_inspection = _core_pair_fasta_inspections(
                inspections["fna"],
                inspections["faa"],
                nucleotide_role=("cds" if fna_content == "Coding sequences (CDS)" else "assembly"),
            )
            if fna_content == "Coding sequences (CDS)":
                overlap_count = int(pair_inspection.syntactic_id_overlap_count)
                relationship["exact_identifier_overlap"] = overlap_count
                relationship["exact_identifier_overlap_preview"] = list(
                    pair_inspection.syntactic_id_overlap_preview
                )
                st.caption(
                    f"Exact record-ID overlap: {overlap_count:,}. This string comparison "
                    "uses the case-sensitive first token after `>` and does not establish "
                    "annotation or biological linkage."
                )
        except _SequenceUploadError as exc:
            handoff_error = True
            st.error(_display_text(str(exc), max_chars=300))
        except Exception as exc:
            handoff_error = True
            st.error(f"The paired handoff could not be validated safely ({type(exc).__name__}).")

    inventory_frame = pd.DataFrame(inventory_rows)
    manifest = {
        "schema_version": "1.0",
        "handoff": "external-fungal-gem-reconstruction",
        "relationship": relationship,
        "fna_content_declaration": fna_content if "fna" in inspections else None,
        "files": [
            _sequence_metadata(inspections[kind]) for kind in ("fna", "faa") if kind in inspections
        ],
        "generated_at": datetime.now(UTC).isoformat(),
        "app_version": APP_VERSION,
        "next_step": (
            "Annotate and reconstruct externally; curate the resulting network and upload "
            "an explicitly bounded SBML model before FBA/FVA."
        ),
        "limitations": [
            "No annotation, metabolic reconstruction or gap filling was performed.",
            "No species, gene–protein mapping, morphology or metabolic capability was inferred.",
            "Sequence files were not sent to Anthropic.",
        ],
    }
    if _core_build_reconstruction_handoff is not None and not handoff_error:
        try:
            if pair_inspection is not None:
                core_manifest = _core_build_reconstruction_handoff(pair_inspection)
            else:
                single_kind, single_inspection = next(iter(inspections.items()))
                handoff_kwargs = (
                    {
                        "nucleotide_role": (
                            "cds" if fna_content == "Coding sequences (CDS)" else "assembly"
                        )
                    }
                    if single_kind == "fna"
                    else {}
                )
                core_manifest = _core_build_reconstruction_handoff(
                    single_inspection, **handoff_kwargs
                )
            core_payload = (
                core_manifest.to_dict()
                if hasattr(core_manifest, "to_dict")
                else _plain(core_manifest)
            )
            if isinstance(core_payload, dict):
                manifest = {
                    **core_payload,
                    "export_context": {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "app_version": APP_VERSION,
                        "relationship_label": relationship["status"],
                    },
                }
        except _SequenceUploadError as exc:
            handoff_error = True
            st.error(_display_text(str(exc), max_chars=300))
        except Exception as exc:
            handoff_error = True
            st.error(
                f"The reconstruction handoff could not be serialized safely ({type(exc).__name__})."
            )
    st.subheader("Reconstruction handoff")
    st.dataframe(inventory_frame, hide_index=True, width="stretch")
    if handoff_error:
        st.warning("Downloads are disabled until the sequence handoff validates cleanly.")
    else:
        _download_pair(inventory_frame, "myco-optima_sequence_handoff", manifest)

    with st.expander("What happens next?"):
        st.markdown(
            "1. Run organism-appropriate annotation and draft GEM reconstruction outside "
            "Myco Optima.\n2. Review gene–protein–reaction rules, gaps, biomass composition "
            "and exchange bounds.\n3. Export the curated model as SBML.\n4. Return to the "
            "**Metabolic model (SBML)** route for FBA/FVA."
        )
        if st.button("Forget sequence uploads"):
            _clear_sequence_state(clear_widgets=True)
            st.session_state.sequence_upload_key = upload_key + 1
            st.rerun()


def _model_notice(source: str, note: str | None = None) -> None:
    is_demo = "demo" in source.lower()
    detail = note or (
        "Illustrative, deterministic outputs for interface exploration."
        if is_demo
        else "Results were returned by the installed scientific core."
    )
    message = f"**{source}** — {detail}"
    if is_demo:
        st.warning(message)
    else:
        st.info(message)


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
        st.markdown("## myco-optima")
    st.caption("Fungal Fermentation Optimisation Tool · 2026")
    st.markdown("---")

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Custom Model",
            "Media Optimiser",
            "Sensitivity & DoE",
            "Gene–Media Explorer",
            "Interpretation & Methods",
        ],
        help="Choose a modelling workflow.",
    )

    process_mode = "Aerobic steady-state surrogate"
    if page != "Custom Model":
        with st.expander("Curated scenario", expanded=True):
            fungus_name_to_id = {item["name"]: fungus_id for fungus_id, item in catalog.items()}
            chosen_name = st.selectbox(
                "Organism",
                list(fungus_name_to_id),
                index=list(fungus_name_to_id).index(catalog[st.session_state.fungus_id]["name"]),
            )
            st.session_state.fungus_id = fungus_name_to_id[chosen_name]
            st.session_state.objective = "Biomass productivity"
            st.caption("Objective · biomass productivity")
            st.caption("Scope · aerobic steady-state surrogate")
    else:
        st.caption("Upload controls and model provenance appear in the custom workspace.")

    st.markdown("---")
    if CORE_AVAILABLE:
        st.success("Scientific core connected")
    else:
        st.warning("Demo model active")
    st.caption(CORE_IMPORT_NOTE)
    st.caption("v0.3 · Edinburgh BioHackathon 2026")


# Pages -----------------------------------------------------------------------

if page == "Overview":
    _page_intro(
        "Model workspace",
        "Fungal fermentation, from model to experiment",
        "Choose a curated fungal surrogate or bring a COBRA-compatible SBML reconstruction. "
        "Use one traceable workspace to inspect feasible flux, rank media constraints and "
        "prepare a focused follow-up design.",
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

    st.subheader("Curated starting models")
    st.caption("Fast, transparent teaching surrogates for exploring the workflow.")
    fungus_cols = st.columns(4)
    for column, (_, fungus) in zip(fungus_cols, catalog.items(), strict=True):
        with column, st.container(border=True):
            st.markdown(f"**_{fungus['name']}_**")
            st.caption(fungus["role"])
            st.write(f"{fungus['temperature']:.0f} °C · pH {fungus['ph']:.1f}")

    workflow_col, context_col = st.columns([1.35, 1])
    with workflow_col:
        st.subheader("A traceable path from model to bench")
        with st.container(border=True):
            st.markdown(
                """
                1. **Define the operating envelope.** Select the organism, substrates,
                   oxygen ceiling and cost boundary.
                2. **Inspect feasible flux.** Read the FBA optimum alongside 95%-optimal
                   FVA ranges.
                3. **Prioritise uncertainty.** Rank influential constraints and generate
                   the 15-run follow-up design.
                4. **Frame a morphology experiment.** Use gene–media rules as explicit,
                   testable hypotheses.
                """
            )
    with context_col:
        st.subheader("Current curated scenario")
        current = catalog[st.session_state.fungus_id]
        with st.container(border=True):
            st.markdown(f"### _{current['name']}_")
            st.write(st.session_state.objective)
            st.caption(process_mode)
            st.caption(
                f"Reference operating point · {current['temperature']:.0f} °C · "
                f"pH {current['ph']:.1f}"
            )
        st.warning(
            "Reduced-order outputs are model-derived hypotheses. Calibrate the model, "
            "complete a safety review and validate experimentally before cultivation."
        )

elif page == "Custom Model":
    _page_intro(
        "Bring your own biological input",
        "Custom model and sequence intake",
        "Start from a COBRA-compatible SBML reconstruction, or inventory nucleotide and "
        "protein FASTA files for an external reconstruction handoff. Both routes are "
        "session-local and are never sent to the optional interpretation service automatically.",
    )

    custom_input_route = st.radio(
        "Custom input type",
        ["Metabolic model (SBML)", "Nucleotide / protein FASTA"],
        horizontal=True,
        help="FASTA intake does not run flux analysis; only a reconstructed SBML model can do that.",
        key="custom_input_route",
    )
    previous_input_route = st.session_state.get("custom_input_route_applied")
    if previous_input_route != custom_input_route:
        if custom_input_route == "Nucleotide / protein FASTA":
            _clear_custom_model_state(clear_widgets=True)
        else:
            _clear_sequence_state(clear_widgets=True)
        st.session_state.custom_input_route_applied = custom_input_route
    if custom_input_route == "Nucleotide / protein FASTA":
        _render_sequence_workspace()
        st.stop()

    st.subheader("Metabolic model (SBML)")
    st.caption(
        "Upload an explicitly bounded reconstruction, review its objective, then run FBA "
        "and a bounded reaction subset for FVA."
    )

    if not MODEL_UPLOAD_AVAILABLE:
        st.error(
            "The custom-model module is unavailable in this installation. Install the "
            "project dependencies and restart Streamlit."
        )
    else:
        upload_col, scope_col = st.columns([1.45, 1])
        with upload_col:
            uploaded_model = st.file_uploader(
                "Upload SBML",
                type=["xml", "sbml"],
                accept_multiple_files=False,
                max_upload_size=int(_CUSTOM_MODEL_MAX_BYTES / 1_000_000),
                key=f"custom_sbml_{st.session_state.get('custom_upload_key', 0)}",
                help=(
                    f"Uncompressed UTF-8 SBML only. Maximum size: "
                    f"{_CUSTOM_MODEL_MAX_BYTES / 1_000_000:.0f} MB."
                ),
            )
        with scope_col, st.container(border=True):
            st.markdown("**What this workspace supports**")
            st.markdown(
                "- Objective selection from uploaded reactions\n"
                "- Single-solution FBA\n"
                f"- FVA for up to {_CUSTOM_MODEL_MAX_FVA} selected reactions\n"
                "- Traceable CSV and JSON exports"
            )
            st.caption(
                "Curated media optimisation, sensitivity-led DoE and morphology "
                "rules remain available only for the four bundled fungal models."
            )

        inspection = None
        if uploaded_model is None:
            if st.session_state.get("custom_active_upload_identity") is not None:
                _clear_custom_model_state()
        else:
            upload_size = getattr(uploaded_model, "size", None)
            if upload_size is not None and upload_size > _CUSTOM_MODEL_MAX_BYTES:
                _clear_custom_model_state()
                st.error(
                    f"The SBML upload exceeds the {_CUSTOM_MODEL_MAX_BYTES / 1_000_000:.0f} MB "
                    "custom-model limit. Sequence uploads use a separate intake route."
                )
                st.stop()
            payload = uploaded_model.getbuffer()
            upload_identity = {
                "filename": uploaded_model.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if st.session_state.get("custom_active_upload_identity") != upload_identity:
                _clear_custom_model_state()
                st.session_state.custom_active_upload_identity = upload_identity
            try:
                if st.session_state.get("custom_inspection_identity") == upload_identity:
                    inspection = st.session_state.get("custom_inspection")
                else:
                    with st.spinner("Validating SBML structure…"):
                        inspection = _load_custom_model(uploaded_model.name, payload)
                    st.session_state.custom_inspection = inspection
                    st.session_state.custom_inspection_identity = upload_identity
            except _ModelUploadError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(
                    f"The upload could not be opened safely ({type(exc).__name__}). "
                    "Check that it is an uncompressed COBRA-compatible SBML file."
                )

        if inspection is None:
            st.info(
                "No custom model is active. Upload an `.xml` or `.sbml` file to begin, "
                "or use the curated fungal workflows from the sidebar."
            )
        else:
            model_label = inspection.model_name or inspection.model_id or inspection.filename
            _model_notice(
                "Validated custom SBML",
                "Identifiers and explicit FBC constraints passed upload preflight.",
            )
            st.text(f"File: {inspection.filename}")
            st.text(f"SHA-256: {inspection.sha256}")
            st.subheader("Uploaded model")
            st.text(_display_text(model_label))
            summary_cols = st.columns(4)
            summary_cols[0].metric("Reactions", f"{inspection.reactions:,}")
            summary_cols[1].metric("Metabolites", f"{inspection.metabolites:,}")
            summary_cols[2].metric("Genes", f"{inspection.genes:,}")
            summary_cols[3].metric("Exchanges", f"{inspection.exchanges:,}")
            if len(inspection.current_objective) > 1:
                st.warning(
                    "The uploaded model defines a composite objective. This workspace "
                    "runs one explicitly selected reaction objective, so review the "
                    "choice below before solving."
                )
            with st.expander("Model validation notes"):
                for warning in inspection.warnings:
                    st.write(f"- {warning}")

            label_by_id = {
                candidate.reaction_id: _display_text(candidate.label)
                for candidate in inspection.objective_candidates
            }
            objective_ids = list(inspection.candidate_objective_reaction_ids)
            current_objective = inspection.current_objective_id
            default_objective_index = (
                objective_ids.index(current_objective) if current_objective in objective_ids else 0
            )

            setup_col, fva_col = st.columns(2)
            widget_suffix = inspection.sha256[:16]
            with setup_col:
                objective_valid = True
                if len(objective_ids) <= 500:
                    selected_objective = st.selectbox(
                        "Objective reaction",
                        objective_ids,
                        index=default_objective_index,
                        format_func=lambda reaction_id: label_by_id[reaction_id],
                        help=(
                            "The uploaded objective is selected by default when it "
                            "contains one term."
                        ),
                        key=f"custom_objective_{widget_suffix}",
                    )
                else:
                    selected_objective = st.text_input(
                        "Objective reaction ID",
                        value=current_objective or objective_ids[0],
                        key=f"custom_objective_{widget_suffix}",
                        help=(
                            "Large models use an exact reaction ID field to avoid sending "
                            "thousands of select options to the browser."
                        ),
                    ).strip()
                    objective_valid = selected_objective in label_by_id
                    with st.expander("Suggested objective reactions"):
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Reaction": candidate.reaction_id,
                                        "Label": candidate.label,
                                        "Current": candidate.is_current,
                                        "Boundary": candidate.boundary,
                                    }
                                    for candidate in inspection.objective_candidates[:30]
                                ]
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                    if not objective_valid:
                        st.error("Enter a reaction ID that exists in the uploaded model.")
                direction_options = ["max", "min"]
                direction = st.radio(
                    "Objective direction",
                    direction_options,
                    index=(
                        direction_options.index(inspection.objective_direction)
                        if inspection.objective_direction in direction_options
                        else 0
                    ),
                    horizontal=True,
                    key=f"custom_direction_{widget_suffix}",
                )
                if objective_valid:
                    selected_candidate = next(
                        candidate
                        for candidate in inspection.objective_candidates
                        if candidate.reaction_id == selected_objective
                    )
                    if selected_candidate.boundary:
                        st.warning(
                            "The selected objective is a boundary reaction. Confirm that "
                            "maximising or minimising this exchange/demand flux matches the "
                            "biological question."
                        )
            with fva_col:
                include_fva = st.checkbox(
                    "Run flux variability analysis",
                    value=True,
                    key=f"custom_include_fva_{widget_suffix}",
                )
                effective_objective = (
                    selected_objective
                    if objective_valid
                    else (current_objective or objective_ids[0])
                )
                default_fva_ids = list(
                    dict.fromkeys([effective_objective, *inspection.exchange_reaction_ids[:7]])
                )
                fva_selection_valid = True
                if len(objective_ids) <= 500:
                    selected_fva_ids = st.multiselect(
                        "FVA reactions",
                        objective_ids,
                        default=default_fva_ids,
                        format_func=lambda reaction_id: label_by_id[reaction_id],
                        max_selections=_CUSTOM_MODEL_MAX_FVA,
                        disabled=not include_fva,
                        help=(
                            "Select a bounded subset; whole-model FVA can be "
                            "unexpectedly expensive."
                        ),
                        key=f"custom_fva_reactions_{widget_suffix}",
                    )
                    fva_selection_valid = not include_fva or bool(selected_fva_ids)
                    if include_fva and not fva_selection_valid:
                        st.error("Choose at least one reaction for FVA, or turn FVA off.")
                else:
                    raw_fva_ids = st.text_area(
                        "FVA reaction IDs (one per line)",
                        value="\n".join(default_fva_ids),
                        height=150,
                        disabled=not include_fva,
                        key=f"custom_fva_reactions_{widget_suffix}",
                    )
                    selected_fva_ids = list(
                        dict.fromkeys(
                            item.strip()
                            for item in raw_fva_ids.replace(",", "\n").splitlines()
                            if item.strip()
                        )
                    )
                    unknown_fva = [
                        reaction_id
                        for reaction_id in selected_fva_ids
                        if reaction_id not in label_by_id
                    ]
                    fva_selection_valid = not include_fva or (
                        bool(selected_fva_ids)
                        and len(selected_fva_ids) <= _CUSTOM_MODEL_MAX_FVA
                        and not unknown_fva
                    )
                    if include_fva and not fva_selection_valid:
                        st.error(f"Choose 1–{_CUSTOM_MODEL_MAX_FVA} valid reaction IDs for FVA.")
                fva_fraction = st.slider(
                    "Fraction of optimum retained",
                    min_value=0.50,
                    max_value=1.00,
                    value=0.95,
                    step=0.01,
                    disabled=not include_fva,
                    key=f"custom_fva_fraction_{widget_suffix}",
                )
                if include_fva and direction == "min" and fva_fraction < 1:
                    fva_selection_valid = False
                    st.error(
                        "Minimisation objectives require an FVA fraction of 1.00 in this workspace."
                    )

            analysis_signature = {
                "sha256": inspection.sha256,
                "objective": selected_objective,
                "direction": direction,
                "fva_reactions": selected_fva_ids if include_fva else [],
                "fva_fraction": fva_fraction if include_fva else None,
            }
            analysis_was_invalidated = bool(
                st.session_state.get("custom_analysis_signature")
                and st.session_state.get("custom_analysis_signature") != analysis_signature
            )
            if analysis_was_invalidated:
                _clear_custom_analysis_state()
                st.info("The analysis setup changed. Run again to create a fresh result.")
            if st.button(
                "Run custom-model analysis",
                type="primary",
                width="stretch",
                disabled=not objective_valid or (include_fva and not fva_selection_valid),
            ):
                _clear_custom_analysis_state()
                try:
                    with st.spinner("Solving the uploaded model…"):
                        custom_result = _core_analyse_custom_model(
                            inspection,
                            selected_objective,
                            direction=direction,
                            fva_reaction_ids=selected_fva_ids if include_fva else None,
                            fraction_of_optimum=fva_fraction,
                        )
                    st.session_state.custom_analysis = _plain(custom_result)
                    st.session_state.custom_analysis_signature = analysis_signature
                    st.session_state.custom_analysis_metadata = inspection.metadata()
                    st.session_state.custom_analysis_generated_at = datetime.now(UTC).isoformat()
                except (_CustomModelAnalysisError, ValueError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(
                        f"The custom analysis did not complete ({type(exc).__name__}). "
                        "Review the objective and model constraints."
                    )

            analysis = None
            if st.session_state.get("custom_analysis_signature") == analysis_signature:
                analysis = st.session_state.get("custom_analysis")

            if analysis:
                st.subheader("Analysis result")
                result_cols = st.columns(4)
                result_cols[0].metric("Solver status", str(analysis["status"]).title())
                objective_value = analysis.get("objective_value")
                result_cols[1].metric(
                    "Objective value",
                    f"{objective_value:.5g}" if objective_value is not None else "Unavailable",
                )
                nonzero_fluxes = sum(
                    abs(float(value)) > 1e-9 for value in analysis.get("fluxes", {}).values()
                )
                result_cols[2].metric("Non-zero fluxes", f"{nonzero_fluxes:,}")
                result_cols[3].metric("FVA reactions", f"{len(analysis.get('fva_ranges', {})):,}")

                raw_reaction_names = {
                    reaction.id: reaction.name or reaction.id
                    for reaction in inspection.model.reactions
                }
                reaction_names = {
                    reaction_id: _display_text(name)
                    for reaction_id, name in raw_reaction_names.items()
                }
                flux_frame = pd.DataFrame(
                    [
                        {
                            "Reaction": reaction_id,
                            "Name": reaction_names.get(reaction_id, reaction_id),
                            "Flux": float(flux),
                            "Absolute flux": abs(float(flux)),
                        }
                        for reaction_id, flux in analysis.get("fluxes", {}).items()
                    ],
                    columns=["Reaction", "Name", "Flux", "Absolute flux"],
                ).sort_values("Absolute flux", ascending=False)
                fva_frame = pd.DataFrame(list(analysis.get("fva_ranges", {}).values()))
                exchange_frame = pd.DataFrame(
                    [
                        {
                            "Reaction": reaction_id,
                            "Name": reaction_names.get(reaction_id, reaction_id),
                            "Lower bound": inspection.model.reactions.get_by_id(
                                reaction_id
                            ).lower_bound,
                            "Upper bound": inspection.model.reactions.get_by_id(
                                reaction_id
                            ).upper_bound,
                        }
                        for reaction_id in inspection.exchange_reaction_ids
                    ]
                )

                flux_tab, variability_tab, provenance_tab = st.tabs(
                    ["FBA fluxes", "FVA ranges", "Model provenance"]
                )
                with flux_tab:
                    st.caption("Sorted by absolute flux; zero-flux reactions are retained.")
                    st.dataframe(
                        flux_frame.drop(columns="Absolute flux"),
                        hide_index=True,
                        width="stretch",
                        height=430,
                    )
                with variability_tab:
                    if fva_frame.empty:
                        st.info("FVA was not requested for this run.")
                    else:
                        fva_frame = fva_frame.rename(
                            columns={
                                "reaction_id": "Reaction",
                                "reaction_name": "Name",
                                "minimum": "Minimum",
                                "maximum": "Maximum",
                            }
                        )
                        fva_frame["Span"] = fva_frame["Maximum"] - fva_frame["Minimum"]
                        fig = go.Figure(
                            go.Bar(
                                y=fva_frame["Reaction"],
                                x=fva_frame["Span"],
                                base=fva_frame["Minimum"],
                                orientation="h",
                                marker_color="#8ecfb1",
                                customdata=fva_frame["Maximum"],
                                hovertemplate=(
                                    "%{y}<br>range: %{base:.4g}–%{customdata:.4g}<extra></extra>"
                                ),
                            )
                        )
                        fig.update_layout(xaxis_title="Flux", yaxis_title=None)
                        st.plotly_chart(
                            _plot_layout(fig, max(320, 36 * len(fva_frame))),
                            width="stretch",
                            config={"displayModeBar": False},
                        )
                        st.dataframe(
                            fva_frame.drop(columns="Span"), hide_index=True, width="stretch"
                        )
                with provenance_tab:
                    st.json(inspection.metadata(), expanded=False)
                    if exchange_frame.empty:
                        st.info("COBRApy did not detect boundary exchange reactions.")
                    else:
                        st.markdown("#### Detected exchanges")
                        st.dataframe(exchange_frame, hide_index=True, width="stretch")

                export_payload = {
                    "metadata": inspection.metadata(),
                    "request": analysis_signature,
                    "analysis": analysis,
                    "environment": {
                        "app_version": APP_VERSION,
                        "cobra_version": COBRA_VERSION,
                        "solver_interface": getattr(
                            inspection.model.solver.interface,
                            "__name__",
                            str(inspection.model.solver.interface),
                        ),
                        "generated_at": st.session_state.get("custom_analysis_generated_at"),
                    },
                    "disclaimer": (
                        "Custom constraints were analysed as uploaded. No bundled "
                        "organism, medium, sensitivity or morphology assumptions were applied."
                    ),
                }
                export_flux_frame = flux_frame.drop(columns="Absolute flux").copy()
                export_flux_frame["Name"] = export_flux_frame["Reaction"].map(raw_reaction_names)
                export_flux_frame.insert(0, "Model SHA-256", inspection.sha256)
                export_flux_frame.insert(1, "Objective", selected_objective)
                export_flux_frame.insert(2, "Direction", direction)
                _download_pair(
                    export_flux_frame,
                    f"myco-optima_custom_{inspection.sha256[:8]}",
                    export_payload,
                )
                for warning in analysis.get("warnings", []):
                    st.warning(warning)

            with st.expander("Upload validation details"):
                st.json(inspection.metadata(), expanded=False)
                if st.button("Forget this upload"):
                    _clear_custom_model_state(clear_widgets=True)
                    st.session_state.custom_upload_key = (
                        st.session_state.get("custom_upload_key", 0) + 1
                    )
                    st.rerun()

elif page == "Media Optimiser":
    _page_intro(
        "Constraint-based design",
        "Media Optimiser",
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
        st.markdown("#### What the optimiser balances")
        with st.container(border=True):
            st.markdown(
                """
                **Carbon economy**

                Enough substrate for the objective without assuming infinite uptake.

                **Nitrogen sufficiency**

                Avoids a low-cost formulation that simply caps biomass.

                **Oxygen feasibility**

                Keeps aerobic capacity explicit instead of hiding it in a fixed recipe.
                """
            )
        with st.expander("How to interpret model units"):
            st.write(
                "Inputs are relative maximum-availability bounds, not flask "
                "concentrations. Mapping a recipe to these bounds requires measured "
                "uptake data or fitted kinetics."
            )
            st.caption(
                f"Objective · {st.session_state.objective}. "
                f"{OBJECTIVES[st.session_state.objective]}"
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

    summary_tab, flux_tab, alternatives_tab = st.tabs(
        ["Optimised bounds", "Flux envelope", "Alternatives"]
    )
    with summary_tab:
        display_composition = result["composition"].copy()
        display_composition["Recommended"] = display_composition["Recommended"].round(3)
        display_composition["Estimated cost"] = display_composition["Estimated cost"].round(2)
        st.dataframe(display_composition, hide_index=True, width="stretch")
        st.caption(f"Highest local sensitivity: **{result['limiting_nutrient']}**")
    with flux_tab:
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
    with alternatives_tab:
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
    st.warning(
        "Before cultivation, calibrate these bounds to uptake data and confirm units, "
        "solubility, osmolarity, oxygen-transfer capacity, strain auxotrophies and local "
        "biosafety requirements."
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
    with explanation_col, st.container(border=True):
        st.caption("SCREENING COMPRESSION")
        st.metric("Candidate grid → follow-up", f"81 → {run_count}")
        st.write(
            f"Model-guided shortlist for the current "
            f"_{catalog[settings['fungus_id']]['short']}_ scenario. Add controls, "
            "biological replicates and process-specific validation."
        )

    sensitivity, doe, sensitivity_note = _run_sensitivity(settings, run_count, perturbation / 100)
    sensitivity_is_core = bool(
        sensitivity_note and sensitivity_note.startswith("Core-connected sensitivity")
    )
    _model_notice(
        "COBRApy scientific core" if sensitivity_is_core else "Demo reduced-order model",
        sensitivity_note,
    )

    sensitivity_tab, design_space_tab = st.tabs(["Sensitivity ranking", "Design space"])
    with sensitivity_tab:
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
    with design_space_tab:
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
    st.dataframe(
        doe,
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
        with st.container(border=True):
            st.caption("MODEL-SUPPORTED MORPHOLOGY")
            st.markdown(f"### {prediction['morphology']}")
            st.metric("Evidence confidence", prediction["confidence"].title())
            st.metric("Top-class separation", f"{prediction['separation']:.2f}")
            st.caption("Qualitative class for this gene–media scenario.")
        st.warning(
            "Support scores are rule-weighted comparisons, not probabilities. Pellet "
            "size, rheology and productivity also depend on inoculum, shear, geometry "
            "and culture history."
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

elif page == "Interpretation & Methods":
    _page_intro(
        "Optional synthesis",
        "Interpretation & methods",
        "Review the modelling boundary and, if useful, ask Anthropic to turn an "
        "already-computed result into a concise engineering brief. Numerical results "
        "never depend on the language model.",
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

    with st.expander("Anthropic connection (optional)"):
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
        model_name = st.text_input(
            "Anthropic model", value=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        )
        st.caption("Only the selected structured result and focus prompt are sent on request.")
    api_key = session_api_key.strip() or configured_api_key

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


st.divider()
st.caption(
    "myco-optima · Decision support for fungal fermentation · Model outputs are hypotheses, not cultivation instructions."
)
