"""Curated configuration for the four myco-optima teaching models.

These profiles deliberately describe small, inspectable stoichiometric
surrogates.  They are not validated genome-scale reconstructions, and their
availability values must not be treated as fitted concentration-to-uptake
kinetics.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .types import FungusProfile, Nutrient

MODEL_DISCLAIMER = (
    "Curated reduced-order teaching model; not a validated genome-scale reconstruction "
    "or a substitute for strain-specific cultivation data."
)

AVAILABILITY_UNIT = "relative maximum uptake"


def _nutrient(
    nutrient_id: str,
    name: str,
    metabolite_id: str,
    category: str,
    cost: float,
    description: str,
    *aliases: str,
) -> Nutrient:
    return Nutrient(
        id=nutrient_id,
        name=name,
        exchange_id=f"EX_{metabolite_id}",
        metabolite_id=metabolite_id,
        category=category,
        unit=AVAILABILITY_UNIT,
        cost_per_unit=cost,
        description=description,
        aliases=tuple(aliases),
    )


NUTRIENTS: dict[str, Nutrient] = {
    "glucose": _nutrient(
        "glucose",
        "D-glucose",
        "glc__D_e",
        "carbon",
        0.72,
        "Six-carbon reference substrate.",
        "glc",
        "d-glucose",
        "carbon",
    ),
    "xylose": _nutrient(
        "xylose",
        "D-xylose",
        "xyl__D_e",
        "carbon",
        0.86,
        "Five-carbon lignocellulosic sugar.",
        "xyl",
        "d-xylose",
    ),
    "glycerol": _nutrient(
        "glycerol",
        "Glycerol",
        "glyc_e",
        "carbon",
        0.94,
        "Three-carbon polyol.",
        "glyc",
    ),
    "sucrose": _nutrient(
        "sucrose",
        "Sucrose",
        "sucr_e",
        "carbon",
        0.81,
        "Twelve-carbon disaccharide represented as carbon equivalents.",
        "sucr",
    ),
    "maltose": _nutrient(
        "maltose",
        "Maltose",
        "malt_e",
        "carbon",
        1.12,
        "Twelve-carbon disaccharide relevant to Aspergillus processes.",
        "malt",
    ),
    "ammonium": _nutrient(
        "ammonium",
        "Ammonium",
        "nh4_e",
        "nitrogen",
        0.38,
        "Reduced inorganic nitrogen reference source.",
        "nh4",
        "ammonia",
        "ammonium sulfate",
        "ammonium sulphate",
        "nitrogen",
    ),
    "nitrate": _nutrient(
        "nitrate",
        "Nitrate",
        "no3_e",
        "nitrogen",
        0.68,
        "Oxidised nitrogen source with an explicit energy cost.",
        "no3",
    ),
    "urea": _nutrient(
        "urea",
        "Urea",
        "urea_e",
        "nitrogen",
        0.52,
        "Two-nitrogen organic source represented by a pseudo-assimilation reaction.",
    ),
    "phosphate": _nutrient(
        "phosphate",
        "Phosphate",
        "pi_e",
        "phosphorus",
        1.05,
        "Phosphorus availability used directly by the biomass pseudo-reaction.",
        "pi",
        "kh2po4",
        "kh₂po₄",
    ),
    "sulfate": _nutrient(
        "sulfate",
        "Sulfate",
        "so4_e",
        "sulfur",
        0.74,
        "Sulfur source with a small assimilation-energy cost.",
        "so4",
        "sulphate",
    ),
    "oxygen": _nutrient(
        "oxygen",
        "Oxygen",
        "o2_e",
        "oxygen",
        0.12,
        "Electron-acceptor availability; not dissolved-oxygen concentration.",
        "o2",
        "o₂",
        "oxygen uptake",
        "o2 uptake",
        "o₂ uptake",
    ),
    "magnesium": _nutrient(
        "magnesium",
        "Magnesium",
        "mg2_e",
        "trace",
        0.74,
        "Essential trace-element availability.",
        "mg",
        "mg2",
        "mg²+",
    ),
    "iron": _nutrient(
        "iron",
        "Iron",
        "fe2_e",
        "trace",
        3.80,
        "Essential trace-element availability.",
        "fe",
        "fe2",
        "fe²+",
    ),
    "zinc": _nutrient(
        "zinc",
        "Zinc",
        "zn2_e",
        "trace",
        3.80,
        "Essential trace-element availability.",
        "zn",
        "zn2",
        "zn²+",
    ),
}


_COMMON_CAPACITIES = {
    "glucose": 40.0,
    "xylose": 40.0,
    "glycerol": 40.0,
    "sucrose": 20.0,
    "maltose": 20.0,
    "ammonium": 20.0,
    "nitrate": 15.0,
    "urea": 10.0,
    "phosphate": 5.0,
    "sulfate": 5.0,
    "oxygen": 60.0,
    "magnesium": 1.0,
    "iron": 0.2,
    "zinc": 0.2,
}


def _profile(
    fungus_id: str,
    name: str,
    short_name: str,
    industrial_use: str,
    signature_product: str,
    signature_reaction: str,
    temperature: float,
    ph: float,
    accent: str,
    default_medium: dict[str, float],
    carbon_efficiencies: dict[str, float],
    biomass_nitrogen: float,
    biomass_energy: float,
    *,
    capacity_overrides: Mapping[str, float] | None = None,
    notes: tuple[str, ...] = (),
) -> FungusProfile:
    capacities = dict(_COMMON_CAPACITIES)
    capacities.update(capacity_overrides or {})
    return FungusProfile(
        id=fungus_id,
        name=name,
        short_name=short_name,
        industrial_use=industrial_use,
        signature_product=signature_product,
        signature_reaction=signature_reaction,
        temperature=temperature,
        ph=ph,
        accent=accent,
        default_medium=dict(default_medium),
        carbon_efficiencies=dict(carbon_efficiencies),
        uptake_capacities=capacities,
        biomass_coefficients={
            "carbon": 1.0,
            "nitrogen": biomass_nitrogen,
            "phosphorus": 0.015,
            "sulfur": 0.006,
            "magnesium": 0.002,
            "iron": 0.0002,
            "zinc": 0.00005,
            "energy": biomass_energy,
        },
        notes=(MODEL_DISCLAIMER, *notes),
    )


_BASE_MINERALS = {
    "phosphate": 0.8,
    "sulfate": 0.35,
    "oxygen": 20.0,
    "magnesium": 0.10,
    "iron": 0.010,
    "zinc": 0.005,
}


FUNGI: dict[str, FungusProfile] = {
    "aspergillus_niger": _profile(
        "aspergillus_niger",
        "Aspergillus niger",
        "A. niger",
        "Organic acids and industrial enzymes",
        "citrate-equivalent",
        "CITRATE_SYN",
        30.0,
        4.5,
        "#1B8F6B",
        {"glucose": 10.0, "ammonium": 7.0, **_BASE_MINERALS},
        {"glucose": 1.00, "xylose": 0.88, "glycerol": 0.75, "sucrose": 0.95, "maltose": 0.90},
        0.18,
        1.60,
        notes=("Citrate synthesis is a carbon/energy sink, not a mechanistic TCA-cycle model.",),
    ),
    "aspergillus_oryzae": _profile(
        "aspergillus_oryzae",
        "Aspergillus oryzae",
        "A. oryzae",
        "Food fermentation and amylolytic enzymes",
        "amylase-equivalent",
        "AMYLASE_SYN",
        30.0,
        5.5,
        "#D89A39",
        {"maltose": 5.0, "ammonium": 7.5, **_BASE_MINERALS},
        {"glucose": 0.95, "xylose": 0.55, "glycerol": 0.70, "sucrose": 1.00, "maltose": 1.05},
        0.20,
        1.70,
        notes=("Amylase synthesis is an illustrative secretory burden.",),
    ),
    "trichoderma_reesei": _profile(
        "trichoderma_reesei",
        "Trichoderma reesei",
        "T. reesei",
        "Cellulases and lignocellulosic biorefining",
        "cellulase-equivalent",
        "CELLULASE_SYN",
        28.0,
        5.0,
        "#397D65",
        {"glucose": 8.0, "xylose": 2.0, "ammonium": 8.0, **_BASE_MINERALS},
        {"glucose": 0.90, "xylose": 1.00, "glycerol": 0.55, "sucrose": 0.45, "maltose": 0.60},
        0.20,
        1.80,
        capacity_overrides={"sucrose": 8.0, "maltose": 8.0},
        notes=("The model does not represent cellulase induction or cellulose hydrolysis.",),
    ),
    "fusarium_venenatum": _profile(
        "fusarium_venenatum",
        "Fusarium venenatum",
        "F. venenatum",
        "Mycoprotein and fungal biomass",
        "protein-equivalent",
        "PROTEIN_SYN",
        28.0,
        6.0,
        "#8367A8",
        {"glucose": 10.0, "ammonium": 10.0, **_BASE_MINERALS},
        {"glucose": 1.00, "xylose": 0.72, "glycerol": 0.70, "sucrose": 0.60, "maltose": 0.55},
        0.24,
        1.90,
        capacity_overrides={"nitrate": 4.0, "urea": 5.0},
        notes=("Protein synthesis is an illustrative nitrogen/energy sink.",),
    ),
}


CONTEXT_KEYS = {
    "ph",
    "temperature",
    "temperature_c",
    "agitation",
    "agitation_rpm",
    "shear",
    "carbon_source",
    "nitrogen_source",
}


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


_FUNGUS_ALIASES: dict[str, str] = {}
for _fungus_id, _fungus in FUNGI.items():
    for _alias in (_fungus_id, _fungus.name, _fungus.short_name, _fungus.name.replace(" ", "_")):
        _FUNGUS_ALIASES[_key(_alias)] = _fungus_id


_NUTRIENT_ALIASES: dict[str, str] = {}
for _nutrient_id, _item in NUTRIENTS.items():
    for _alias in (_nutrient_id, _item.name, *_item.aliases):
        _NUTRIENT_ALIASES[_key(_alias)] = _nutrient_id


def list_fungi() -> tuple[FungusProfile, ...]:
    """Return the four supported fungi in a stable display order."""

    return tuple(FUNGI.values())


def get_fungus(fungus: str | FungusProfile) -> FungusProfile:
    """Resolve a fungus id, scientific name, short name, or existing profile."""

    if isinstance(fungus, FungusProfile):
        return fungus
    try:
        return FUNGI[_FUNGUS_ALIASES[_key(fungus)]]
    except KeyError as exc:
        supported = ", ".join(FUNGI)
        raise KeyError(f"Unknown fungus {fungus!r}. Supported ids: {supported}.") from exc


def list_nutrients(category: str | None = None) -> tuple[Nutrient, ...]:
    """Return modelled nutrients, optionally filtered by category."""

    if category is None:
        return tuple(NUTRIENTS.values())
    category_key = _key(category)
    return tuple(item for item in NUTRIENTS.values() if _key(item.category) == category_key)


def get_nutrient(nutrient: str | Nutrient) -> Nutrient:
    """Resolve a nutrient id or common display alias."""

    if isinstance(nutrient, Nutrient):
        return nutrient
    try:
        return NUTRIENTS[_NUTRIENT_ALIASES[_key(nutrient)]]
    except KeyError as exc:
        raise KeyError(f"Unknown medium component {nutrient!r}.") from exc


def normalise_medium(
    medium: Mapping[str, float] | None,
    *,
    allow_context: bool = True,
) -> dict[str, float]:
    """Convert display aliases to canonical nutrient ids and validate amounts.

    Recognised process-context fields such as pH and temperature are ignored:
    they are not stoichiometric exchange constraints.  Composite display labels
    used by the UI are conservatively mapped to their primary modelled nutrient.
    """

    if medium is None:
        return {}
    if not isinstance(medium, Mapping):
        raise TypeError("medium must be a mapping of component names to non-negative amounts")

    result: dict[str, float] = {}
    for raw_name, raw_amount in medium.items():
        name_key = _key(raw_name)
        if allow_context and name_key in CONTEXT_KEYS:
            continue
        # Composite ingredients are intentionally reduced to the factor varied
        # by the UI; counter-ion uptake remains at the profile baseline.
        if name_key in {"mgso4", "mgso_4"}:
            nutrient_id = "magnesium"
        elif name_key in {"trace_elements", "trace_element_mix", "micronutrients"}:
            amount = _finite_nonnegative(raw_name, raw_amount)
            result["iron"] = amount
            result["zinc"] = amount
            continue
        else:
            try:
                nutrient_id = _NUTRIENT_ALIASES[name_key]
            except KeyError as exc:
                raise KeyError(f"Unknown medium component {raw_name!r}.") from exc
        result[nutrient_id] = _finite_nonnegative(raw_name, raw_amount)
    return result


def _finite_nonnegative(name: Any, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Amount for {name!r} must be a real number.") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Amount for {name!r} must be finite and non-negative.")
    return number


def resolve_medium(
    fungus: str | FungusProfile,
    medium: Mapping[str, float] | None = None,
    *,
    replace_source_categories: bool = True,
) -> dict[str, float]:
    """Overlay a scenario on the species baseline and return every nutrient.

    Providing any carbon source closes the other carbon sources; the same rule
    applies to nitrogen sources.  This prevents a selected xylose or nitrate
    scenario from silently retaining baseline glucose or ammonium.
    """

    profile = get_fungus(fungus)
    resolved = {nutrient_id: 0.0 for nutrient_id in NUTRIENTS}
    resolved.update(profile.default_medium)
    overrides = normalise_medium(medium)

    if replace_source_categories:
        represented = {NUTRIENTS[item].category for item in overrides}
        for category in ("carbon", "nitrogen"):
            if category in represented:
                for nutrient_id, nutrient in NUTRIENTS.items():
                    if nutrient.category == category:
                        resolved[nutrient_id] = 0.0
    resolved.update(overrides)
    return resolved
