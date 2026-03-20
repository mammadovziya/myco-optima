"""Build transparent reduced-order COBRApy models for four industrial fungi.

The generated models are curated stoichiometric teaching surrogates.  They are
not validated genome-scale reconstructions, do not encode regulation or
concentration-to-uptake kinetics, and must be calibrated before quantitative
process use.  Their small size is intentional: every source, sink and
assumption can be inspected in a hackathon demonstration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cobra import Metabolite, Model, Reaction

from .catalog import MODEL_DISCLAIMER, NUTRIENTS, FungusProfile, get_fungus, resolve_medium

CARBON_EQUIVALENTS = {
    "glucose": 6.0,
    "xylose": 5.0,
    "glycerol": 3.0,
    "sucrose": 12.0,
    "maltose": 12.0,
}

ASSIMILATION_REACTIONS = {
    "glucose": "ASSIM_GLC",
    "xylose": "ASSIM_XYL",
    "glycerol": "ASSIM_GLYC",
    "sucrose": "ASSIM_SUCR",
    "maltose": "ASSIM_MALT",
    "ammonium": "ASSIM_NH4",
    "nitrate": "ASSIM_NO3",
    "urea": "ASSIM_UREA",
    "sulfate": "ASSIM_SO4",
}

BIOMASS_REACTION = "BIOMASS"
CORE_REACTIONS = (BIOMASS_REACTION, "RESP", "ATPM")


def _reaction(
    reaction_id: str,
    name: str,
    stoichiometry: Mapping[Metabolite, float],
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1000.0,
    subsystem: str = "Reduced-order surrogate",
) -> Reaction:
    reaction = Reaction(reaction_id, name=name, lower_bound=lower_bound, upper_bound=upper_bound)
    reaction.add_metabolites(dict(stoichiometry))
    reaction.subsystem = subsystem
    reaction.annotation["sbo"] = "SBO:0000176"  # biochemical reaction (illustrative)
    return reaction


def _cytosolic_id(extracellular_id: str) -> str:
    if not extracellular_id.endswith("_e"):
        raise ValueError(f"Expected an extracellular metabolite id, got {extracellular_id!r}")
    return f"{extracellular_id[:-2]}_c"


def build_model(
    fungus: str | FungusProfile,
    medium: Mapping[str, float] | None = None,
) -> Model:
    """Construct a fresh COBRApy model and apply a medium scenario.

    Parameters
    ----------
    fungus:
        Catalogue id, scientific/short name, or :class:`FungusProfile`.
    medium:
        Optional partial scenario.  Source categories are exclusive: selecting
        xylose closes baseline glucose, and selecting nitrate closes baseline
        ammonium.  Other omitted essentials retain their profile defaults.

    Returns
    -------
    cobra.Model
        An independent model with finite bounds and biomass as its objective.
    """

    profile = get_fungus(fungus)
    model = Model(f"myco_optima_{profile.id}", name=f"{profile.name} reduced-order surrogate")
    model.compartments = {
        "e": "extracellular availability pool",
        "c": "cytosolic equivalent pool",
    }
    model.notes.update(
        {
            "fungus_id": profile.id,
            "scope": "curated reduced-order stoichiometric teaching surrogate",
            "disclaimer": MODEL_DISCLAIMER,
            "availability_unit": "relative maximum uptake; not concentration",
            "carbon_efficiencies": ", ".join(
                f"{key}={value:g}" for key, value in profile.carbon_efficiencies.items()
            ),
        }
    )

    extracellular: dict[str, Metabolite] = {}
    cytosolic: dict[str, Metabolite] = {}
    reactions: list[Reaction] = []

    for nutrient in NUTRIENTS.values():
        external = Metabolite(
            nutrient.metabolite_id,
            name=f"{nutrient.name} availability",
            compartment="e",
        )
        internal = Metabolite(
            _cytosolic_id(nutrient.metabolite_id),
            name=f"{nutrient.name} equivalent",
            compartment="c",
        )
        extracellular[nutrient.id] = external
        cytosolic[nutrient.id] = internal

        exchange = _reaction(
            nutrient.exchange_id,
            f"{nutrient.name} exchange",
            {external: -1.0},
            upper_bound=1000.0,
            subsystem="Boundary",
        )
        exchange.annotation["sbo"] = "SBO:0000627"
        transport = _reaction(
            f"T_{nutrient.id.upper()}",
            f"{nutrient.name} uptake",
            {external: -1.0, internal: 1.0},
            upper_bound=profile.uptake_capacities[nutrient.id],
            subsystem="Transport",
        )
        reactions.extend((exchange, transport))

    carbon = Metabolite("c1_c", name="one-carbon biomass precursor equivalent", compartment="c")
    nitrogen = Metabolite("n1_c", name="one-nitrogen precursor equivalent", compartment="c")
    sulfur = Metabolite("s1_c", name="one-sulfur precursor equivalent", compartment="c")
    energy = Metabolite("energy_c", name="generic biosynthetic energy equivalent", compartment="c")
    co2_c = Metabolite("co2_c", name="carbon dioxide", compartment="c")
    co2_e = Metabolite("co2_e", name="secreted carbon dioxide", compartment="e")

    for nutrient_id, carbon_atoms in CARBON_EQUIVALENTS.items():
        energy_cost = 0.0 if nutrient_id in {"glucose", "glycerol"} else 0.15
        efficiency = profile.carbon_efficiencies[nutrient_id]
        stoichiometry: dict[Metabolite, float] = {
            cytosolic[nutrient_id]: -1.0,
            carbon: carbon_atoms * efficiency,
        }
        if energy_cost:
            stoichiometry[energy] = -energy_cost
        reactions.append(
            _reaction(
                ASSIMILATION_REACTIONS[nutrient_id],
                f"{NUTRIENTS[nutrient_id].name} assimilation",
                stoichiometry,
                subsystem="Carbon assimilation",
            )
        )

    reactions.extend(
        [
            _reaction(
                "ASSIM_NH4",
                "Ammonium assimilation",
                {cytosolic["ammonium"]: -1.0, nitrogen: 1.0},
                subsystem="Nitrogen assimilation",
            ),
            _reaction(
                "ASSIM_NO3",
                "Nitrate assimilation",
                {cytosolic["nitrate"]: -1.0, energy: -2.0, nitrogen: 1.0},
                subsystem="Nitrogen assimilation",
            ),
            _reaction(
                "ASSIM_UREA",
                "Urea-equivalent assimilation",
                {cytosolic["urea"]: -1.0, energy: -0.25, nitrogen: 2.0},
                subsystem="Nitrogen assimilation",
            ),
            _reaction(
                "ASSIM_SO4",
                "Sulfate assimilation",
                {cytosolic["sulfate"]: -1.0, energy: -1.0, sulfur: 1.0},
                subsystem="Sulfur assimilation",
            ),
            _reaction(
                "RESP",
                "Aerobic respiration equivalent",
                {carbon: -1.0, cytosolic["oxygen"]: -1.0, co2_c: 1.0, energy: 4.0},
                subsystem="Energy metabolism",
            ),
            _reaction(
                "T_CO2",
                "Carbon dioxide secretion",
                {co2_c: -1.0, co2_e: 1.0},
                subsystem="Transport",
            ),
            _reaction(
                "EX_co2_e",
                "Carbon dioxide exchange",
                {co2_e: -1.0},
                subsystem="Boundary",
            ),
            _reaction(
                "ATPM",
                "Optional generic energy maintenance sink",
                {energy: -1.0},
                subsystem="Maintenance",
            ),
        ]
    )

    coefficients = profile.biomass_coefficients
    biomass = _reaction(
        BIOMASS_REACTION,
        "Fungal biomass pseudo-reaction",
        {
            carbon: -coefficients["carbon"],
            nitrogen: -coefficients["nitrogen"],
            cytosolic["phosphate"]: -coefficients["phosphorus"],
            sulfur: -coefficients["sulfur"],
            cytosolic["magnesium"]: -coefficients["magnesium"],
            cytosolic["iron"]: -coefficients["iron"],
            cytosolic["zinc"]: -coefficients["zinc"],
            energy: -coefficients["energy"],
        },
        subsystem="Biomass pseudo-reaction",
    )
    biomass.annotation["sbo"] = "SBO:0000629"
    biomass.notes["disclaimer"] = "Pseudo-reaction; coefficients are illustrative equivalents."
    reactions.append(biomass)

    product_c = Metabolite(
        f"{profile.signature_product.replace('-', '_')}_c",
        name=profile.signature_product,
        compartment="c",
    )
    product_e = Metabolite(
        f"{profile.signature_product.replace('-', '_')}_e",
        name=f"secreted {profile.signature_product}",
        compartment="e",
    )
    if profile.id == "aspergillus_niger":
        product_stoichiometry = {carbon: -6.0, energy: -2.0, product_c: 1.0}
    else:
        product_stoichiometry = {carbon: -4.0, nitrogen: -0.8, energy: -8.0, product_c: 1.0}
    reactions.extend(
        [
            _reaction(
                profile.signature_reaction,
                f"{profile.signature_product} synthesis burden",
                product_stoichiometry,
                subsystem="Illustrative product sink",
            ),
            _reaction(
                "T_PRODUCT",
                f"{profile.signature_product} secretion",
                {product_c: -1.0, product_e: 1.0},
                subsystem="Transport",
            ),
            _reaction(
                "EX_product_e",
                f"{profile.signature_product} exchange",
                {product_e: -1.0},
                subsystem="Boundary",
            ),
        ]
    )

    model.add_reactions(reactions)
    model.objective = BIOMASS_REACTION
    model.objective_direction = "max"
    return apply_medium(model, profile.default_medium if medium is None else medium, inplace=True)


def get_model_fungus(model: Model) -> FungusProfile:
    """Return the catalogue profile recorded in a generated model."""

    fungus_id = model.notes.get("fungus_id")
    if not isinstance(fungus_id, str):
        raise ValueError("Model does not contain a myco-optima fungus_id note.")
    return get_fungus(fungus_id)


def apply_medium(
    model: Model,
    medium: Mapping[str, float] | None,
    *,
    inplace: bool = False,
) -> Model:
    """Apply validated exchange availability bounds to a model.

    A copy is returned by default.  Requested values are capped at the
    species-specific transport capacity so all optimisation bounds remain
    finite.  Passing ``None`` preserves the model's current bounds.
    """

    target = model if inplace else model.copy()
    if medium is None:
        return target

    profile = get_model_fungus(target)
    resolved = resolve_medium(profile, medium)
    for nutrient_id, nutrient in NUTRIENTS.items():
        amount = min(resolved[nutrient_id], profile.uptake_capacities[nutrient_id])
        reaction = target.reactions.get_by_id(nutrient.exchange_id)
        reaction.lower_bound = -amount
        reaction.upper_bound = 1000.0
    return target


def effective_medium(model: Model) -> dict[str, float]:
    """Read the currently permitted uptake for each canonical nutrient."""

    return {
        nutrient_id: max(0.0, -float(model.reactions.get_by_id(item.exchange_id).lower_bound))
        for nutrient_id, item in NUTRIENTS.items()
    }


def build_all_models() -> dict[str, Model]:
    """Build one independent baseline model per catalogue entry."""

    from .catalog import list_fungi

    return {profile.id: build_model(profile) for profile in list_fungi()}


def model_summary(model: Model) -> dict[str, Any]:
    """Return small, serialization-safe structural metadata for inspection."""

    profile = get_model_fungus(model)
    return {
        "id": model.id,
        "fungus_id": profile.id,
        "reactions": len(model.reactions),
        "metabolites": len(model.metabolites),
        "genes": len(model.genes),
        "objective": BIOMASS_REACTION,
        "disclaimer": MODEL_DISCLAIMER,
    }
