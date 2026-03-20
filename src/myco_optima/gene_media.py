"""Deterministic gene–media morphology hypotheses for filamentous fungi.

This module is deliberately separate from the COBRA model: gene regulation and
pellet morphology are not inferred by FBA.  The rules below are a small,
inspectable teaching set backed by named studies where available.  Outputs are
qualitative hypotheses, not validated phenotype predictions, and support
scores are not probabilities.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .catalog import MODEL_DISCLAIMER, FungusProfile, get_fungus
from .types import GeneInteraction, MorphologyPrediction


@dataclass(frozen=True)
class _Rule:
    id: str
    genes: tuple[str, ...]
    states: tuple[str, ...]
    effects: dict[str, float]
    explanation: str
    evidence_url: str | None
    confidence: str
    require_all: bool = True
    carbon_sources: tuple[str, ...] = ()


RACA_EVIDENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3722221/"
ARFA_EVIDENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5952172/"
RAC1_EVIDENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6798449/"
AO_AGGREGATION_EVIDENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6753227/"


_RULES: dict[str, tuple[_Rule, ...]] = {
    "aspergillus_niger": (
        _Rule(
            id="an_raca_loss",
            genes=("raca",),
            states=("knockout", "knockdown"),
            effects={"branching": 2.0, "aggregation": -2.0},
            explanation="Reduced RacA activity supports a hyperbranched, more dispersed tendency.",
            evidence_url=RACA_EVIDENCE,
            confidence="moderate",
        ),
        _Rule(
            id="an_arfa_glucose_up",
            genes=("arfa",),
            states=("overexpressed",),
            effects={"pellet_size": -1.0, "aggregation": -1.0, "secretion": 1.0},
            explanation="ArfA over-expression under glucose is represented as smaller, looser growth with higher secretory tendency.",
            evidence_url=ARFA_EVIDENCE,
            confidence="moderate",
            carbon_sources=("glucose",),
        ),
    ),
    "trichoderma_reesei": (
        _Rule(
            id="tr_rac1_loss",
            genes=("rac1",),
            states=("knockout", "knockdown"),
            effects={"branching": 2.0, "aggregation": -1.0},
            explanation="Reduced Rac1 activity supports hyperbranching and a more dispersed tendency.",
            evidence_url=RAC1_EVIDENCE,
            confidence="moderate",
        ),
        _Rule(
            id="tr_rac1_lactose_interaction",
            genes=("rac1",),
            states=("knockout", "knockdown"),
            effects={"secretion": 2.0},
            explanation="The Rac1-loss secretion effect is applied only when lactose is the represented carbon source.",
            evidence_url=RAC1_EVIDENCE,
            confidence="moderate",
            carbon_sources=("lactose",),
        ),
        _Rule(
            id="tr_gul1_loss",
            genes=("gul1",),
            states=("knockout", "knockdown"),
            effects={"branching": 2.0, "viscosity": -2.0},
            explanation="Reduced Gul1 is represented as increased lateral branching and lower viscosity tendency.",
            evidence_url=None,
            confidence="low",
        ),
    ),
    "aspergillus_oryzae": (
        *(
            _Rule(
                id=f"ao_{gene}_loss",
                genes=(gene,),
                states=("knockout", "knockdown"),
                effects={"pellet_size": -0.7, "aggregation": -0.4},
                explanation=f"Reduced {gene.upper()} activity contributes to a smaller-pellet tendency.",
                evidence_url=AO_AGGREGATION_EVIDENCE,
                confidence="moderate",
            )
            for gene in ("agsa", "agsb", "agsc")
        ),
        _Rule(
            id="ao_gag_pathway_combined_loss",
            genes=("agsa", "agsb", "agsc", "sphz", "ugez"),
            states=("knockout", "knockdown"),
            effects={"aggregation": -3.0, "branching": 2.0, "pellet_size": -1.5},
            explanation="Combined AG/GAG-pathway disruption strongly supports dispersed hyphae in this rule set.",
            evidence_url=AO_AGGREGATION_EVIDENCE,
            confidence="moderate",
        ),
        _Rule(
            id="ao_nsdc_loss",
            genes=("nsdc",),
            states=("knockout", "knockdown"),
            effects={"branching": 2.0, "aggregation": -1.0},
            explanation="NsdC loss is represented as a hyperbranched, dispersed-clump tendency.",
            evidence_url=None,
            confidence="low",
        ),
    ),
    # No species-specific regulatory effect is asserted for F. venenatum.
    "fusarium_venenatum": (),
}


_BASE_LATENT: dict[str, dict[str, float]] = {
    "aspergillus_niger": {
        "branching": 0.0,
        "aggregation": 0.5,
        "pellet_size": 0.5,
        "fragmentation": 0.0,
        "secretion": 0.0,
        "viscosity": 0.0,
        "stress": 0.0,
    },
    "aspergillus_oryzae": {
        "branching": 0.2,
        "aggregation": 0.5,
        "pellet_size": 0.4,
        "fragmentation": 0.0,
        "secretion": 0.0,
        "viscosity": 0.0,
        "stress": 0.0,
    },
    "trichoderma_reesei": {
        "branching": 0.6,
        "aggregation": 0.0,
        "pellet_size": 0.0,
        "fragmentation": 0.2,
        "secretion": 0.2,
        "viscosity": 0.0,
        "stress": 0.0,
    },
    "fusarium_venenatum": {
        "branching": 0.5,
        "aggregation": 0.1,
        "pellet_size": 0.0,
        "fragmentation": 0.2,
        "secretion": 0.0,
        "viscosity": 0.0,
        "stress": 0.0,
    },
}


_SPECIES_SUPPORT_BIAS: dict[str, dict[str, float]] = {
    "aspergillus_niger": {
        "Compact pellets": 0.3,
        "Loose pellets": 0.1,
        "Dispersed hyphae": 0.0,
        "Dense clumps": 0.0,
    },
    "aspergillus_oryzae": {
        "Compact pellets": 0.1,
        "Loose pellets": 0.3,
        "Dispersed hyphae": 0.0,
        "Dense clumps": 0.0,
    },
    "trichoderma_reesei": {
        "Compact pellets": 0.0,
        "Loose pellets": 0.1,
        "Dispersed hyphae": 0.4,
        "Dense clumps": 0.0,
    },
    "fusarium_venenatum": {
        "Compact pellets": 0.0,
        "Loose pellets": 0.1,
        "Dispersed hyphae": 0.3,
        "Dense clumps": 0.1,
    },
}


_STATE_ALIASES = {
    "native": "native",
    "wild_type": "native",
    "wildtype": "native",
    "wt": "native",
    "unmodified": "native",
    "knockout": "knockout",
    "knock_out": "knockout",
    "ko": "knockout",
    "deleted": "knockout",
    "deletion": "knockout",
    "knockdown": "knockdown",
    "knock_down": "knockdown",
    "kd": "knockdown",
    "down": "knockdown",
    "repressed": "knockdown",
    "overexpressed": "overexpressed",
    "over_expressed": "overexpressed",
    "overexpression": "overexpressed",
    "up": "overexpressed",
}


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _state(value: Any) -> str:
    try:
        return _STATE_ALIASES[_key(value)]
    except KeyError as exc:
        valid = "native, knockout, knockdown, overexpressed"
        raise ValueError(f"Unknown gene state {value!r}; use one of {valid}.") from exc


def _amount(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _media_context(medium: Mapping[str, Any] | None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "carbon_sources": set(),
        "carbon": None,
        "nitrogen": None,
        "oxygen": None,
        "ph": None,
        "agitation": None,
        "shear": None,
    }
    if not medium:
        return context

    carbon_aliases = {
        "glucose": "glucose",
        "d_glucose": "glucose",
        "xylose": "xylose",
        "glycerol": "glycerol",
        "sucrose": "sucrose",
        "maltose": "maltose",
        "lactose": "lactose",
        "cellulose": "cellulose",
        "cellobiose": "cellobiose",
    }
    nitrogen_keys = {"nitrogen", "ammonium", "ammonium_sulfate", "nitrate", "urea"}
    for raw_name, raw_value in medium.items():
        name = _key(raw_name)
        amount = _amount(raw_value)
        if name in carbon_aliases and (amount is None or amount > 0):
            context["carbon_sources"].add(carbon_aliases[name])
            if amount is not None:
                context["carbon"] = (context["carbon"] or 0.0) + amount
        elif name == "carbon" and amount is not None:
            context["carbon"] = amount
        elif name in nitrogen_keys and amount is not None:
            context["nitrogen"] = (context["nitrogen"] or 0.0) + amount
        elif name in {"oxygen", "o2", "oxygen_uptake", "o2_uptake"}:
            context["oxygen"] = amount
        elif name == "ph":
            context["ph"] = amount
        elif name in {"agitation", "agitation_rpm"}:
            context["agitation"] = amount
        elif name == "shear":
            context["shear"] = amount
    return context


def _rule_applies(rule: _Rule, states: Mapping[str, str], carbon_sources: set[str]) -> bool:
    gene_matches = [states.get(gene) in rule.states for gene in rule.genes]
    genes_apply = all(gene_matches) if rule.require_all else any(gene_matches)
    carbon_applies = not rule.carbon_sources or bool(
        carbon_sources.intersection(rule.carbon_sources)
    )
    return genes_apply and carbon_applies


def _support_scores(profile: FungusProfile, latent: Mapping[str, float]) -> dict[str, float]:
    branching = latent["branching"]
    aggregation = latent["aggregation"]
    pellet_size = latent["pellet_size"]
    fragmentation = latent["fragmentation"]
    bias = _SPECIES_SUPPORT_BIAS[profile.id]
    scores = {
        "Compact pellets": 1.0
        + 0.8 * aggregation
        + 0.8 * pellet_size
        - 0.5 * branching
        - 0.4 * fragmentation,
        "Loose pellets": 1.0
        + 0.3 * aggregation
        + 0.4 * branching
        + 0.1 * pellet_size
        + 0.2 * fragmentation,
        "Dispersed hyphae": 1.0
        + 0.9 * branching
        + 0.8 * fragmentation
        - 0.7 * aggregation
        - 0.5 * pellet_size,
        "Dense clumps": 1.0
        + 0.9 * aggregation
        + 0.25 * branching
        - 0.5 * fragmentation
        - 0.2 * pellet_size,
    }
    return {name: round(max(0.0, score + bias[name]), 4) for name, score in scores.items()}


def predict_morphology(
    fungus: str | FungusProfile,
    medium: Mapping[str, Any] | None = None,
    gene_states: Mapping[str, Any] | None = None,
) -> MorphologyPrediction:
    """Return a qualitative morphology class with a complete rule trace.

    Unknown interventions never receive invented effects.  They are listed in
    ``insufficient_evidence`` and leave the latent scores unchanged.
    """

    profile = get_fungus(fungus)
    states = {_key(gene): _state(state) for gene, state in (gene_states or {}).items()}
    context = _media_context(medium)
    carbon_sources: set[str] = context["carbon_sources"]
    latent = dict(_BASE_LATENT[profile.id])
    trace: list[GeneInteraction] = []
    drivers: list[str] = []
    warnings = [
        MODEL_DISCLAIMER,
        "Morphology rules are qualitative hypotheses; support scores are not probabilities.",
    ]

    rules = _RULES[profile.id]
    supported_genes = {gene for rule in rules for gene in rule.genes}
    insufficient: list[str] = []
    for gene, state in states.items():
        if state == "native":
            continue
        gene_rules = [rule for rule in rules if gene in rule.genes]
        if not gene_rules:
            insufficient.append(f"{gene} {state}: no species-specific rule in the curated set")
        elif not any(state in rule.states for rule in gene_rules):
            insufficient.append(
                f"{gene} {state}: intervention direction is outside the curated evidence"
            )

    applied_confidences: list[str] = []
    for rule in rules:
        applied = _rule_applies(rule, states, carbon_sources)
        condition = "any represented medium"
        if rule.carbon_sources:
            condition = "carbon source in: " + ", ".join(rule.carbon_sources)
        if applied:
            for effect, contribution in rule.effects.items():
                latent[effect] = latent.get(effect, 0.0) + contribution
            drivers.append(rule.explanation)
            applied_confidences.append(rule.confidence)
        # Include a rule in the trace when one of its genes was supplied, even
        # if its media condition or intervention direction did not match.
        if any(gene in states for gene in rule.genes):
            trace.append(
                GeneInteraction(
                    rule_id=rule.id,
                    gene=" + ".join(rule.genes),
                    state=" + ".join(states.get(gene, "not supplied") for gene in rule.genes),
                    condition=condition,
                    effects=dict(rule.effects),
                    explanation=rule.explanation,
                    evidence_url=rule.evidence_url,
                    confidence=rule.confidence,
                    applied=applied,
                )
            )

    # Process heuristics are intentionally weak and never presented as gene
    # evidence.  Low oxygen/off-window pH raise stress rather than dictating a
    # pellet class.  High shear can contribute to fragmentation.
    oxygen = context["oxygen"]
    if oxygen is not None and oxygen < 5.0:
        latent["stress"] += 1.0
        message = (
            "Low oxygen availability raised a stress flag; no direct pellet class was asserted."
        )
        drivers.append(message)
        warnings.append(message)
        trace.append(
            GeneInteraction(
                rule_id="process_low_oxygen",
                gene="(process)",
                state="low oxygen",
                condition="oxygen availability < 5",
                effects={"stress": 1.0},
                explanation=message,
                evidence_url=None,
                confidence="low",
                applied=True,
            )
        )

    ph = context["ph"]
    if ph is not None and abs(ph - profile.ph) > 1.5:
        latent["stress"] += 0.8
        message = "pH was outside the profile's illustrative window and raised a stress flag."
        drivers.append(message)
        warnings.append(message)
        trace.append(
            GeneInteraction(
                rule_id="process_ph_stress",
                gene="(process)",
                state=f"pH {ph:g}",
                condition=f"more than 1.5 pH units from {profile.ph:g}",
                effects={"stress": 0.8},
                explanation=message,
                evidence_url=None,
                confidence="low",
                applied=True,
            )
        )

    agitation = context["agitation"]
    shear = context["shear"]
    if (agitation is not None and agitation >= 500.0) or (shear is not None and shear >= 0.75):
        latent["fragmentation"] += 0.8
        latent["pellet_size"] -= 0.4
        message = "High shear/agitation weakly supports fragmentation and smaller structures."
        drivers.append(message)
        trace.append(
            GeneInteraction(
                rule_id="process_high_shear",
                gene="(process)",
                state="high shear",
                condition="agitation ≥ 500 rpm or relative shear ≥ 0.75",
                effects={"fragmentation": 0.8, "pellet_size": -0.4},
                explanation=message,
                evidence_url=None,
                confidence="low",
                applied=True,
            )
        )

    carbon = context["carbon"]
    nitrogen = context["nitrogen"]
    if carbon is not None and nitrogen is not None and nitrogen > 0 and carbon / nitrogen >= 10:
        latent["aggregation"] += 0.25
        message = (
            "A high relative C:N availability weakly supports aggregation in the process heuristic."
        )
        drivers.append(message)
        trace.append(
            GeneInteraction(
                rule_id="process_high_cn",
                gene="(process)",
                state="high C:N",
                condition="relative C:N availability ≥ 10",
                effects={"aggregation": 0.25},
                explanation=message,
                evidence_url=None,
                confidence="low",
                applied=True,
            )
        )

    scores = _support_scores(profile, latent)
    morphology = max(scores, key=lambda item: (scores[item], -list(scores).index(item)))
    confidence = "moderate" if "moderate" in applied_confidences else "low"
    if not drivers:
        drivers.append(
            "Species baseline tendency only; no supplied condition matched a curated rule."
        )
    if profile.id == "fusarium_venenatum":
        warnings.append(
            "No species-specific F. venenatum regulatory morphology rules are asserted; the class is a low-confidence process baseline."
        )
    if insufficient:
        warnings.append(
            "One or more interventions had insufficient species-specific evidence and were not scored."
        )

    # Retain supported_genes as an explicit local invariant: every non-native
    # gene not in this set must have been reported above.
    assert all(
        gene in supported_genes or any(item.startswith(gene + " ") for item in insufficient)
        for gene, state in states.items()
        if state != "native"
    )

    return MorphologyPrediction(
        fungus_id=profile.id,
        predicted_morphology=morphology,
        confidence=confidence,
        support_scores=scores,
        latent_scores={key: round(value, 4) for key, value in latent.items()},
        drivers=tuple(drivers),
        interaction_trace=tuple(trace),
        insufficient_evidence=tuple(insufficient),
        warnings=tuple(warnings),
    )


def supported_genes(fungus: str | FungusProfile) -> tuple[str, ...]:
    """List genes with at least one curated rule for the selected species."""

    profile = get_fungus(fungus)
    return tuple(dict.fromkeys(gene for rule in _RULES[profile.id] for gene in rule.genes))
