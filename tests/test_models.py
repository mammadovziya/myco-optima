"""Structural and biological-invariant tests for the reduced-order models."""

from __future__ import annotations

import math

import pytest

from myco_optima.catalog import MODEL_DISCLAIMER, get_fungus, list_fungi, normalise_medium
from myco_optima.models import BIOMASS_REACTION, apply_medium, build_model, effective_medium
from myco_optima.optimization import run_fba


def test_catalogue_contains_four_industrial_filamentous_fungi() -> None:
    profiles = list_fungi()
    assert [profile.id for profile in profiles] == [
        "aspergillus_niger",
        "aspergillus_oryzae",
        "trichoderma_reesei",
        "fusarium_venenatum",
    ]
    assert get_fungus("A. niger").id == "aspergillus_niger"
    assert get_fungus("Trichoderma reesei").id == "trichoderma_reesei"
    assert all(MODEL_DISCLAIMER in profile.notes for profile in profiles)


def test_medium_aliases_are_validated() -> None:
    assert normalise_medium({"Glucose": 10, "Ammonium sulfate": 2, "O₂ uptake": 8}) == {
        "glucose": 10.0,
        "ammonium": 2.0,
        "oxygen": 8.0,
    }
    with pytest.raises(ValueError, match="non-negative"):
        normalise_medium({"glucose": -1})
    with pytest.raises(KeyError, match="Unknown medium"):
        normalise_medium({"mystery powder": 1})


@pytest.mark.parametrize("profile", list_fungi(), ids=lambda profile: profile.id)
def test_each_baseline_model_is_finite_feasible_and_disclaimed(profile) -> None:
    model = build_model(profile)
    result = run_fba(model)

    assert model.objective.expression is not None
    assert BIOMASS_REACTION in model.reactions
    assert result.status == "optimal"
    assert result.growth_rate > 0
    assert math.isfinite(result.growth_rate)
    assert model.notes["disclaimer"] == MODEL_DISCLAIMER
    assert all(math.isfinite(bound) for reaction in model.reactions for bound in reaction.bounds)


def test_selected_carbon_and_nitrogen_sources_are_exclusive() -> None:
    model = build_model("aspergillus_niger", {"xylose": 8, "nitrate": 5})
    medium = effective_medium(model)

    assert medium["xylose"] == 8
    assert medium["glucose"] == 0
    assert medium["nitrate"] == 5
    assert medium["ammonium"] == 0
    assert run_fba(model).growth_rate > 0


@pytest.mark.parametrize(
    "essential",
    ["glucose", "ammonium", "oxygen", "phosphate", "sulfate", "magnesium", "iron", "zinc"],
)
def test_closing_each_essential_removes_free_biomass(essential: str) -> None:
    model = build_model("aspergillus_niger")
    medium = effective_medium(model)
    medium[essential] = 0.0

    assert run_fba(model, medium).growth_rate == pytest.approx(0.0, abs=1e-9)


def test_medium_application_is_copy_on_write_and_caps_capacity() -> None:
    model = build_model("aspergillus_niger")
    original = effective_medium(model)
    changed = apply_medium(model, {"glucose": 10_000})

    assert effective_medium(model) == original
    assert (
        effective_medium(changed)["glucose"]
        == get_fungus("aspergillus_niger").uptake_capacities["glucose"]
    )


def test_fresh_models_do_not_share_reaction_bounds() -> None:
    first = build_model("aspergillus_niger")
    second = build_model("aspergillus_niger")
    first.reactions.EX_glc__D_e.lower_bound = 0

    assert second.reactions.EX_glc__D_e.lower_bound < 0
