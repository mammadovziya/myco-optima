"""Tests for the in-memory custom COBRA SBML boundary."""

from __future__ import annotations

import hashlib
import re
from io import StringIO

import pytest
from cobra import Metabolite, Model, Reaction
from cobra.io import write_sbml_model
from cobra.util.solver import linear_reaction_coefficients

import myco_optima.model_io as model_io
from myco_optima.model_io import (
    MAX_FVA_REACTIONS,
    CustomModelAnalysisError,
    InvalidSBMLError,
    ModelStructureError,
    ObjectiveSelectionError,
    UnsupportedModelFileError,
    UploadTooLargeError,
    analyse_custom_model,
    inspect_sbml_upload,
    load_sbml_upload,
    select_objective,
)


def _model(*, objective: bool = True) -> Model:
    model = Model("uploaded_demo", name="Uploaded demo model")
    precursor = Metabolite("M_precursor_c", compartment="c")
    biomass = Metabolite("M_biomass_c", compartment="c")

    source = Reaction("R_SOURCE", name="Precursor source", lower_bound=0, upper_bound=10)
    source.add_metabolites({precursor: 1})
    convert = Reaction("R_CONVERT", name="Biomass precursor conversion")
    convert.add_metabolites({precursor: -1, biomass: 1})
    convert.gene_reaction_rule = "G_geneA"
    sink = Reaction("R_BIOMASS", name="Biomass objective")
    sink.add_metabolites({biomass: -1})
    model.add_reactions([source, convert, sink])
    if objective:
        model.objective = sink
    return model


def _sbml_bytes(model: Model) -> bytes:
    handle = StringIO()
    write_sbml_model(model, handle, f_replace={})
    return handle.getvalue().encode("utf-8")


def _objective_ids(model: Model) -> tuple[str, ...]:
    return tuple(reaction.id for reaction in linear_reaction_coefficients(model))


def test_valid_upload_is_inspected_in_memory_and_preserves_ids() -> None:
    payload = _sbml_bytes(_model())
    inspection = load_sbml_upload(payload, "network.SBML")

    assert inspection.filename == "network.SBML"
    assert inspection.size_bytes == len(payload)
    assert inspection.sha256 == hashlib.sha256(payload).hexdigest()
    assert inspection.model_id == "uploaded_demo"
    assert inspection.model_name == "Uploaded demo model"
    assert inspection.reactions == 3
    assert inspection.metabolites == 2
    assert inspection.genes == 1
    assert inspection.current_objective_id == "R_BIOMASS"
    assert "R_BIOMASS" in inspection.candidate_objective_reaction_ids
    assert "R_SOURCE" in inspection.exchange_reaction_ids
    assert inspection.objective_candidates[0].reaction_id == "R_BIOMASS"
    assert "Biomass objective" in inspection.objective_candidates[0].label
    assert inspection.model.reactions.has_id("R_CONVERT")
    assert inspection.model.metabolites.has_id("M_precursor_c")
    assert any("does not enable" in warning for warning in inspection.warnings)
    assert inspection.metadata()["sha256"] == inspection.sha256


def test_inspect_alias_returns_equivalent_metadata() -> None:
    payload = _sbml_bytes(_model())
    first = load_sbml_upload(payload, "model.xml")
    second = inspect_sbml_upload(memoryview(payload), "model.xml")
    assert first.metadata() == second.metadata()


@pytest.mark.parametrize("filename", ["model.json", "model.xml.gz", "../model.xml", "", "a/b.sbml"])
def test_unsafe_or_unsupported_filenames_are_rejected(filename: str) -> None:
    with pytest.raises(UnsupportedModelFileError):
        load_sbml_upload(b"<sbml/>", filename)


def test_empty_non_bytes_oversized_and_non_utf8_uploads_are_rejected() -> None:
    with pytest.raises(InvalidSBMLError, match="empty"):
        load_sbml_upload(b"  ", "model.xml")
    with pytest.raises(UnsupportedModelFileError, match="bytes"):
        load_sbml_upload("<sbml/>", "model.xml")  # type: ignore[arg-type]
    with pytest.raises(UploadTooLargeError, match="limit"):
        load_sbml_upload(b"<sbml/>" * 10, "model.xml", max_bytes=8)
    with pytest.raises(UnsupportedModelFileError, match="UTF-8"):
        load_sbml_upload(b"\xff\xfe<sbml/>", "model.xml")


@pytest.mark.parametrize(
    "payload",
    [
        b'<?xml version="1.0"?><!DOCTYPE sbml [<!ENTITY x "x">]><sbml/>',
        b'<?xml version="1.0"?><!ENTITY x "x"><sbml/>',
    ],
)
def test_dtd_and_entity_declarations_are_rejected(payload: bytes) -> None:
    with pytest.raises(UnsupportedModelFileError, match="DTD and entity"):
        load_sbml_upload(payload, "model.xml")


def test_malformed_xml_wrong_root_and_non_sbml_xml_are_rejected() -> None:
    with pytest.raises(InvalidSBMLError, match="well-formed"):
        load_sbml_upload(b"<sbml>", "model.xml")
    with pytest.raises(InvalidSBMLError, match="root"):
        load_sbml_upload(b"<html></html>", "model.xml")
    with pytest.raises(InvalidSBMLError, match="COBRA-compatible"):
        load_sbml_upload(
            b'<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"/>', "model.xml"
        )


def test_models_without_reactions_are_rejected() -> None:
    model = Model("no_reactions")
    model.add_metabolites([Metabolite("M_a_c", compartment="c")])
    with pytest.raises(ModelStructureError, match="no reactions"):
        load_sbml_upload(_sbml_bytes(model), "empty.sbml")


def test_models_without_metabolites_are_rejected() -> None:
    model = Model("no_metabolites")
    reaction = Reaction("R_empty", lower_bound=0, upper_bound=1)
    model.add_reactions([reaction])
    model.objective = reaction
    with pytest.raises(ModelStructureError, match="no metabolites"):
        load_sbml_upload(_sbml_bytes(model), "empty.sbml")


def test_models_without_objective_are_rejected() -> None:
    with pytest.raises(ModelStructureError, match="objective"):
        load_sbml_upload(_sbml_bytes(_model(objective=False)), "no_objective.xml")


def test_missing_reaction_bound_is_rejected_instead_of_defaulting_to_1000() -> None:
    text = _sbml_bytes(_model()).decode("utf-8")
    altered, replacements = re.subn(
        r'(<reaction id="R_SOURCE"[^>]*?)\s+fbc:upperFluxBound="[^"]+"',
        r"\1",
        text,
        count=1,
    )
    assert replacements == 1

    with pytest.raises(ModelStructureError, match="R_SOURCE: missing upperFluxBound"):
        load_sbml_upload(altered.encode(), "missing_bound.xml")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'id="R_SOURCE_upper_bound" value="10"',
            'id="renamed_bound" value="10"',
            "bound parameter 'R_SOURCE_upper_bound' is missing",
        ),
        (
            'id="R_SOURCE_upper_bound" value="10"',
            'id="R_SOURCE_upper_bound" value="INF"',
            "missing or non-finite",
        ),
    ],
)
def test_bound_references_must_resolve_to_finite_parameters(
    old: str,
    new: str,
    message: str,
) -> None:
    text = _sbml_bytes(_model()).decode("utf-8")
    assert old in text
    altered = text.replace(old, new, 1)
    with pytest.raises(ModelStructureError, match=re.escape(message)):
        load_sbml_upload(altered.encode(), "unsafe_bound.sbml")


@pytest.mark.parametrize("value", ["NaN", "INF", "-INF", "1e309"])
def test_nonfinite_species_reference_stoichiometry_is_rejected(value: str) -> None:
    text = _sbml_bytes(_model()).decode("utf-8")
    assert 'stoichiometry="1"' in text
    altered = text.replace('stoichiometry="1"', f'stoichiometry="{value}"', 1)
    with pytest.raises(ModelStructureError, match="invalid stoichiometry"):
        load_sbml_upload(altered.encode(), "unsafe_stoichiometry.xml")


def test_stoichiometry_math_and_nonfinite_objective_coefficients_are_rejected() -> None:
    text = _sbml_bytes(_model()).decode("utf-8")
    altered, replacements = re.subn(
        r"(<speciesReference[^>]*)\s*/>",
        r"\1><stoichiometryMath/></speciesReference>",
        text,
        count=1,
    )
    assert replacements == 1
    with pytest.raises(ModelStructureError, match="stoichiometryMath"):
        load_sbml_upload(altered.encode(), "stoichiometry_math.xml")

    assert 'coefficient="1"' in text
    nonfinite_objective = text.replace('coefficient="1"', 'coefficient="INF"', 1)
    with pytest.raises(ModelStructureError, match="objective coefficient"):
        load_sbml_upload(nonfinite_objective.encode(), "unsafe_objective.xml")


def test_boundary_species_that_trigger_implicit_cobra_exchanges_are_rejected() -> None:
    text = _sbml_bytes(_model()).decode("utf-8")
    assert 'boundaryCondition="false"' in text
    altered = text.replace('boundaryCondition="false"', 'boundaryCondition="true"', 1)
    with pytest.raises(ModelStructureError, match="implicit exchange reactions"):
        load_sbml_upload(altered.encode(), "implicit_exchange.xml")


def test_objective_selection_uses_a_copy_and_can_preserve_or_change_direction() -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    original_bounds = {reaction.id: reaction.bounds for reaction in inspection.model.reactions}

    selected = select_objective(inspection, "R_CONVERT", direction="min")

    assert selected is not inspection.model
    assert _objective_ids(selected) == ("R_CONVERT",)
    assert selected.objective_direction == "min"
    assert _objective_ids(inspection.model) == ("R_BIOMASS",)
    assert {
        reaction.id: reaction.bounds for reaction in inspection.model.reactions
    } == original_bounds

    preserved = select_objective(inspection, "R_CONVERT")
    assert preserved.objective_direction == inspection.objective_direction


def test_objective_selection_rejects_unknown_reaction_and_direction() -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    with pytest.raises(ObjectiveSelectionError, match="not present"):
        select_objective(inspection, "R_MISSING")
    with pytest.raises(ObjectiveSelectionError, match="direction"):
        select_objective(inspection, "R_CONVERT", direction="sideways")


def test_unvalidated_raw_model_cannot_bypass_upload_checks() -> None:
    with pytest.raises(TypeError, match="validated ModelInspection"):
        select_objective(_model(), "R_BIOMASS")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validated ModelInspection"):
        analyse_custom_model(_model(), "R_BIOMASS")  # type: ignore[arg-type]


def test_custom_fba_and_explicit_fva_are_deterministic_and_copy_safe() -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    before_objective = _objective_ids(inspection.model)
    before_bounds = {reaction.id: reaction.bounds for reaction in inspection.model.reactions}

    first = analyse_custom_model(
        inspection,
        "R_CONVERT",
        fva_reaction_ids=["R_SOURCE", "R_CONVERT"],
        fraction_of_optimum=0.9,
    )
    second = analyse_custom_model(
        inspection,
        "R_CONVERT",
        fva_reaction_ids=["R_SOURCE", "R_CONVERT"],
        fraction_of_optimum=0.9,
    )

    assert first == second
    assert first.status == "optimal"
    assert first.objective_value == pytest.approx(10)
    assert first.fluxes["R_CONVERT"] == pytest.approx(10)
    assert first.fva_ranges["R_CONVERT"].minimum >= 9 - 1e-8
    assert first.fva_ranges["R_CONVERT"].maximum == pytest.approx(10)
    assert first.fva_fraction_of_optimum == 0.9
    assert _objective_ids(inspection.model) == before_objective
    assert {
        reaction.id: reaction.bounds for reaction in inspection.model.reactions
    } == before_bounds


def test_custom_analysis_skips_fva_unless_explicitly_requested() -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    source_timeout = inspection.model.solver.configuration.timeout
    result = analyse_custom_model(inspection, "R_BIOMASS")
    assert result.status == "optimal"
    assert result.solver_timeout_seconds == 30
    assert inspection.model.solver.configuration.timeout == source_timeout
    assert not any("no solver timeout" in warning for warning in result.warnings)
    assert result.fva_ranges == {}
    assert result.fva_fraction_of_optimum is None


def test_solver_timeout_validation_and_unsupported_warning(monkeypatch) -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    with pytest.raises(CustomModelAnalysisError, match="solver_timeout_seconds"):
        analyse_custom_model(inspection, "R_BIOMASS", solver_timeout_seconds=0)
    with pytest.raises(CustomModelAnalysisError, match="solver_timeout_seconds"):
        analyse_custom_model(inspection, "R_BIOMASS", solver_timeout_seconds=121)

    monkeypatch.setattr(
        model_io,
        "_configure_solver_timeout",
        lambda _model, _seconds: "Test solver does not support a timeout.",
    )
    result = analyse_custom_model(
        inspection,
        "R_BIOMASS",
        solver_timeout_seconds=12,
        fva_reaction_ids=["R_BIOMASS"],
    )
    assert result.solver_timeout_seconds == 12
    assert "Test solver does not support a timeout." in result.warnings
    assert result.fva_ranges == {}
    assert any("FVA was skipped" in warning for warning in result.warnings)


def test_custom_fva_request_is_validated_and_bounded() -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    with pytest.raises(CustomModelAnalysisError, match="cannot be empty"):
        analyse_custom_model(inspection, "R_BIOMASS", fva_reaction_ids=[])
    with pytest.raises(CustomModelAnalysisError, match="Unknown FVA"):
        analyse_custom_model(inspection, "R_BIOMASS", fva_reaction_ids=["R_UNKNOWN"])
    with pytest.raises(CustomModelAnalysisError, match="interval"):
        analyse_custom_model(
            inspection,
            "R_BIOMASS",
            fva_reaction_ids=["R_BIOMASS"],
            fraction_of_optimum=0,
        )
    with pytest.raises(CustomModelAnalysisError, match="limited"):
        analyse_custom_model(
            inspection,
            "R_BIOMASS",
            fva_reaction_ids=[f"R_{index}" for index in range(MAX_FVA_REACTIONS + 1)],
        )


@pytest.mark.parametrize(
    ("direction", "lower", "upper"),
    [("min", 5, 10), ("max", -10, -5)],
)
def test_fractional_fva_rejects_unsafe_objective_direction_or_sign(
    direction: str,
    lower: float,
    upper: float,
) -> None:
    model = Model("signed_objective")
    model.add_metabolites([Metabolite("M_dummy_c", compartment="c")])
    objective = Reaction("R_OBJECTIVE", lower_bound=lower, upper_bound=upper)
    model.add_reactions([objective])
    model.objective = objective
    inspection = load_sbml_upload(_sbml_bytes(model), "signed-objective.xml")

    with pytest.raises(CustomModelAnalysisError, match="non-negative maximization"):
        analyse_custom_model(
            inspection,
            "R_OBJECTIVE",
            direction=direction,
            fva_reaction_ids=["R_OBJECTIVE"],
            fraction_of_optimum=0.95,
        )


def test_fva_enforces_aggregate_wall_clock_budget(monkeypatch) -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    clock = iter((0.0, model_io.MAX_FVA_WALL_SECONDS + 1))
    monkeypatch.setattr(model_io, "monotonic", lambda: next(clock))
    with pytest.raises(CustomModelAnalysisError, match="aggregate time budget"):
        analyse_custom_model(
            inspection,
            "R_BIOMASS",
            fva_reaction_ids=["R_BIOMASS"],
        )


def test_infeasible_custom_model_returns_status_and_no_fabricated_fluxes() -> None:
    inspection = load_sbml_upload(_sbml_bytes(_model()), "model.xml")
    inspection.model.reactions.get_by_id("R_SOURCE").upper_bound = 0
    inspection.model.reactions.get_by_id("R_BIOMASS").lower_bound = 1

    result = analyse_custom_model(inspection, "R_BIOMASS")

    assert result.status == "infeasible"
    assert result.objective_value is None
    assert result.fluxes == {}
    assert result.fva_ranges == {}
    assert any("unavailable" in warning for warning in result.warnings)
