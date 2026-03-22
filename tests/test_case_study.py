"""Reproducibility checks for the published iJB1325 case study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myco_optima.case_study import (
    DEFAULT_MODEL_PATH,
    EXPECTED_CUSTOM_ATTRIBUTES,
    SOURCE_SHA256,
    SOURCE_SIZE_BYTES,
    CaseStudyError,
    read_pinned_source,
    run_published_case_study,
    validate_source_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / DEFAULT_MODEL_PATH


@pytest.fixture(scope="module")
def report():
    return run_published_case_study(MODEL_PATH)


def test_bundled_source_is_the_exact_publisher_file() -> None:
    payload = read_pinned_source(MODEL_PATH)

    assert len(payload) == SOURCE_SIZE_BYTES
    assert SOURCE_SHA256 == "c8f55761d925aa2b532e0b3279d740de29c5cd444ebc1aff0b38d26e007a5ea3"


def test_source_checksum_rejects_a_changed_model() -> None:
    payload = bytearray(read_pinned_source(MODEL_PATH))
    payload[-20] ^= 1

    with pytest.raises(CaseStudyError, match="checksum"):
        validate_source_bytes(payload)


def test_real_model_dimensions_and_published_result_are_reproduced(report) -> None:
    assert report.reactions == 2_320
    assert report.metabolites == 1_818
    assert report.genes == 1_325
    assert report.compartments == 7
    assert report.embedded_tests == 471
    assert report.passed == 373
    assert report.failed == 98
    assert report.pass_rate_percent == 79.2
    assert report.paper_result_reproduced
    assert report.compatibility_attributes_removed == EXPECTED_CUSTOM_ATTRIBUTES
    assert (
        report.growth_media_cases + report.deletion_phenotype_cases + report.system_check_cases
        == report.embedded_tests
    )


def test_default_glucose_fba_and_fva_are_real_and_bounded(report) -> None:
    scenario = report.default_glucose_scenario

    assert scenario.status == "optimal"
    assert scenario.biomass_objective == pytest.approx(0.9398547241493161)
    assert scenario.glucose_flux == pytest.approx(-10.0)
    assert scenario.oxygen_flux == pytest.approx(-7.778196990496574)
    assert scenario.biomass_fva.minimum == pytest.approx(0.8928619879418503)
    assert scenario.biomass_fva.maximum == pytest.approx(0.9398547241493161)
    assert scenario.glucose_fva.minimum <= scenario.glucose_flux
    assert scenario.glucose_fva.maximum > scenario.glucose_flux
    assert scenario.oxygen_fva.minimum <= scenario.oxygen_flux <= scenario.oxygen_fva.maximum


def test_report_is_strict_json_and_states_its_limit(report) -> None:
    payload = report.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert json.loads(encoded)["paper_result_reproduced"] is True
    assert "not an independent validation" in payload["interpretation"]
