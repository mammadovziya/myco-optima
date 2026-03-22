"""Reproduce the published iJB1325 Aspergillus niger case study.

The bundled source is the ATCC 1015 SBML supplement from Brandl et al. (2018).
Its embedded cases helped curate the model, so this runner checks reproducibility,
not independent biological accuracy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cobra import Model
from cobra.flux_analysis import flux_variability_analysis
from cobra.io import read_sbml_model

SOURCE_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1186%2Fs40694-018-0060-7/MediaObjects/"
    "40694_2018_60_MOESM2_ESM.xml"
)
PAPER_URL = "https://doi.org/10.1186/s40694-018-0060-7"
SOURCE_SHA256 = "c8f55761d925aa2b532e0b3279d740de29c5cd444ebc1aff0b38d26e007a5ea3"
SOURCE_SIZE_BYTES = 6_676_295
DEFAULT_MODEL_PATH = Path("case_studies/aspergillus_niger_iJB1325/model/iJB1325_ATCC1015.xml.gz")

EXPECTED_REACTIONS = 2_320
EXPECTED_METABOLITES = 1_818
EXPECTED_GENES = 1_325
EXPECTED_COMPARTMENTS = 7
EXPECTED_TESTS = 471
EXPECTED_PASSED = 373
EXPECTED_FAILED = 98
EXPECTED_CUSTOM_ATTRIBUTES = 3_738

_GEM_NS = "http://bitbucket.org/JuBra/gem-editor"
_GEM = {"gem": _GEM_NS}
_CUSTOM_ATTRIBUTE = re.compile(rb'\s+gem:(?:genome|subsystem|id|comment)="[^"]*"')
_OUTCOME_MARGIN = 0.01


class CaseStudyError(ValueError):
    """Raised when the pinned model or embedded benchmark is not reproducible."""


@dataclass(frozen=True)
class FluxRange:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class DefaultGlucoseScenario:
    """One FBA/FVA check using the source model's default glucose bounds."""

    status: str
    biomass_objective: float
    glucose_flux: float
    oxygen_flux: float
    fva_fraction_of_optimum: float
    biomass_fva: FluxRange
    glucose_fva: FluxRange
    oxygen_fva: FluxRange


@dataclass(frozen=True)
class PublishedCaseStudyReport:
    """Compact evidence from the pinned model and its embedded test suite."""

    model: str
    strain: str
    source_sha256: str
    source_size_bytes: int
    reactions: int
    metabolites: int
    genes: int
    compartments: int
    embedded_tests: int
    passed: int
    failed: int
    pass_rate_percent: float
    paper_reported_passed: int
    paper_reported_failed: int
    paper_result_reproduced: bool
    growth_media_cases: int
    deletion_phenotype_cases: int
    system_check_cases: int
    compatibility_attributes_removed: int
    outcome_margin: float
    default_glucose_scenario: DefaultGlucoseScenario

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-safe evidence with its interpretation."""

        return {
            **asdict(self),
            "source_url": SOURCE_URL,
            "paper_url": PAPER_URL,
            "interpretation": (
                "Reproduction of the published phenotype consistency benchmark. "
                "The embedded cases informed model curation and are not an independent "
                "validation or a measure of biological accuracy."
            ),
        }


def read_pinned_source(path: str | Path = DEFAULT_MODEL_PATH) -> bytes:
    """Read the exact source XML from a plain or gzip-compressed file."""

    path = Path(path)
    if not path.is_file():
        raise CaseStudyError(f"Case-study model was not found: {path}")

    if path.suffix.lower() == ".gz":
        try:
            with gzip.open(path, "rb") as handle:
                payload = handle.read(SOURCE_SIZE_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise CaseStudyError("Case-study gzip file is invalid.") from exc
    else:
        if path.stat().st_size != SOURCE_SIZE_BYTES:
            raise CaseStudyError("Case-study source size does not match the pinned file.")
        payload = path.read_bytes()

    validate_source_bytes(payload)
    return payload


def validate_source_bytes(payload: bytes | bytearray | memoryview) -> None:
    """Verify the publisher's exact byte count and checksum."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    view = memoryview(payload)
    if len(view) != SOURCE_SIZE_BYTES:
        raise CaseStudyError("Case-study source size does not match the pinned file.")
    digest = hashlib.sha256(view).hexdigest()
    if digest != SOURCE_SHA256:
        raise CaseStudyError("Case-study source checksum does not match the pinned file.")


def _cobra_compatible_xml(payload: bytes) -> tuple[str, int]:
    """Remove only legacy GEMEditor attributes rejected by current libSBML.

    The original, checksummed source remains unchanged. The exact source hash makes
    this narrow compatibility conversion safe and reproducible.
    """

    cleaned, removed = _CUSTOM_ATTRIBUTE.subn(b"", payload)
    if removed != EXPECTED_CUSTOM_ATTRIBUTES:
        raise CaseStudyError(
            "The source contains an unexpected number of legacy GEMEditor attributes."
        )
    model_marker = b'<sbml:model fbc:strict="true">'
    if cleaned.count(model_marker) != 1:
        raise CaseStudyError("The source model element does not match the pinned file.")
    cleaned = cleaned.replace(
        model_marker,
        b'<sbml:model id="iJB1325_ATCC1015" fbc:strict="true">',
        1,
    )
    try:
        return cleaned.decode("utf-8"), removed
    except UnicodeDecodeError as exc:
        raise CaseStudyError("Case-study source is not valid UTF-8 XML.") from exc


def load_published_model(payload: bytes) -> tuple[Model, int]:
    """Load the pinned model in COBRApy while preserving source identifiers."""

    validate_source_bytes(payload)
    cleaned, removed = _cobra_compatible_xml(payload)
    try:
        model = read_sbml_model(io.StringIO(cleaned), f_replace={})
    except Exception as exc:
        raise CaseStudyError("COBRApy could not load the pinned case-study model.") from exc
    model.id = "iJB1325_ATCC1015"
    return model, removed


def _default_glucose_analysis(model: Model) -> DefaultGlucoseScenario:
    solution = model.optimize()
    if solution.status != "optimal" or not math.isfinite(float(solution.objective_value)):
        raise CaseStudyError("The source model's default glucose scenario was not optimal.")

    reaction_ids = ("R_DRAIN_Biomass", "R_BOUNDARY_GLCe", "R_BOUNDARY_O2e")
    missing = set(reaction_ids) - {reaction.id for reaction in model.reactions}
    if missing:
        raise CaseStudyError(f"Default glucose scenario is missing reactions: {sorted(missing)}")
    fva = flux_variability_analysis(
        model,
        reaction_list=list(reaction_ids),
        fraction_of_optimum=0.95,
        processes=1,
    )

    def flux_range(reaction_id: str) -> FluxRange:
        return FluxRange(
            minimum=float(fva.loc[reaction_id, "minimum"]),
            maximum=float(fva.loc[reaction_id, "maximum"]),
        )

    return DefaultGlucoseScenario(
        status=solution.status,
        biomass_objective=float(solution.objective_value),
        glucose_flux=float(solution.fluxes["R_BOUNDARY_GLCe"]),
        oxygen_flux=float(solution.fluxes["R_BOUNDARY_O2e"]),
        fva_fraction_of_optimum=0.95,
        biomass_fva=flux_range("R_DRAIN_Biomass"),
        glucose_fva=flux_range("R_BOUNDARY_GLCe"),
        oxygen_fva=flux_range("R_BOUNDARY_O2e"),
    )


def _prepare_for_embedded_tests(model: Model) -> None:
    """Match GEMEditor 0.3.2 test preparation exactly."""

    for reaction in model.reactions:
        coefficients = tuple(reaction.metabolites.values())
        if coefficients and all(value < 0 for value in coefficients):
            reaction.lower_bound = 0
        elif coefficients and all(value > 0 for value in coefficients):
            reaction.upper_bound = 0
    model.objective = model.problem.Objective(0, direction="max")


def _run_embedded_tests(model: Model, source: bytes) -> tuple[int, int]:
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise CaseStudyError("Case-study source XML is malformed.") from exc
    cases = root.findall(".//gem:testCase", _GEM)
    if len(cases) != EXPECTED_TESTS:
        raise CaseStudyError("The source does not contain the expected 471 test cases.")

    _prepare_for_embedded_tests(model)
    passed = 0
    for case in cases:
        with model:
            objective: dict[Any, float] = {}
            for setting in case.findall("./gem:listOfSettings/gem:reactionSetting", _GEM):
                reaction_id = setting.attrib["reactionId"]
                try:
                    reaction = model.reactions.get_by_id(reaction_id)
                except KeyError as exc:
                    raise CaseStudyError(
                        f"Embedded test references unknown reaction {reaction_id}."
                    ) from exc
                reaction.bounds = (
                    float(setting.attrib["lowerBound"]),
                    float(setting.attrib["upperBound"]),
                )
                coefficient = float(setting.attrib.get("objectiveCoefficient", 0))
                if coefficient:
                    objective[reaction] = coefficient
            model.objective = (
                objective if objective else model.problem.Objective(0, direction="max")
            )

            for setting in case.findall("./gem:listOfSettings/gem:geneSetting", _GEM):
                gene_id = setting.attrib["geneId"]
                try:
                    gene = model.genes.get_by_id(gene_id)
                except KeyError as exc:
                    raise CaseStudyError(
                        f"Embedded test references unknown gene {gene_id}."
                    ) from exc
                active_text = setting.attrib.get("active")
                if active_text not in {"True", "False"}:
                    raise CaseStudyError("Embedded test contains an invalid gene activity flag.")
                gene.functional = active_text == "True"
                if not gene.functional:
                    for reaction in gene.reactions:
                        if not reaction.functional:
                            reaction.bounds = (0, 0)

            solution = model.optimize()
            case_passed = solution.status == "optimal"
            if case_passed:
                for outcome in case.findall("./gem:listOfOutcomes/gem:outcome", _GEM):
                    reaction_id = outcome.attrib["reactionId"]
                    value = float(outcome.attrib["value"])
                    operator = outcome.attrib["operator"]
                    try:
                        flux = float(solution.fluxes[reaction_id])
                    except KeyError as exc:
                        raise CaseStudyError(
                            f"Embedded outcome references unknown reaction {reaction_id}."
                        ) from exc
                    if operator == "greater than":
                        matches = flux - _OUTCOME_MARGIN > value
                    elif operator == "less than":
                        matches = flux + _OUTCOME_MARGIN < value
                    else:
                        raise CaseStudyError(f"Unsupported embedded outcome operator: {operator}")
                    if not matches:
                        case_passed = False
                        break
            passed += int(case_passed)

    return passed, len(cases) - passed


def run_published_case_study(
    path: str | Path = DEFAULT_MODEL_PATH,
) -> PublishedCaseStudyReport:
    """Run FBA/FVA and reproduce the 471 embedded iJB1325 test cases."""

    source = read_pinned_source(path)
    model, removed = load_published_model(source)
    counts = (
        len(model.reactions),
        len(model.metabolites),
        len(model.genes),
        len(model.compartments),
    )
    expected_counts = (
        EXPECTED_REACTIONS,
        EXPECTED_METABOLITES,
        EXPECTED_GENES,
        EXPECTED_COMPARTMENTS,
    )
    if counts != expected_counts:
        raise CaseStudyError(f"Loaded model dimensions changed: {counts!r}")

    glucose = _default_glucose_analysis(model)
    passed, failed = _run_embedded_tests(model, source)
    return PublishedCaseStudyReport(
        model="iJB1325",
        strain="Aspergillus niger ATCC 1015",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=len(source),
        reactions=counts[0],
        metabolites=counts[1],
        genes=counts[2],
        compartments=counts[3],
        embedded_tests=passed + failed,
        passed=passed,
        failed=failed,
        pass_rate_percent=round(100 * passed / (passed + failed), 1),
        paper_reported_passed=EXPECTED_PASSED,
        paper_reported_failed=EXPECTED_FAILED,
        paper_result_reproduced=(passed, failed) == (EXPECTED_PASSED, EXPECTED_FAILED),
        growth_media_cases=392,
        deletion_phenotype_cases=73,
        system_check_cases=6,
        compatibility_attributes_removed=removed,
        outcome_margin=_OUTCOME_MARGIN,
        default_glucose_scenario=glucose,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the pinned ATCC 1015 XML or XML.GZ file.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = run_published_case_study(args.model)
    output = json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if report.paper_result_reproduced else 1


if __name__ == "__main__":
    raise SystemExit(_main())
