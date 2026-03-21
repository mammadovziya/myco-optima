"""Safe, in-memory loading and analysis of user-supplied COBRA SBML models.

Uploaded models remain scientifically separate from the bundled myco-optima
teaching surrogates.  Loading a custom SBML model does not confer medium,
sensitivity, organism, or gene–media support.  The functions here preserve SBML
identifiers and constraints, never write uploads to disk, and analyse copies.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from time import monotonic
from typing import Any
from xml.etree import ElementTree

from cobra import Model
from cobra.flux_analysis import flux_variability_analysis
from cobra.io import read_sbml_model
from cobra.util.solver import linear_reaction_coefficients

from .types import FluxRange

DEFAULT_MAX_UPLOAD_BYTES = 5_000_000
DEFAULT_SOLVER_TIMEOUT_SECONDS = 30.0
MAX_SOLVER_TIMEOUT_SECONDS = 120.0
MAX_FVA_WALL_SECONDS = 60.0
MAX_FVA_REACTIONS = 50
MAX_MODEL_REACTIONS = 20_000
ALLOWED_EXTENSIONS = (".xml", ".sbml")

_CUSTOM_MODEL_WARNING = (
    "Custom SBML constraints are analysed as uploaded; this does not enable bundled "
    "fungus, medium, sensitivity, or gene–media claims."
)


class ModelUploadError(ValueError):
    """Base class for safe, user-facing upload failures."""


class UploadTooLargeError(ModelUploadError):
    """Raised when an upload exceeds the configured byte cap."""


class UnsupportedModelFileError(ModelUploadError):
    """Raised for unsupported filenames, encodings, or XML features."""


class InvalidSBMLError(ModelUploadError):
    """Raised when content is not parseable COBRA-compatible SBML."""


class ModelStructureError(ModelUploadError):
    """Raised when parsed SBML lacks required modelling structure."""


class ObjectiveSelectionError(ValueError):
    """Raised when a requested objective is not valid for a model."""


class CustomModelAnalysisError(RuntimeError):
    """Raised when a custom model cannot be analysed safely."""


@dataclass(frozen=True)
class ObjectiveTerm:
    """One reaction coefficient in the uploaded linear objective."""

    reaction_id: str
    coefficient: float


@dataclass(frozen=True)
class ObjectiveCandidate:
    """A selectable reaction id and human-readable label."""

    reaction_id: str
    label: str
    is_current: bool
    boundary: bool


@dataclass(frozen=True)
class ModelInspection:
    """A parsed model and immutable inspection metadata."""

    model: Model
    filename: str
    size_bytes: int
    sha256: str
    model_id: str
    model_name: str
    reactions: int
    metabolites: int
    genes: int
    exchanges: int
    exchange_reaction_ids: tuple[str, ...]
    objective_candidates: tuple[ObjectiveCandidate, ...]
    candidate_objective_reaction_ids: tuple[str, ...]
    current_objective: tuple[ObjectiveTerm, ...]
    objective_direction: str
    warnings: tuple[str, ...]

    @property
    def current_objective_id(self) -> str | None:
        """Return the objective id when the upload has one objective term."""

        if len(self.current_objective) == 1:
            return self.current_objective[0].reaction_id
        return None

    @property
    def reaction_count(self) -> int:
        return self.reactions

    @property
    def metabolite_count(self) -> int:
        return self.metabolites

    @property
    def gene_count(self) -> int:
        return self.genes

    @property
    def exchange_count(self) -> int:
        return self.exchanges

    def metadata(self) -> dict[str, Any]:
        """Return serialization-safe metadata without embedding the model."""

        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "reactions": self.reactions,
            "metabolites": self.metabolites,
            "genes": self.genes,
            "exchanges": self.exchanges,
            "exchange_reaction_ids": list(self.exchange_reaction_ids),
            "objective_candidates": [
                {
                    "reaction_id": candidate.reaction_id,
                    "label": candidate.label,
                    "is_current": candidate.is_current,
                    "boundary": candidate.boundary,
                }
                for candidate in self.objective_candidates
            ],
            "candidate_objective_reaction_ids": list(self.candidate_objective_reaction_ids),
            "current_objective": [
                {"reaction_id": term.reaction_id, "coefficient": term.coefficient}
                for term in self.current_objective
            ],
            "objective_direction": self.objective_direction,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CustomModelAnalysis:
    """FBA and optional bounded FVA result for a copied custom model."""

    model_id: str
    objective_reaction_id: str
    objective_direction: str
    solver_timeout_seconds: float
    status: str
    objective_value: float | None
    fluxes: dict[str, float]
    fva_fraction_of_optimum: float | None
    fva_ranges: dict[str, FluxRange]
    warnings: tuple[str, ...]


def _validate_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise UnsupportedModelFileError("filename must be a string ending in .xml or .sbml")
    cleaned = filename.strip()
    if not cleaned:
        raise UnsupportedModelFileError("filename cannot be empty")
    if len(cleaned) > 255 or "\x00" in cleaned or "/" in cleaned or "\\" in cleaned:
        raise UnsupportedModelFileError("filename must be a plain filename without path components")
    if not cleaned.casefold().endswith(ALLOWED_EXTENSIONS):
        raise UnsupportedModelFileError(
            "Only uncompressed .xml and .sbml files are accepted; archives are not supported."
        )
    return cleaned


def _validated_text(
    uploaded_bytes: bytes | bytearray | memoryview,
    max_bytes: int,
) -> tuple[str, bytes]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(uploaded_bytes, (bytes, bytearray, memoryview)):
        raise UnsupportedModelFileError("uploaded content must be bytes")
    payload = bytes(uploaded_bytes)
    if not payload or not payload.strip():
        raise InvalidSBMLError("The uploaded SBML file is empty.")
    if len(payload) > max_bytes:
        raise UploadTooLargeError(
            f"The uploaded file is {len(payload):,} bytes; the limit is {max_bytes:,} bytes."
        )
    if b"\x00" in payload:
        raise UnsupportedModelFileError("NUL bytes are not permitted in an SBML upload.")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedModelFileError("SBML uploads must be UTF-8 encoded XML.") from exc

    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, flags=re.IGNORECASE):
        raise UnsupportedModelFileError("DTD and entity declarations are not permitted.")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise InvalidSBMLError("The upload is not well-formed XML.") from exc
    local_name = root.tag.rsplit("}", 1)[-1].casefold() if isinstance(root.tag, str) else ""
    if local_name != "sbml":
        raise InvalidSBMLError("The XML document root must be <sbml>.")
    _validate_explicit_fbc_bounds(root)
    _reject_implicit_boundary_exchanges(root)
    return text, payload


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _attribute(element: ElementTree.Element, local_name: str) -> str | None:
    for attribute_name, value in element.attrib.items():
        if _local_name(attribute_name) == local_name:
            return value
    return None


def _direct_child(element: ElementTree.Element, local_name: str) -> ElementTree.Element | None:
    return next(
        (child for child in element if _local_name(child.tag) == local_name),
        None,
    )


def _validate_explicit_fbc_bounds(root: ElementTree.Element) -> None:
    """Reject reactions for which COBRApy would silently invent flux bounds."""

    model_element = _direct_child(root, "model")
    if model_element is None:
        return  # COBRApy will produce the more useful no-model parse error.

    parameters: dict[str, tuple[str | None, str | None]] = {}
    parameter_list = _direct_child(model_element, "listOfParameters")
    if parameter_list is not None:
        for parameter in parameter_list:
            if _local_name(parameter.tag) != "parameter":
                continue
            parameter_id = _attribute(parameter, "id")
            if parameter_id:
                parameters[parameter_id] = (
                    _attribute(parameter, "value"),
                    _attribute(parameter, "constant"),
                )

    reaction_list = _direct_child(model_element, "listOfReactions")
    if reaction_list is None:
        return

    issues: list[str] = []
    for index, reaction in enumerate(reaction_list, start=1):
        if _local_name(reaction.tag) != "reaction":
            continue
        reaction_id = _attribute(reaction, "id") or f"reaction #{index}"
        for participant_list_name in ("listOfReactants", "listOfProducts"):
            participant_list = _direct_child(reaction, participant_list_name)
            if participant_list is None:
                continue
            for reference in participant_list:
                if _local_name(reference.tag) != "speciesReference":
                    continue
                species_id = _attribute(reference, "species") or "unnamed species"
                if any(
                    _local_name(descendant.tag) == "stoichiometryMath"
                    for descendant in reference.iter()
                ):
                    issues.append(
                        f"{reaction_id}: stoichiometryMath for {species_id!r} is unsupported"
                    )
                    continue
                raw_stoichiometry = _attribute(reference, "stoichiometry")
                if raw_stoichiometry is None:
                    continue  # SBML defines the omitted stoichiometry default as 1.
                try:
                    stoichiometry = float(raw_stoichiometry)
                except ValueError:
                    stoichiometry = math.nan
                if not math.isfinite(stoichiometry) or stoichiometry < 0:
                    issues.append(
                        f"{reaction_id}: speciesReference for {species_id!r} has invalid stoichiometry"
                    )
        bound_values: dict[str, float] = {}
        for side, attribute_name in (
            ("lower", "lowerFluxBound"),
            ("upper", "upperFluxBound"),
        ):
            parameter_id = _attribute(reaction, attribute_name)
            if not parameter_id:
                issues.append(f"{reaction_id}: missing {attribute_name}")
                continue
            parameter = parameters.get(parameter_id)
            if parameter is None:
                issues.append(f"{reaction_id}: bound parameter {parameter_id!r} is missing")
                continue
            raw_value, constant = parameter
            if constant is None or constant.casefold() not in {"true", "1"}:
                issues.append(f"{reaction_id}: bound parameter {parameter_id!r} is not constant")
                continue
            try:
                value = float(raw_value) if raw_value is not None else math.nan
            except ValueError:
                value = math.nan
            if not math.isfinite(value):
                issues.append(
                    f"{reaction_id}: bound parameter {parameter_id!r} is missing or non-finite"
                )
                continue
            bound_values[side] = value
        if (
            "lower" in bound_values
            and "upper" in bound_values
            and bound_values["lower"] > bound_values["upper"]
        ):
            issues.append(f"{reaction_id}: lower bound exceeds upper bound")

    for flux_objective in model_element.iter():
        if _local_name(flux_objective.tag) != "fluxObjective":
            continue
        reaction_id = _attribute(flux_objective, "reaction") or "unknown reaction"
        raw_coefficient = _attribute(flux_objective, "coefficient")
        try:
            coefficient = float(raw_coefficient) if raw_coefficient is not None else math.nan
        except ValueError:
            coefficient = math.nan
        if not math.isfinite(coefficient):
            issues.append(f"objective coefficient for {reaction_id!r} is missing or non-finite")

    if issues:
        preview = "; ".join(issues[:10])
        remainder = len(issues) - 10
        if remainder > 0:
            preview += f"; and {remainder} more"
        raise ModelStructureError(
            "Uploaded reactions must use explicit finite FBC bounds, finite stoichiometries, "
            f"and finite objective coefficients. {preview}."
        )


def _reject_implicit_boundary_exchanges(root: ElementTree.Element) -> None:
    """Reject species for which COBRApy would invent a default exchange reaction."""

    model_element = _direct_child(root, "model")
    species_list = (
        _direct_child(model_element, "listOfSpecies") if model_element is not None else None
    )
    if species_list is None:
        return
    boundary_species = [
        _attribute(species, "id") or "unnamed species"
        for species in species_list
        if _local_name(species.tag) == "species"
        and (_attribute(species, "boundaryCondition") or "false").casefold() in {"true", "1"}
    ]
    if boundary_species:
        preview = ", ".join(boundary_species[:10])
        raise ModelStructureError(
            "Species with boundaryCondition=true are not accepted because COBRApy would add "
            f"implicit exchange reactions with default bounds. Encode explicit exchanges for: {preview}."
        )


def _objective_terms(model: Model) -> tuple[ObjectiveTerm, ...]:
    coefficients = linear_reaction_coefficients(model)
    terms: list[ObjectiveTerm] = []
    for reaction, coefficient in coefficients.items():
        value = float(coefficient)
        if not math.isfinite(value):
            raise ModelStructureError(
                f"The objective coefficient for reaction {reaction.id!r} is non-finite."
            )
        if abs(value) > 1e-12:
            terms.append(ObjectiveTerm(reaction_id=reaction.id, coefficient=value))
    return tuple(terms)


def _objective_candidates(
    model: Model,
    current: tuple[ObjectiveTerm, ...],
) -> tuple[ObjectiveCandidate, ...]:
    current_ids = [term.reaction_id for term in current]
    keyword = re.compile(r"(?:biomass|growth|objective|product|demand|sink)", re.IGNORECASE)
    likely = [
        reaction.id
        for reaction in model.reactions
        if keyword.search(reaction.id) or keyword.search(reaction.name or "")
    ]
    drains = [
        reaction.id
        for reaction in model.reactions
        if reaction.reactants and not reaction.products and not reaction.boundary
    ]
    # Any reaction is mathematically selectable.  Likely biological objectives
    # are ordered first, while the complete list prevents the heuristic from
    # hiding a valid upload-specific choice.
    ordered_ids = tuple(
        dict.fromkeys(
            [*current_ids, *likely, *drains, *(reaction.id for reaction in model.reactions)]
        )
    )
    current_set = set(current_ids)
    records: list[ObjectiveCandidate] = []
    for reaction_id in ordered_ids:
        reaction = model.reactions.get_by_id(reaction_id)
        name = (reaction.name or "").strip()
        records.append(
            ObjectiveCandidate(
                reaction_id=reaction_id,
                label=f"{reaction_id} — {name}" if name and name != reaction_id else reaction_id,
                is_current=reaction_id in current_set,
                boundary=bool(reaction.boundary),
            )
        )
    return tuple(records)


def _inspect_model(model: Model, filename: str, size_bytes: int, digest: str) -> ModelInspection:
    reaction_count = len(model.reactions)
    metabolite_count = len(model.metabolites)
    if reaction_count == 0:
        raise ModelStructureError("The SBML model contains no reactions.")
    if reaction_count > MAX_MODEL_REACTIONS:
        raise ModelStructureError(
            f"The SBML model contains {reaction_count:,} reactions; the safety limit is "
            f"{MAX_MODEL_REACTIONS:,}."
        )
    if metabolite_count == 0:
        raise ModelStructureError("The SBML model contains no metabolites.")

    current = _objective_terms(model)
    if not current:
        raise ModelStructureError("The SBML model does not define a non-zero linear objective.")

    warnings = [_CUSTOM_MODEL_WARNING]
    if not model.id:
        warnings.append("The SBML model has no model id.")
    if not model.name:
        warnings.append("The SBML model has no display name.")
    if len(model.genes) == 0:
        warnings.append("The SBML model contains no parsed gene associations.")
    exchange_reaction_ids = tuple(reaction.id for reaction in model.exchanges)
    if not exchange_reaction_ids:
        warnings.append("COBRApy detected no boundary exchange reactions.")
    if len(current) > 1:
        warnings.append("The uploaded objective contains multiple reaction terms.")

    objective_candidates = _objective_candidates(model, current)
    return ModelInspection(
        model=model,
        filename=filename,
        size_bytes=size_bytes,
        sha256=digest,
        model_id=str(model.id or ""),
        model_name=str(model.name or ""),
        reactions=reaction_count,
        metabolites=metabolite_count,
        genes=len(model.genes),
        exchanges=len(exchange_reaction_ids),
        exchange_reaction_ids=exchange_reaction_ids,
        objective_candidates=objective_candidates,
        candidate_objective_reaction_ids=tuple(
            candidate.reaction_id for candidate in objective_candidates
        ),
        current_objective=current,
        objective_direction=str(model.objective_direction),
        warnings=tuple(warnings),
    )


def load_sbml_upload(
    uploaded_bytes: bytes | bytearray | memoryview,
    filename: str,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> ModelInspection:
    """Validate and parse an uncompressed SBML upload entirely in memory.

    COBRApy's identifier-replacement map is disabled so reaction, metabolite,
    and gene identifiers remain as encoded in the upload.
    """

    safe_filename = _validate_filename(filename)
    text, payload = _validated_text(uploaded_bytes, max_bytes)
    try:
        model = read_sbml_model(StringIO(text), f_replace={})
    except Exception as exc:
        raise InvalidSBMLError(
            "The file is not a valid COBRA-compatible SBML model. Check its SBML/FBC structure."
        ) from exc
    return _inspect_model(model, safe_filename, len(payload), sha256(payload).hexdigest())


def inspect_sbml_upload(
    uploaded_bytes: bytes | bytearray | memoryview,
    filename: str,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> ModelInspection:
    """Alias for :func:`load_sbml_upload`."""

    return load_sbml_upload(uploaded_bytes, filename, max_bytes=max_bytes)


def _source_model(model_or_inspection: ModelInspection) -> Model:
    if isinstance(model_or_inspection, ModelInspection):
        return model_or_inspection.model
    raise TypeError(
        "Custom analysis requires a validated ModelInspection returned by load_sbml_upload."
    )


def select_objective(
    model_or_inspection: ModelInspection,
    reaction_id: str,
    *,
    direction: str | None = None,
) -> Model:
    """Return a copied model with one reaction selected as its objective."""

    source = _source_model(model_or_inspection)
    if not isinstance(reaction_id, str) or not reaction_id.strip():
        raise ObjectiveSelectionError("reaction_id must be a non-empty string")
    selected_id = reaction_id.strip()
    if selected_id not in source.reactions:
        raise ObjectiveSelectionError(f"Reaction {selected_id!r} is not present in the model.")
    if direction is None:
        selected_direction = str(source.objective_direction)
    else:
        selected_direction = direction.strip().lower() if isinstance(direction, str) else ""
        if selected_direction not in {"max", "min"}:
            raise ObjectiveSelectionError("direction must be 'max', 'min', or None")

    copied = source.copy()
    copied.objective = copied.reactions.get_by_id(selected_id)
    copied.objective_direction = selected_direction
    return copied


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _configure_solver_timeout(model: Model, timeout_seconds: float) -> str | None:
    """Set a timeout on a copied solver, returning a warning if unsupported."""

    configuration = getattr(model.solver, "configuration", None)
    if configuration is None:
        return "The active solver exposes no timeout configuration; no solver timeout was applied."
    try:
        # GLPK's SWIG boundary requires an integer number of seconds; other
        # optlang interfaces accept the same value. Round up so a positive
        # configured timeout never becomes zero.
        configuration.timeout = int(math.ceil(timeout_seconds))
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return (
            "The active solver does not support a configurable timeout; "
            "no solver timeout was applied."
        )
    return None


def analyse_custom_model(
    model_or_inspection: ModelInspection,
    objective_reaction_id: str,
    *,
    direction: str = "max",
    fva_reaction_ids: Sequence[str] | None = None,
    fraction_of_optimum: float = 0.95,
    solver_timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
) -> CustomModelAnalysis:
    """Run FBA and optional explicitly bounded FVA on a copied custom model.

    FVA is skipped unless ``fva_reaction_ids`` is supplied.  At most
    :data:`MAX_FVA_REACTIONS` unique reactions can be requested, and COBRApy is
    forced to one process to keep an upload-triggered analysis bounded.
    """

    try:
        selected_timeout = float(solver_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise CustomModelAnalysisError("solver_timeout_seconds must be a finite number") from exc
    if (
        not math.isfinite(selected_timeout)
        or selected_timeout <= 0
        or selected_timeout > MAX_SOLVER_TIMEOUT_SECONDS
    ):
        raise CustomModelAnalysisError(
            f"solver_timeout_seconds must be in the interval (0, {MAX_SOLVER_TIMEOUT_SECONDS:g}]"
        )

    working = select_objective(
        model_or_inspection,
        objective_reaction_id,
        direction=direction,
    )
    warnings = [_CUSTOM_MODEL_WARNING]
    timeout_warning = _configure_solver_timeout(working, selected_timeout)
    if timeout_warning:
        warnings.append(timeout_warning)

    selected_fva: list[str] | None = None
    selected_fraction: float | None = None
    if fva_reaction_ids is not None:
        if isinstance(fva_reaction_ids, (str, bytes)):
            raise CustomModelAnalysisError("fva_reaction_ids must be a sequence of reaction ids")
        requested_fva = list(fva_reaction_ids)
        if any(
            not isinstance(reaction_id, str) or not reaction_id.strip()
            for reaction_id in requested_fva
        ):
            raise CustomModelAnalysisError("Every FVA reaction id must be a non-empty string")
        selected_fva = list(dict.fromkeys(reaction_id.strip() for reaction_id in requested_fva))
        if not selected_fva:
            raise CustomModelAnalysisError("fva_reaction_ids cannot be empty when FVA is requested")
        if len(selected_fva) > MAX_FVA_REACTIONS:
            raise CustomModelAnalysisError(
                f"FVA is limited to {MAX_FVA_REACTIONS} unique reactions per request."
            )
        missing = [
            reaction_id for reaction_id in selected_fva if reaction_id not in working.reactions
        ]
        if missing:
            raise CustomModelAnalysisError("Unknown FVA reaction(s): " + ", ".join(missing))
        try:
            selected_fraction = float(fraction_of_optimum)
        except (TypeError, ValueError) as exc:
            raise CustomModelAnalysisError(
                "fraction_of_optimum must be a finite number in the interval (0, 1]"
            ) from exc
        if not math.isfinite(selected_fraction) or not 0 < selected_fraction <= 1:
            raise CustomModelAnalysisError("fraction_of_optimum must be in the interval (0, 1]")

    try:
        solution = working.optimize()
    except Exception as exc:
        raise CustomModelAnalysisError("The solver could not analyse the uploaded model.") from exc
    status = str(solution.status)
    if status != "optimal":
        warnings.append(f"Solver status was {status!r}; fluxes and FVA ranges are unavailable.")
        return CustomModelAnalysis(
            model_id=str(working.id or ""),
            objective_reaction_id=objective_reaction_id,
            objective_direction=str(working.objective_direction),
            solver_timeout_seconds=selected_timeout,
            status=status,
            objective_value=None,
            fluxes={},
            fva_fraction_of_optimum=None,
            fva_ranges={},
            warnings=tuple(warnings),
        )

    objective_value = _finite_float(solution.objective_value)
    if objective_value is None:
        raise CustomModelAnalysisError("The solver returned a non-finite objective value.")
    fluxes: dict[str, float] = {}
    for reaction in working.reactions:
        value = _finite_float(solution.fluxes[reaction.id])
        if value is None:
            raise CustomModelAnalysisError(
                f"The solver returned a non-finite flux for reaction {reaction.id!r}."
            )
        fluxes[reaction.id] = value

    fva_ranges: dict[str, FluxRange] = {}
    if selected_fva is not None and timeout_warning:
        warnings.append(
            "FVA was skipped because the active solver cannot enforce the required time limit."
        )
        selected_fva = None
        selected_fraction = None

    if selected_fva is not None:
        assert selected_fraction is not None
        if selected_fraction < 1 and (working.objective_direction != "max" or objective_value < 0):
            raise CustomModelAnalysisError(
                "Fractional FVA is only supported for a non-negative maximization optimum; "
                "use fraction_of_optimum=1 for this objective."
            )
        deadline = monotonic() + MAX_FVA_WALL_SECONDS
        for reaction_id in selected_fva:
            remaining = deadline - monotonic()
            if remaining < 2:
                raise CustomModelAnalysisError(
                    f"FVA exceeded its {MAX_FVA_WALL_SECONDS:g}-second aggregate time budget."
                )
            per_solve_timeout = min(selected_timeout, max(1, math.floor(remaining / 2)))
            if _configure_solver_timeout(working, per_solve_timeout):
                raise CustomModelAnalysisError(
                    "The active solver stopped accepting timeout configuration during FVA."
                )
            try:
                frame = flux_variability_analysis(
                    working,
                    reaction_list=[reaction_id],
                    fraction_of_optimum=selected_fraction,
                    processes=1,
                )
            except Exception as exc:
                raise CustomModelAnalysisError("FVA failed for the uploaded model.") from exc
            if monotonic() > deadline:
                raise CustomModelAnalysisError(
                    f"FVA exceeded its {MAX_FVA_WALL_SECONDS:g}-second aggregate time budget."
                )
            minimum = _finite_float(frame.loc[reaction_id, "minimum"])
            maximum = _finite_float(frame.loc[reaction_id, "maximum"])
            if minimum is None or maximum is None:
                raise CustomModelAnalysisError(
                    f"FVA returned a non-finite range for reaction {reaction_id!r}."
                )
            reaction = working.reactions.get_by_id(reaction_id)
            fva_ranges[reaction_id] = FluxRange(
                reaction_id=reaction_id,
                reaction_name=reaction.name,
                minimum=minimum,
                maximum=maximum,
            )
        warnings.append(
            "FVA ranges are reaction-wise alternatives and do not form one simultaneous flux vector."
        )

    return CustomModelAnalysis(
        model_id=str(working.id or ""),
        objective_reaction_id=objective_reaction_id,
        objective_direction=str(working.objective_direction),
        solver_timeout_seconds=selected_timeout,
        status=status,
        objective_value=objective_value,
        fluxes=fluxes,
        fva_fraction_of_optimum=selected_fraction,
        fva_ranges=fva_ranges,
        warnings=tuple(warnings),
    )
