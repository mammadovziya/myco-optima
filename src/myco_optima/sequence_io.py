"""Bounded, sequence-only FASTA intake for reconstruction handoffs.

This module deliberately does not annotate sequences or construct a metabolic
model.  It validates nucleotide (``.fna``) and protein (``.faa``) FASTA files,
computes descriptive statistics, and returns metadata that can be carried into
an external reconstruction workflow.  Raw sequence bodies are never retained in
the returned inspection objects.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import PurePath
from statistics import median
from typing import Any, Literal

DEFAULT_MAX_SEQUENCE_BYTES = 100_000_000
DEFAULT_MAX_COMBINED_SEQUENCE_BYTES = 150_000_000
DEFAULT_MAX_SEQUENCE_RECORDS = 200_000
DEFAULT_MAX_SEQUENCE_RESIDUES = 100_000_000
DEFAULT_MAX_COMBINED_SEQUENCE_RESIDUES = 150_000_000
DEFAULT_MAX_RECORD_RESIDUES = 50_000_000
DEFAULT_MAX_SEQUENCE_LINE_LENGTH = 100_000_000
DEFAULT_MAX_PHYSICAL_LINES = 1_700_000
DEFAULT_MAX_BLANK_LINES = 10_000
DEFAULT_MAX_HEADER_LENGTH = 4_096
DEFAULT_MAX_IDENTIFIER_LENGTH = 512
DEFAULT_MAX_IDENTIFIER_BYTES = 8_000_000
DEFAULT_PREVIEW_RECORDS = 20
MAX_PREVIEW_RECORDS = 100
PARSER_VERSION = "1"

SequenceType = Literal["nucleotide", "protein"]
NucleotideRole = Literal["assembly", "cds"]

_NUCLEOTIDE_ALPHABET = frozenset(b"ACGTURYSWKMBDHVN")
_CANONICAL_NUCLEOTIDES = frozenset(b"ACGTU")
_PROTEIN_ALPHABET = frozenset(b"ACDEFGHIKLMNPQRSTVWYBXZJUO")
_AMBIGUOUS_PROTEINS = frozenset(b"BXZJ")
_CHUNK_BYTES = 1_048_576


class SequenceIntakeError(ValueError):
    """Base class for safe, user-facing sequence-intake failures."""


class UnsupportedSequenceFileError(SequenceIntakeError):
    """Raised when a payload or filename is not a supported FASTA input."""


class SequenceUploadTooLargeError(SequenceIntakeError):
    """Raised when an upload exceeds its byte budget."""


class InvalidFastaError(SequenceIntakeError):
    """Raised when FASTA structure or sequence symbols are invalid."""


class SequenceLimitError(SequenceIntakeError):
    """Raised when a bounded parser limit is exceeded."""


class SequencePairingError(SequenceIntakeError):
    """Raised when nucleotide and protein inspections cannot be paired safely."""


@dataclass(frozen=True, slots=True)
class FastaLimits:
    """Resource limits applied before and during one FASTA inspection."""

    max_bytes: int = DEFAULT_MAX_SEQUENCE_BYTES
    max_records: int = DEFAULT_MAX_SEQUENCE_RECORDS
    max_total_residues: int = DEFAULT_MAX_SEQUENCE_RESIDUES
    max_record_residues: int = DEFAULT_MAX_RECORD_RESIDUES
    max_sequence_line_length: int = DEFAULT_MAX_SEQUENCE_LINE_LENGTH
    max_physical_lines: int = DEFAULT_MAX_PHYSICAL_LINES
    max_blank_lines: int = DEFAULT_MAX_BLANK_LINES
    max_header_length: int = DEFAULT_MAX_HEADER_LENGTH
    max_identifier_length: int = DEFAULT_MAX_IDENTIFIER_LENGTH
    max_identifier_bytes: int = DEFAULT_MAX_IDENTIFIER_BYTES
    preview_records: int = DEFAULT_PREVIEW_RECORDS

    def __post_init__(self) -> None:
        for name, value in (
            ("max_bytes", self.max_bytes),
            ("max_records", self.max_records),
            ("max_total_residues", self.max_total_residues),
            ("max_record_residues", self.max_record_residues),
            ("max_sequence_line_length", self.max_sequence_line_length),
            ("max_physical_lines", self.max_physical_lines),
            ("max_blank_lines", self.max_blank_lines),
            ("max_header_length", self.max_header_length),
            ("max_identifier_length", self.max_identifier_length),
            ("max_identifier_bytes", self.max_identifier_bytes),
            ("preview_records", self.preview_records),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        hard_maxima = {
            "max_bytes": DEFAULT_MAX_SEQUENCE_BYTES,
            "max_records": DEFAULT_MAX_SEQUENCE_RECORDS,
            "max_total_residues": DEFAULT_MAX_SEQUENCE_RESIDUES,
            "max_record_residues": DEFAULT_MAX_RECORD_RESIDUES,
            "max_sequence_line_length": DEFAULT_MAX_SEQUENCE_LINE_LENGTH,
            "max_physical_lines": DEFAULT_MAX_PHYSICAL_LINES,
            "max_blank_lines": DEFAULT_MAX_BLANK_LINES,
            "max_header_length": DEFAULT_MAX_HEADER_LENGTH,
            "max_identifier_length": DEFAULT_MAX_IDENTIFIER_LENGTH,
            "max_identifier_bytes": DEFAULT_MAX_IDENTIFIER_BYTES,
        }
        for name, maximum in hard_maxima.items():
            if getattr(self, name) > maximum:
                raise ValueError(f"{name} must not exceed the safe default of {maximum:,}")
        if self.preview_records > MAX_PREVIEW_RECORDS:
            raise ValueError(f"preview_records must not exceed {MAX_PREVIEW_RECORDS}")


@dataclass(frozen=True, slots=True)
class FastaRecordPreview:
    """Bounded record metadata; no nucleotide or amino-acid sequence is stored."""

    identifier: str
    description: str
    length: int
    gc_percent: float | None = None
    ambiguous_residues: int = 0
    terminal_stop_marker: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identifier, str)
            or not self.identifier
            or len(self.identifier) > DEFAULT_MAX_IDENTIFIER_LENGTH
            or any(
                not character.isprintable() or character.isspace() for character in self.identifier
            )
        ):
            raise ValueError("FASTA preview identifier is invalid")
        if (
            not isinstance(self.description, str)
            or len(self.description) > DEFAULT_MAX_HEADER_LENGTH
            or any(
                not character.isprintable() or (character.isspace() and character != " ")
                for character in self.description
            )
        ):
            raise ValueError("FASTA preview description is invalid")
        if not isinstance(self.length, int) or isinstance(self.length, bool) or self.length <= 0:
            raise ValueError("FASTA preview length must be a positive integer")
        if self.gc_percent is not None and (
            not isinstance(self.gc_percent, (int, float))
            or isinstance(self.gc_percent, bool)
            or not math.isfinite(self.gc_percent)
            or not 0 <= self.gc_percent <= 100
        ):
            raise ValueError("FASTA preview GC percentage must be finite and bounded")
        if (
            not isinstance(self.ambiguous_residues, int)
            or isinstance(self.ambiguous_residues, bool)
            or not 0 <= self.ambiguous_residues <= self.length
        ):
            raise ValueError("FASTA preview ambiguity count is inconsistent")
        if not isinstance(self.terminal_stop_marker, bool):
            raise ValueError("FASTA preview terminal-stop marker must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "description": self.description,
            "length": self.length,
            "gc_percent": self.gc_percent,
            "ambiguous_residues": self.ambiguous_residues,
            "terminal_stop_marker": self.terminal_stop_marker,
        }


@dataclass(frozen=True, slots=True)
class SequenceInspection:
    """Validated sequence-file metadata without raw sequence bodies."""

    filename: str
    sequence_type: SequenceType
    extension: str
    size_bytes: int
    sha256: str
    record_count: int
    total_residues: int
    minimum_length: int
    median_length: float
    maximum_length: int
    ambiguous_residue_count: int
    ambiguous_residue_fraction: float
    preview_records: tuple[FastaRecordPreview, ...]
    warnings: tuple[str, ...]
    gc_percent: float | None = None
    n_percent: float | None = None
    n50: int | None = None
    terminal_stop_marker_count: int = 0
    record_ids: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        extension, inferred_type = _validate_filename(self.filename)
        if self.extension != extension or self.sequence_type != inferred_type:
            raise ValueError("Sequence inspection type does not match its filename extension")
        integer_fields = (
            self.size_bytes,
            self.record_count,
            self.total_residues,
            self.minimum_length,
            self.maximum_length,
            self.ambiguous_residue_count,
            self.terminal_stop_marker_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
            raise ValueError("Sequence inspection counts must be integers")
        if self.size_bytes <= 0 or self.record_count <= 0 or self.total_residues <= 0:
            raise ValueError("Sequence inspection counts and byte size must be positive")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("Sequence inspection SHA-256 must be lowercase hexadecimal")
        if not isinstance(self.median_length, (int, float)) or isinstance(self.median_length, bool):
            raise ValueError("Sequence inspection median length must be numeric")
        if not (
            0
            < self.minimum_length
            <= self.median_length
            <= self.maximum_length
            <= self.total_residues
        ) or not math.isfinite(self.median_length):
            raise ValueError("Sequence inspection length statistics are inconsistent")
        if not 0 <= self.ambiguous_residue_count <= self.total_residues:
            raise ValueError("Sequence inspection ambiguity count is inconsistent")
        expected_fraction = self.ambiguous_residue_count / self.total_residues
        if not math.isfinite(self.ambiguous_residue_fraction) or not math.isclose(
            self.ambiguous_residue_fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Sequence inspection ambiguity fraction is inconsistent")
        if not isinstance(self.record_ids, tuple) or (
            len(self.record_ids) != self.record_count
            or len(set(self.record_ids)) != self.record_count
        ):
            raise ValueError("Sequence inspection record identifiers are inconsistent")
        if (
            sum(len(identifier.encode("utf-8")) for identifier in self.record_ids)
            > DEFAULT_MAX_IDENTIFIER_BYTES
        ):
            raise ValueError("Sequence inspection retains too much identifier text")
        if any(
            not identifier
            or len(identifier) > DEFAULT_MAX_IDENTIFIER_LENGTH
            or any(not character.isprintable() or character.isspace() for character in identifier)
            for identifier in self.record_ids
        ):
            raise ValueError("Sequence inspection contains an invalid record identifier")
        if not isinstance(self.preview_records, tuple) or len(self.preview_records) > min(
            self.record_count, MAX_PREVIEW_RECORDS
        ):
            raise ValueError("Sequence inspection preview exceeds its bounded size")
        if any(
            not isinstance(preview, FastaRecordPreview) or preview.length > self.maximum_length
            for preview in self.preview_records
        ):
            raise ValueError("Sequence inspection preview is inconsistent")
        if (
            tuple(preview.identifier for preview in self.preview_records)
            != self.record_ids[: len(self.preview_records)]
        ):
            raise ValueError("Sequence inspection preview identifiers are inconsistent")
        if not 0 <= self.terminal_stop_marker_count <= self.record_count:
            raise ValueError("Sequence inspection terminal-stop count is inconsistent")
        if any(
            not isinstance(warning, str)
            or len(warning) > 500
            or any(not character.isprintable() for character in warning)
            for warning in self.warnings
        ):
            raise ValueError("Sequence inspection warnings must be bounded printable text")
        for percentage in (self.gc_percent, self.n_percent):
            if percentage is not None and (
                not math.isfinite(percentage) or not 0 <= percentage <= 100
            ):
                raise ValueError("Sequence inspection percentages must be finite and bounded")
        if self.sequence_type == "nucleotide":
            if self.terminal_stop_marker_count or self.n50 is None:
                raise ValueError("Nucleotide inspection stop/N50 fields are inconsistent")
            if (
                not isinstance(self.n50, int)
                or isinstance(self.n50, bool)
                or not self.minimum_length <= self.n50 <= self.maximum_length
                or any(preview.terminal_stop_marker for preview in self.preview_records)
            ):
                raise ValueError("Nucleotide inspection N50 is inconsistent")
        elif (
            self.gc_percent is not None
            or self.n_percent is not None
            or self.n50 is not None
            or any(preview.gc_percent is not None for preview in self.preview_records)
        ):
            raise ValueError("Protein inspection must not contain nucleotide statistics")

    @property
    def sequence_count(self) -> int:
        return self.record_count if self.sequence_type == "nucleotide" else 0

    @property
    def protein_count(self) -> int:
        return self.record_count if self.sequence_type == "protein" else 0

    @property
    def total_bp(self) -> int:
        return self.total_residues if self.sequence_type == "nucleotide" else 0

    @property
    def total_aa(self) -> int:
        return self.total_residues if self.sequence_type == "protein" else 0

    def preview(self, limit: int = DEFAULT_PREVIEW_RECORDS) -> list[dict[str, Any]]:
        """Return at most the bounded preview captured while parsing."""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        return [item.to_dict() for item in self.preview_records[:limit]]

    def metadata(self) -> dict[str, Any]:
        """Return strict-JSON-safe, bounded provenance and descriptive statistics."""

        result: dict[str, Any] = {
            "parser_version": PARSER_VERSION,
            "filename": self.filename,
            "sequence_type": self.sequence_type,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "record_count": self.record_count,
            "total_residues": self.total_residues,
            "minimum_length": self.minimum_length,
            "median_length": self.median_length,
            "maximum_length": self.maximum_length,
            "ambiguous_residue_count": self.ambiguous_residue_count,
            "ambiguous_residue_fraction": self.ambiguous_residue_fraction,
            "terminal_stop_marker_count": self.terminal_stop_marker_count,
            "warnings": list(self.warnings),
            "preview": self.preview(),
            "identifier_policy": "exact, case-sensitive first FASTA header token",
            "raw_sequences_retained": False,
            "supports_fba": False,
        }
        if self.sequence_type == "nucleotide":
            result.update(
                {
                    "sequence_count": self.record_count,
                    "total_bp": self.total_residues,
                    "gc_percent": self.gc_percent,
                    "gc_denominator": "A+C+G+T+U (ambiguity codes excluded)",
                    "n_percent": self.n_percent,
                    "n50": self.n50,
                }
            )
        else:
            result.update(
                {
                    "protein_count": self.record_count,
                    "total_aa": self.total_residues,
                }
            )
        return result

    to_dict = metadata


@dataclass(frozen=True, slots=True)
class SequencePairInspection:
    """A bounded co-upload relationship, not an inferred biological linkage."""

    nucleotide: SequenceInspection
    protein: SequenceInspection
    nucleotide_role: NucleotideRole
    combined_bytes: int
    combined_residues: int
    syntactic_id_overlap_evaluated: bool
    syntactic_id_overlap_count: int
    syntactic_id_overlap_preview: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.nucleotide.sequence_type != "nucleotide" or self.protein.sequence_type != "protein":
            raise ValueError("Sequence pair must contain nucleotide then protein inspections")
        if self.nucleotide_role not in {"assembly", "cds"}:
            raise ValueError("Sequence pair nucleotide role is invalid")
        for name, value in (
            ("combined_bytes", self.combined_bytes),
            ("combined_residues", self.combined_residues),
            ("syntactic_id_overlap_count", self.syntactic_id_overlap_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Sequence pair {name} must be a non-negative integer")
        if not isinstance(self.syntactic_id_overlap_evaluated, bool):
            raise ValueError("Sequence pair overlap-evaluated flag must be boolean")
        if (
            not isinstance(self.syntactic_id_overlap_preview, tuple)
            or len(self.syntactic_id_overlap_preview) > DEFAULT_PREVIEW_RECORDS
            or len(set(self.syntactic_id_overlap_preview)) != len(self.syntactic_id_overlap_preview)
            or any(
                not isinstance(identifier, str)
                or identifier not in self.nucleotide.record_ids
                or identifier not in self.protein.record_ids
                for identifier in self.syntactic_id_overlap_preview
            )
        ):
            raise ValueError("Sequence pair overlap preview is inconsistent")
        if self.combined_bytes != self.nucleotide.size_bytes + self.protein.size_bytes:
            raise ValueError("Sequence pair byte total is inconsistent")
        if self.combined_residues != self.nucleotide.total_residues + self.protein.total_residues:
            raise ValueError("Sequence pair residue total is inconsistent")
        if self.syntactic_id_overlap_count < len(self.syntactic_id_overlap_preview):
            raise ValueError("Sequence pair overlap preview is inconsistent")
        if self.syntactic_id_overlap_count > min(
            self.nucleotide.record_count, self.protein.record_count
        ):
            raise ValueError("Sequence pair overlap count is inconsistent")
        if self.nucleotide_role == "assembly" and (
            self.syntactic_id_overlap_evaluated
            or self.syntactic_id_overlap_count
            or self.syntactic_id_overlap_preview
        ):
            raise ValueError("Assembly/protein pair must not claim identifier linkage")
        if self.nucleotide_role == "cds" and not self.syntactic_id_overlap_evaluated:
            raise ValueError("CDS/protein pair must record that exact identifiers were compared")

    def metadata(self) -> dict[str, Any]:
        return {
            "nucleotide": _handoff_file_metadata(self.nucleotide),
            "protein": _handoff_file_metadata(self.protein),
            "nucleotide_role": self.nucleotide_role,
            "combined_bytes": self.combined_bytes,
            "combined_residues": self.combined_residues,
            "syntactic_id_overlap_evaluated": self.syntactic_id_overlap_evaluated,
            "syntactic_id_overlap_count": self.syntactic_id_overlap_count,
            "syntactic_id_overlap_preview": list(self.syntactic_id_overlap_preview),
            "identifier_policy": "exact, case-sensitive first FASTA header token",
            "biological_linkage_inferred": False,
        }

    to_dict = metadata


@dataclass(frozen=True, slots=True)
class ReconstructionHandoff:
    """Serialization-safe manifest for an external GEM reconstruction workflow."""

    payload: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise ValueError("Reconstruction handoff payload must be a dictionary")
        _validate_strict_json_value(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return _copy_json_value(self.payload)

    metadata = to_dict


@dataclass(slots=True)
class _RecordAccumulator:
    identifier: str
    description: str
    length: int = 0
    canonical_nucleotides: int = 0
    gc_count: int = 0
    n_count: int = 0
    ambiguous_count: int = 0
    terminal_stop_marker: bool = False


def _copy_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_json_value(item) for item in value]
    return value


def _validate_strict_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError("Reconstruction handoff numbers must be finite")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Reconstruction handoff keys must be text")
        for item in value.values():
            _validate_strict_json_value(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_strict_json_value(item)
        return
    raise ValueError("Reconstruction handoff contains a non-JSON value")


def _safe_token(value: str, limit: int = 80) -> str:
    visible = "".join(character if character.isprintable() else "?" for character in value)
    return visible if len(visible) <= limit else visible[: limit - 1] + "…"


def _validate_filename(filename: str) -> tuple[str, SequenceType]:
    if not isinstance(filename, str):
        raise UnsupportedSequenceFileError("Sequence filename must be text.")
    if not filename or len(filename) > 255:
        raise UnsupportedSequenceFileError("Sequence filename is empty or too long.")
    if filename != PurePath(filename).name or "/" in filename or "\\" in filename:
        raise UnsupportedSequenceFileError("Sequence filename must not contain a path.")
    if filename != filename.strip(" ") or any(
        not character.isprintable() or (character.isspace() and character != " ")
        for character in filename
    ):
        raise UnsupportedSequenceFileError(
            "Sequence filename contains invisible or control characters."
        )
    lowered = filename.lower()
    if lowered.endswith((".gz", ".zip", ".bz2", ".xz")):
        raise UnsupportedSequenceFileError("Compressed sequence uploads are not supported.")
    extension = PurePath(filename).suffix.lower()
    if extension == ".fna":
        return extension, "nucleotide"
    if extension == ".faa":
        return extension, "protein"
    raise UnsupportedSequenceFileError("Sequence upload must use a .fna or .faa extension.")


def _payload_view(payload: bytes | bytearray | memoryview, limits: FastaLimits) -> memoryview:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise UnsupportedSequenceFileError("Sequence upload must be bytes-like.")
    try:
        view = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise UnsupportedSequenceFileError("Sequence upload must be contiguous bytes.") from exc
    if not view.c_contiguous:
        raise UnsupportedSequenceFileError("Sequence upload must be contiguous bytes.")
    if len(view) == 0:
        raise InvalidFastaError("Sequence upload is empty.")
    if len(view) > limits.max_bytes:
        raise SequenceUploadTooLargeError(
            f"Sequence upload exceeds the {limits.max_bytes:,}-byte limit."
        )
    return view


def _parse_header(
    raw_header: bytes,
    *,
    line_number: int,
    limits: FastaLimits,
    seen_ids: set[str],
) -> tuple[str, str]:
    if len(raw_header) > limits.max_header_length + 1:
        raise SequenceLimitError(
            f"FASTA header on line {line_number} exceeds the configured length limit."
        )
    try:
        decoded = raw_header[1:].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InvalidFastaError(f"FASTA header on line {line_number} is not valid UTF-8.") from exc
    if any(
        not character.isprintable() or (character.isspace() and character != " ")
        for character in decoded
    ):
        raise InvalidFastaError(f"FASTA header on line {line_number} contains control characters.")
    text = decoded.strip(" ")
    if not text:
        raise InvalidFastaError(f"FASTA header on line {line_number} has no identifier.")
    parts = text.split(maxsplit=1)
    identifier = parts[0]
    description = parts[1] if len(parts) == 2 else ""
    if len(identifier) > limits.max_identifier_length:
        raise SequenceLimitError(
            f"FASTA identifier on line {line_number} exceeds the configured length limit."
        )
    if identifier in seen_ids:
        raise InvalidFastaError(f"Duplicate FASTA identifier: {_safe_token(identifier)}.")
    seen_ids.add(identifier)
    return identifier, description


def _n50(lengths: list[int], total: int) -> int:
    threshold = math.ceil(total / 2)
    cumulative = 0
    for length in sorted(lengths, reverse=True):
        cumulative += length
        if cumulative >= threshold:
            return length
    return 0


def _build_inspection(
    view: memoryview,
    filename: str,
    extension: str,
    sequence_type: SequenceType,
    limits: FastaLimits,
) -> SequenceInspection:
    seen_ids: set[str] = set()
    record_ids: list[str] = []
    previews: list[FastaRecordPreview] = []
    lengths: list[int] = []
    current: _RecordAccumulator | None = None
    total_residues = 0
    total_ambiguous = 0
    total_canonical_nucleotides = 0
    total_gc = 0
    total_n = 0
    terminal_stops = 0
    blank_lines = 0
    physical_lines = 0
    identifier_bytes_total = 0
    nucleotide_u_count = 0
    content_hasher = hashlib.sha256()

    line_number = 1
    at_line_start = True
    line_kind: Literal["header", "sequence"] | None = None
    header_buffer = bytearray()
    sequence_line_length = 0
    pending_sequence_cr = False

    def finish_record() -> None:
        nonlocal current
        nonlocal total_residues, total_ambiguous
        nonlocal total_canonical_nucleotides, total_gc, total_n, terminal_stops
        if current is None:
            return
        if current.length == 0:
            raise InvalidFastaError(
                f"FASTA record {_safe_token(current.identifier)} has no sequence residues."
            )
        lengths.append(current.length)
        total_residues += current.length
        total_ambiguous += current.ambiguous_count
        total_canonical_nucleotides += current.canonical_nucleotides
        total_gc += current.gc_count
        total_n += current.n_count
        terminal_stops += int(current.terminal_stop_marker)
        if total_residues > limits.max_total_residues:
            raise SequenceLimitError(
                f"FASTA content exceeds the {limits.max_total_residues:,}-residue limit."
            )
        if len(previews) < limits.preview_records:
            record_gc = (
                100.0 * current.gc_count / current.canonical_nucleotides
                if current.canonical_nucleotides
                else None
            )
            previews.append(
                FastaRecordPreview(
                    identifier=current.identifier,
                    description=current.description,
                    length=current.length,
                    gc_percent=record_gc if sequence_type == "nucleotide" else None,
                    ambiguous_residues=current.ambiguous_count,
                    terminal_stop_marker=current.terminal_stop_marker,
                )
            )
        current = None

    def finish_header() -> None:
        nonlocal current, identifier_bytes_total
        if header_buffer.endswith(b"\r"):
            del header_buffer[-1]
        finish_record()
        identifier, description = _parse_header(
            bytes(header_buffer),
            line_number=line_number,
            limits=limits,
            seen_ids=seen_ids,
        )
        if len(record_ids) >= limits.max_records:
            raise SequenceLimitError(
                f"FASTA content exceeds the {limits.max_records:,}-record limit."
            )
        identifier_bytes_total += len(identifier.encode("utf-8"))
        if identifier_bytes_total > limits.max_identifier_bytes:
            raise SequenceLimitError("FASTA identifiers exceed the aggregate retained-text limit.")
        record_ids.append(identifier)
        current = _RecordAccumulator(identifier=identifier, description=description)

    def consume_sequence(fragment: bytes, *, ends_line: bool) -> None:
        nonlocal pending_sequence_cr, sequence_line_length, nucleotide_u_count
        if pending_sequence_cr:
            if fragment or not ends_line:
                raise InvalidFastaError(
                    f"FASTA sequence line {line_number} contains a carriage return."
                )
            pending_sequence_cr = False
            return
        if not ends_line and fragment.endswith(b"\r"):
            fragment = fragment[:-1]
            pending_sequence_cr = True
        elif ends_line and fragment.endswith(b"\r"):
            fragment = fragment[:-1]
        sequence_line_length += len(fragment)
        if sequence_line_length > limits.max_sequence_line_length:
            raise SequenceLimitError(
                f"FASTA sequence line {line_number} exceeds the configured length limit."
            )
        if not fragment:
            return
        if current is None:
            raise InvalidFastaError(
                f"FASTA sequence data appears before the first header on line {line_number}."
            )
        upper = fragment.upper()
        if sequence_type == "nucleotide":
            invalid = set(upper).difference(_NUCLEOTIDE_ALPHABET)
            if invalid:
                symbol = chr(min(invalid)) if min(invalid) < 128 else "non-ASCII"
                raise InvalidFastaError(
                    f"Invalid nucleotide symbol {symbol!r} on FASTA line {line_number}."
                )
            fragment_length = len(upper)
            current.length += fragment_length
            current.gc_count += upper.count(b"G") + upper.count(b"C")
            current.canonical_nucleotides += sum(
                upper.count(bytes((symbol,))) for symbol in _CANONICAL_NUCLEOTIDES
            )
            current.n_count += upper.count(b"N")
            current.ambiguous_count += sum(
                upper.count(bytes((symbol,)))
                for symbol in _NUCLEOTIDE_ALPHABET.difference(_CANONICAL_NUCLEOTIDES)
            )
            nucleotide_u_count += upper.count(b"U")
        else:
            invalid = set(upper).difference(_PROTEIN_ALPHABET | {ord("*")})
            if invalid:
                symbol = chr(min(invalid)) if min(invalid) < 128 else "non-ASCII"
                raise InvalidFastaError(
                    f"Invalid amino-acid symbol {symbol!r} on FASTA line {line_number}."
                )
            stop_count = upper.count(b"*")
            if stop_count:
                if stop_count != 1 or upper[-1:] != b"*" or current.terminal_stop_marker:
                    raise InvalidFastaError(
                        f"Protein record {_safe_token(current.identifier)} has an internal or repeated stop marker."
                    )
                upper = upper[:-1]
                current.terminal_stop_marker = True
            elif current.terminal_stop_marker and upper:
                raise InvalidFastaError(
                    f"Protein record {_safe_token(current.identifier)} continues after a terminal stop marker."
                )
            current.length += len(upper)
            current.ambiguous_count += sum(
                upper.count(bytes((symbol,))) for symbol in _AMBIGUOUS_PROTEINS
            )
        if current.length > limits.max_record_residues:
            raise SequenceLimitError(
                f"FASTA record {_safe_token(current.identifier)} exceeds the per-record residue limit."
            )
        if total_residues + current.length > limits.max_total_residues:
            raise SequenceLimitError(
                f"FASTA content exceeds the {limits.max_total_residues:,}-residue limit."
            )

    for start in range(0, len(view), _CHUNK_BYTES):
        chunk = bytes(view[start : start + _CHUNK_BYTES])
        content_hasher.update(chunk)
        cursor = 0
        while cursor < len(chunk):
            newline = chunk.find(b"\n", cursor)
            ends_line = newline >= 0
            end = newline if ends_line else len(chunk)
            fragment = chunk[cursor:end]

            if at_line_start:
                line_kind = "header" if fragment.startswith(b">") else "sequence"
                at_line_start = False
                sequence_line_length = 0
                header_buffer.clear()

            if line_kind == "header":
                header_buffer.extend(fragment)
                if len(header_buffer) > limits.max_header_length + 2:
                    raise SequenceLimitError(
                        f"FASTA header on line {line_number} exceeds the configured length limit."
                    )
            else:
                consume_sequence(fragment, ends_line=ends_line)

            if ends_line:
                if line_kind == "header":
                    finish_header()
                elif sequence_line_length == 0 and not pending_sequence_cr:
                    blank_lines += 1
                    if blank_lines > limits.max_blank_lines:
                        raise SequenceLimitError("FASTA content exceeds the blank-line limit.")
                if pending_sequence_cr:
                    pending_sequence_cr = False
                physical_lines += 1
                if physical_lines > limits.max_physical_lines:
                    raise SequenceLimitError("FASTA content exceeds the physical-line limit.")
                line_number += 1
                at_line_start = True
                line_kind = None
                cursor = newline + 1
            else:
                cursor = len(chunk)

    if not at_line_start:
        physical_lines += 1
        if physical_lines > limits.max_physical_lines:
            raise SequenceLimitError("FASTA content exceeds the physical-line limit.")
        if pending_sequence_cr:
            raise InvalidFastaError(
                f"FASTA sequence line {line_number} ends with a bare carriage return."
            )
        if line_kind == "header":
            finish_header()
    finish_record()

    if not lengths:
        raise InvalidFastaError("FASTA upload contains no sequence records.")

    warnings: list[str] = []
    if blank_lines:
        warnings.append(f"Ignored {blank_lines:,} blank FASTA line(s).")
    if total_ambiguous:
        warnings.append(
            f"Found {total_ambiguous:,} ambiguity-coded residue(s); no biological meaning was inferred."
        )
    if nucleotide_u_count and sequence_type == "nucleotide":
        warnings.append(
            "Nucleotide input contains U; it was counted in the canonical GC denominator."
        )
    if sequence_type == "nucleotide" and total_canonical_nucleotides == 0:
        warnings.append(
            "GC percentage is undefined because no canonical A/C/G/T/U bases were present."
        )
    if terminal_stops:
        warnings.append(
            f"Excluded {terminal_stops:,} terminal stop marker(s) from amino-acid lengths."
        )

    gc_percent = (
        100.0 * total_gc / total_canonical_nucleotides
        if sequence_type == "nucleotide" and total_canonical_nucleotides
        else None
    )
    n_percent = 100.0 * total_n / total_residues if sequence_type == "nucleotide" else None
    ambiguous_fraction = total_ambiguous / total_residues
    return SequenceInspection(
        filename=filename,
        sequence_type=sequence_type,
        extension=extension,
        size_bytes=len(view),
        sha256=content_hasher.hexdigest(),
        record_count=len(lengths),
        total_residues=total_residues,
        minimum_length=min(lengths),
        median_length=float(median(lengths)),
        maximum_length=max(lengths),
        ambiguous_residue_count=total_ambiguous,
        ambiguous_residue_fraction=ambiguous_fraction,
        preview_records=tuple(previews),
        warnings=tuple(warnings),
        gc_percent=gc_percent,
        n_percent=n_percent,
        n50=_n50(lengths, total_residues) if sequence_type == "nucleotide" else None,
        terminal_stop_marker_count=terminal_stops,
        record_ids=tuple(record_ids),
    )


def load_fasta_upload(
    payload: bytes | bytearray | memoryview,
    filename: str,
    *,
    limits: FastaLimits | None = None,
) -> SequenceInspection:
    """Validate one ``.fna`` or ``.faa`` payload without retaining its sequences."""

    selected_limits = FastaLimits() if limits is None else limits
    if not isinstance(selected_limits, FastaLimits):
        raise TypeError("limits must be a FastaLimits instance")
    extension, sequence_type = _validate_filename(filename)
    view = _payload_view(payload, selected_limits)
    return _build_inspection(view, filename, extension, sequence_type, selected_limits)


inspect_fasta_upload = load_fasta_upload


def pair_fasta_inspections(
    nucleotide: SequenceInspection,
    protein: SequenceInspection,
    *,
    nucleotide_role: NucleotideRole = "assembly",
    max_combined_bytes: int = DEFAULT_MAX_COMBINED_SEQUENCE_BYTES,
    max_combined_residues: int = DEFAULT_MAX_COMBINED_SEQUENCE_RESIDUES,
) -> SequencePairInspection:
    """Create a bounded co-upload record without claiming biological linkage."""

    if not isinstance(nucleotide, SequenceInspection) or nucleotide.sequence_type != "nucleotide":
        raise SequencePairingError("The first paired inspection must be nucleotide FASTA.")
    if not isinstance(protein, SequenceInspection) or protein.sequence_type != "protein":
        raise SequencePairingError("The second paired inspection must be protein FASTA.")
    if nucleotide_role not in {"assembly", "cds"}:
        raise SequencePairingError("nucleotide_role must be 'assembly' or 'cds'.")
    for name, value in (
        ("max_combined_bytes", max_combined_bytes),
        ("max_combined_residues", max_combined_residues),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    combined_bytes = nucleotide.size_bytes + protein.size_bytes
    combined_residues = nucleotide.total_residues + protein.total_residues
    if combined_bytes > max_combined_bytes:
        raise SequencePairingError(
            f"Co-upload exceeds the {max_combined_bytes:,}-byte accepted-processing limit."
        )
    if combined_residues > max_combined_residues:
        raise SequencePairingError(
            f"Co-upload exceeds the {max_combined_residues:,}-residue limit."
        )

    overlap_count = 0
    overlap_preview: list[str] = []
    evaluated = nucleotide_role == "cds"
    if evaluated:
        protein_ids = set(protein.record_ids)
        for identifier in nucleotide.record_ids:
            if identifier in protein_ids:
                overlap_count += 1
                if len(overlap_preview) < DEFAULT_PREVIEW_RECORDS:
                    overlap_preview.append(identifier)
    return SequencePairInspection(
        nucleotide=nucleotide,
        protein=protein,
        nucleotide_role=nucleotide_role,
        combined_bytes=combined_bytes,
        combined_residues=combined_residues,
        syntactic_id_overlap_evaluated=evaluated,
        syntactic_id_overlap_count=overlap_count,
        syntactic_id_overlap_preview=tuple(sorted(overlap_preview)),
    )


def _handoff_file_metadata(inspection: SequenceInspection) -> dict[str, Any]:
    metadata = inspection.metadata()
    metadata.pop("preview", None)
    return metadata


def build_reconstruction_handoff(
    intake: SequenceInspection | SequencePairInspection,
    *,
    nucleotide_role: NucleotideRole | None = None,
) -> ReconstructionHandoff:
    """Build a deterministic, strict-JSON-safe external-reconstruction manifest."""

    if isinstance(intake, SequencePairInspection):
        files = [
            _handoff_file_metadata(intake.nucleotide),
            _handoff_file_metadata(intake.protein),
        ]
        declared_role: str | None = intake.nucleotide_role
        relationship = {
            "status": "co-uploaded",
            "syntactic_id_overlap_evaluated": intake.syntactic_id_overlap_evaluated,
            "syntactic_id_overlap_count": intake.syntactic_id_overlap_count,
            "syntactic_id_overlap_preview": list(intake.syntactic_id_overlap_preview),
            "claim": "Exact identifier comparison does not establish annotation or biological linkage.",
        }
    elif isinstance(intake, SequenceInspection):
        files = [_handoff_file_metadata(intake)]
        declared_role = nucleotide_role if intake.sequence_type == "nucleotide" else None
        if intake.sequence_type == "nucleotide" and declared_role not in {"assembly", "cds"}:
            raise SequencePairingError(
                "A single nucleotide handoff requires nucleotide_role='assembly' or 'cds'."
            )
        relationship = {
            "status": "single-input",
            "syntactic_id_overlap_evaluated": False,
            "syntactic_id_overlap_count": None,
            "syntactic_id_overlap_preview": [],
            "claim": "No sequence linkage, annotation, or gene-protein mapping was inferred.",
        }
    else:
        raise TypeError("intake must be a SequenceInspection or SequencePairInspection")

    payload = {
        "schema_version": "myco-optima.sequence-handoff.v1",
        "parser_version": PARSER_VERSION,
        "handoff": "external-fungal-gem-reconstruction",
        "files": files,
        "nucleotide_role": declared_role,
        "relationship": relationship,
        "capabilities": {
            "format_validated": True,
            "annotation_performed": False,
            "metabolic_reconstruction_performed": False,
            "gap_filling_performed": False,
            "supports_fba": False,
        },
        "required_output": (
            "Curated COBRA-compatible SBML with explicit finite FBC bounds and an objective"
        ),
        "next_step": (
            "Run organism-appropriate annotation and GEM reconstruction externally, curate "
            "the result, then upload its SBML model to the metabolic-model route."
        ),
        "limitations": [
            "Co-uploading files does not prove that their records correspond.",
            "No species, gene function, reaction, phenotype, or metabolic capability was inferred.",
            "Raw sequences are not included in this manifest.",
        ],
    }
    return ReconstructionHandoff(payload)


__all__ = [
    "DEFAULT_MAX_BLANK_LINES",
    "DEFAULT_MAX_COMBINED_SEQUENCE_BYTES",
    "DEFAULT_MAX_COMBINED_SEQUENCE_RESIDUES",
    "DEFAULT_MAX_IDENTIFIER_BYTES",
    "DEFAULT_MAX_PHYSICAL_LINES",
    "DEFAULT_MAX_SEQUENCE_BYTES",
    "DEFAULT_MAX_SEQUENCE_RECORDS",
    "DEFAULT_MAX_SEQUENCE_RESIDUES",
    "FastaLimits",
    "FastaRecordPreview",
    "InvalidFastaError",
    "ReconstructionHandoff",
    "SequenceInspection",
    "SequenceIntakeError",
    "SequenceLimitError",
    "SequencePairInspection",
    "SequencePairingError",
    "SequenceUploadTooLargeError",
    "UnsupportedSequenceFileError",
    "build_reconstruction_handoff",
    "inspect_fasta_upload",
    "load_fasta_upload",
    "pair_fasta_inspections",
]
