"""Tests for bounded nucleotide/protein FASTA intake and handoff metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from myco_optima.model_io import analyse_custom_model
from myco_optima.sequence_io import (
    FastaLimits,
    InvalidFastaError,
    ReconstructionHandoff,
    SequenceLimitError,
    SequencePairingError,
    SequenceUploadTooLargeError,
    UnsupportedSequenceFileError,
    build_reconstruction_handoff,
    inspect_fasta_upload,
    load_fasta_upload,
    pair_fasta_inspections,
)


def _fna() -> bytes:
    return b">contig_1 first fragment\r\nacgtNNry\r\nGC\r\n>contig_2\r\nAATT\r\n"


def _faa() -> bytes:
    return b">protein_1 enzyme\nMKTAYI*\n>protein_2\nBXZJUO\n"


@pytest.mark.parametrize(
    "factory",
    [
        lambda payload: payload,
        bytearray,
        memoryview,
    ],
)
def test_bytes_bytearray_and_memoryview_are_supported(factory) -> None:
    payload = _fna()
    inspection = load_fasta_upload(factory(payload), "assembly.FNA")
    assert inspection.record_count == 2
    assert inspection.sha256 == hashlib.sha256(payload).hexdigest()


def test_nucleotide_statistics_use_explicit_denominators() -> None:
    inspection = inspect_fasta_upload(_fna(), "assembly.fna")

    assert inspection.sequence_type == "nucleotide"
    assert inspection.sequence_count == 2
    assert inspection.total_bp == 14
    assert inspection.minimum_length == 4
    assert inspection.median_length == 7.0
    assert inspection.maximum_length == 10
    assert inspection.n50 == 10
    assert inspection.ambiguous_residue_count == 4
    assert inspection.ambiguous_residue_fraction == pytest.approx(4 / 14)
    assert inspection.n_percent == pytest.approx(2 / 14 * 100)
    assert inspection.gc_percent == pytest.approx(4 / 10 * 100)
    assert inspection.preview()[0]["description"] == "first fragment"
    assert inspection.preview()[0]["length"] == 10


def test_all_ambiguous_nucleotide_input_has_undefined_gc() -> None:
    inspection = load_fasta_upload(b">unknown\nNNNNRYSW\n", "unknown.fna")

    assert inspection.gc_percent is None
    assert inspection.n_percent == 50.0
    assert inspection.ambiguous_residue_fraction == 1.0
    assert any("GC percentage is undefined" in warning for warning in inspection.warnings)
    assert inspection.metadata()["gc_denominator"].startswith("A+C+G")


def test_rna_u_is_accepted_but_reported() -> None:
    inspection = load_fasta_upload(b">rna_like\nAUGC\n", "input.fna")
    assert inspection.gc_percent == 50.0
    assert any("contains U" in warning for warning in inspection.warnings)


def test_protein_terminal_stop_is_not_counted_as_an_amino_acid() -> None:
    inspection = load_fasta_upload(_faa(), "proteins.faa")

    assert inspection.sequence_type == "protein"
    assert inspection.protein_count == 2
    assert inspection.total_aa == 12
    assert inspection.minimum_length == 6
    assert inspection.maximum_length == 6
    assert inspection.terminal_stop_marker_count == 1
    assert inspection.ambiguous_residue_count == 4
    assert inspection.preview()[0]["terminal_stop_marker"] is True
    assert any("Excluded 1 terminal stop" in warning for warning in inspection.warnings)


def test_metadata_and_repr_do_not_contain_raw_sequences_or_unbounded_id_lists() -> None:
    secret_sequence = b"ACGTACGTACGTACGTACGT"
    inspection = load_fasta_upload(b">record private\n" + secret_sequence + b"\n", "x.fna")
    encoded = json.dumps(inspection.metadata(), allow_nan=False)

    assert secret_sequence.decode() not in encoded
    assert secret_sequence.decode() not in repr(inspection)
    assert "record_ids" not in inspection.metadata()
    assert inspection.metadata()["raw_sequences_retained"] is False
    assert inspection.metadata()["supports_fba"] is False


def test_preview_is_bounded_during_parsing() -> None:
    payload = b"".join(f">r{index}\nACGT\n".encode() for index in range(8))
    inspection = load_fasta_upload(
        payload,
        "many.fna",
        limits=FastaLimits(preview_records=3),
    )
    assert inspection.record_count == 8
    assert len(inspection.preview()) == 3
    assert len(inspection.preview(limit=2)) == 2


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "model.xml",
        "sequences.fa",
        "sequences.fna.gz",
        "sequences.faa.zip",
        "../sequences.fna",
        "/tmp/sequences.faa",
        r"folder\sequences.fna",
        "bad\nname.fna",
        " safe.fna",
        "safe\u202eevil.fna",
        "safe\u200bevil.fna",
        "safe\u0085evil.fna",
        "x" * 252 + ".fna",
    ],
)
def test_unsupported_or_unsafe_filenames_are_rejected(filename: str) -> None:
    with pytest.raises(UnsupportedSequenceFileError):
        load_fasta_upload(b">id\nACGT\n", filename)


def test_non_bytes_empty_invalid_utf8_and_nul_are_rejected() -> None:
    with pytest.raises(UnsupportedSequenceFileError, match="bytes-like"):
        load_fasta_upload(">id\nACGT\n", "x.fna")  # type: ignore[arg-type]
    with pytest.raises(InvalidFastaError, match="empty"):
        load_fasta_upload(b"", "x.fna")
    with pytest.raises(InvalidFastaError, match="non-ASCII|symbol"):
        load_fasta_upload(b">id\nACG\xff\n", "x.fna")
    with pytest.raises(InvalidFastaError, match="symbol"):
        load_fasta_upload(b">id\nACG\x00T\n", "x.fna")


@pytest.mark.parametrize(
    "payload",
    [
        b">\tid\nACGT\n",
        b">id\tcomment\nACGT\n",
        ">\u200bid\nACGT\n".encode(),
        ">id\u202ecomment\nACGT\n".encode(),
    ],
)
def test_header_controls_are_rejected_before_normalization(payload: bytes) -> None:
    with pytest.raises(InvalidFastaError, match="control"):
        load_fasta_upload(payload, "x.fna")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"ACGT\n", "before the first header"),
        (b">\nACGT\n", "no identifier"),
        (b">one\n>two\nACGT\n", "has no sequence"),
        (b">same first\nACGT\n>same second\nACGT\n", "Duplicate"),
        (b"\n\n", "no sequence records"),
    ],
)
def test_structurally_invalid_fasta_is_rejected(payload: bytes, message: str) -> None:
    with pytest.raises(InvalidFastaError, match=message):
        load_fasta_upload(payload, "bad.fna")


@pytest.mark.parametrize("symbol", [b"E", b"1", b"-", b" ", b"\t"])
def test_invalid_nucleotide_symbols_are_rejected(symbol: bytes) -> None:
    with pytest.raises(InvalidFastaError, match="nucleotide symbol"):
        load_fasta_upload(b">id\nAC" + symbol + b"GT\n", "bad.fna")


@pytest.mark.parametrize("symbol", [b"1", b"-", b" ", b"\t"])
def test_invalid_protein_symbols_are_rejected(symbol: bytes) -> None:
    with pytest.raises(InvalidFastaError, match="amino-acid symbol"):
        load_fasta_upload(b">id\nMK" + symbol + b"TA\n", "bad.faa")


@pytest.mark.parametrize(
    "payload",
    [
        b">id\nMK*TA\n",
        b">id\nMK**\n",
        b">id\nMK*\nTA\n",
    ],
)
def test_internal_repeated_or_continued_protein_stops_are_rejected(payload: bytes) -> None:
    with pytest.raises(InvalidFastaError, match="stop marker|continues"):
        load_fasta_upload(payload, "bad.faa")


def test_exact_byte_limit_is_accepted_and_cap_plus_one_is_rejected() -> None:
    payload = b">id\nACGT\n"
    exact = FastaLimits(max_bytes=len(payload))
    assert load_fasta_upload(payload, "x.fna", limits=exact).total_bp == 4

    with pytest.raises(SequenceUploadTooLargeError):
        load_fasta_upload(payload + b"\n", "x.fna", limits=exact)


def test_record_residue_header_identifier_and_line_limits_are_enforced() -> None:
    with pytest.raises(SequenceLimitError, match="record limit"):
        load_fasta_upload(
            b">a\nA\n>b\nA\n",
            "x.fna",
            limits=FastaLimits(max_records=1),
        )
    with pytest.raises(SequenceLimitError, match="residue limit"):
        load_fasta_upload(
            b">a\nAAAA\n>b\nAAAA\n",
            "x.fna",
            limits=FastaLimits(max_total_residues=7),
        )
    with pytest.raises(SequenceLimitError, match="per-record"):
        load_fasta_upload(
            b">a\nAAAA\n",
            "x.fna",
            limits=FastaLimits(max_record_residues=3),
        )
    with pytest.raises(SequenceLimitError, match="header"):
        load_fasta_upload(
            b">abcd\nA\n",
            "x.fna",
            limits=FastaLimits(max_header_length=3),
        )
    with pytest.raises(SequenceLimitError, match="identifier"):
        load_fasta_upload(
            b">abcd\nA\n",
            "x.fna",
            limits=FastaLimits(max_identifier_length=3),
        )
    with pytest.raises(SequenceLimitError, match="sequence line"):
        load_fasta_upload(
            b">a\nAAAA\n",
            "x.fna",
            limits=FastaLimits(max_sequence_line_length=3),
        )
    with pytest.raises(SequenceLimitError, match="physical-line"):
        load_fasta_upload(
            b">a\nA\nA\nA\n",
            "x.fna",
            limits=FastaLimits(max_physical_lines=3),
        )
    with pytest.raises(SequenceLimitError, match="blank-line"):
        load_fasta_upload(
            b"\n\n\n>a\nA\n",
            "x.fna",
            limits=FastaLimits(max_blank_lines=2),
        )
    with pytest.raises(SequenceLimitError, match="retained-text"):
        load_fasta_upload(
            b">abcd\nA\n>efgh\nA\n",
            "x.fna",
            limits=FastaLimits(max_identifier_bytes=7),
        )


def test_long_unwrapped_contig_is_stream_scanned() -> None:
    payload = b">long\n" + b"ACGT" * 50_000 + b"\n"
    inspection = load_fasta_upload(payload, "long.fna")
    assert inspection.total_bp == 200_000
    assert inspection.gc_percent == 50.0


def test_pairing_is_explicit_and_id_overlap_is_case_sensitive_cds_only() -> None:
    nucleotide = load_fasta_upload(b">same\nATG\n>Case\nATG\n", "cds.fna")
    protein = load_fasta_upload(b">same\nM\n>case\nM\n", "proteins.faa")

    assembly_pair = pair_fasta_inspections(nucleotide, protein, nucleotide_role="assembly")
    assert assembly_pair.syntactic_id_overlap_evaluated is False
    assert assembly_pair.syntactic_id_overlap_count == 0

    cds_pair = pair_fasta_inspections(nucleotide, protein, nucleotide_role="cds")
    assert cds_pair.syntactic_id_overlap_evaluated is True
    assert cds_pair.syntactic_id_overlap_count == 1
    assert cds_pair.syntactic_id_overlap_preview == ("same",)
    assert cds_pair.metadata()["biological_linkage_inferred"] is False


def test_invalid_pair_order_role_and_combined_caps_are_rejected() -> None:
    nucleotide = load_fasta_upload(b">n\nATG\n", "n.fna")
    protein = load_fasta_upload(b">p\nM\n", "p.faa")

    with pytest.raises(SequencePairingError, match="first"):
        pair_fasta_inspections(protein, nucleotide)  # type: ignore[arg-type]
    with pytest.raises(SequencePairingError, match="second"):
        pair_fasta_inspections(nucleotide, nucleotide)
    with pytest.raises(SequencePairingError, match="nucleotide_role"):
        pair_fasta_inspections(nucleotide, protein, nucleotide_role="unknown")  # type: ignore[arg-type]
    with pytest.raises(SequencePairingError, match="byte"):
        pair_fasta_inspections(
            nucleotide,
            protein,
            max_combined_bytes=nucleotide.size_bytes + protein.size_bytes - 1,
        )
    with pytest.raises(SequencePairingError, match="residue"):
        pair_fasta_inspections(
            nucleotide,
            protein,
            max_combined_residues=nucleotide.total_residues + protein.total_residues - 1,
        )


def test_pair_handoff_is_strict_json_and_explicitly_not_fba_ready() -> None:
    nucleotide = load_fasta_upload(b">gene\nATGCGT\n", "cds.fna")
    protein = load_fasta_upload(b">gene\nMR\n", "proteins.faa")
    pair = pair_fasta_inspections(nucleotide, protein, nucleotide_role="cds")
    manifest = build_reconstruction_handoff(pair).to_dict()
    encoded = json.dumps(manifest, allow_nan=False)

    assert manifest["schema_version"] == "myco-optima.sequence-handoff.v1"
    assert manifest["capabilities"]["annotation_performed"] is False
    assert manifest["capabilities"]["metabolic_reconstruction_performed"] is False
    assert manifest["capabilities"]["supports_fba"] is False
    assert manifest["nucleotide_role"] == "cds"
    assert manifest["relationship"]["syntactic_id_overlap_count"] == 1
    assert "ATGCGT" not in encoded
    assert "api_key" not in encoded.lower()


def test_single_input_handoff_omits_irrelevant_nucleotide_role() -> None:
    protein = load_fasta_upload(b">p\nMKT\n", "p.faa")
    protein_manifest = build_reconstruction_handoff(protein, nucleotide_role="assembly").to_dict()
    assert protein_manifest["nucleotide_role"] is None

    nucleotide = load_fasta_upload(b">n\nATG\n", "n.fna")
    with pytest.raises(SequencePairingError, match="requires"):
        build_reconstruction_handoff(nucleotide)
    assert (
        build_reconstruction_handoff(nucleotide, nucleotide_role="assembly").to_dict()[
            "nucleotide_role"
        ]
        == "assembly"
    )


def test_sequence_inspection_cannot_enter_custom_model_solver_path() -> None:
    inspection = load_fasta_upload(b">n\nATG\n", "n.fna")
    with pytest.raises(TypeError, match="ModelInspection"):
        analyse_custom_model(inspection, "objective")  # type: ignore[arg-type]


def test_sbml_renamed_as_fasta_is_rejected_as_sequence_content() -> None:
    payload = b'<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"></sbml>'
    with pytest.raises(InvalidFastaError):
        load_fasta_upload(payload, "model.fna")


def test_error_messages_are_bounded_and_do_not_echo_huge_headers() -> None:
    tail = "DO_NOT_ECHO_" * 200
    payload = (">id " + tail + "\nACGT\n").encode()
    with pytest.raises(SequenceLimitError) as caught:
        load_fasta_upload(
            payload,
            "x.fna",
            limits=FastaLimits(max_header_length=32),
        )
    assert len(str(caught.value)) < 200
    assert "DO_NOT_ECHO" not in str(caught.value)


def test_limits_must_be_a_valid_limits_object() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FastaLimits(max_records=0)
    with pytest.raises(TypeError, match="FastaLimits"):
        load_fasta_upload(b">id\nA\n", "x.fna", limits={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="safe default"):
        FastaLimits(max_blank_lines=10_001)


def test_public_inspection_invariants_prevent_forged_handoff_claims() -> None:
    inspection = load_fasta_upload(b">id\nACGT\n", "x.fna")
    with pytest.raises(ValueError, match="ambiguity fraction"):
        replace(inspection, ambiguous_residue_fraction=float("nan"))
    with pytest.raises(ValueError, match="counts and byte size"):
        replace(inspection, total_residues=-1)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(inspection, sha256="not-a-digest")
    with pytest.raises(ValueError, match="GC percentage"):
        replace(
            inspection,
            preview_records=(replace(inspection.preview_records[0], gc_percent=float("nan")),),
        )

    protein = load_fasta_upload(b">id\nM\n", "x.faa")
    pair = pair_fasta_inspections(inspection, protein, nucleotide_role="cds")
    with pytest.raises(ValueError, match="overlap_count"):
        replace(pair, syntactic_id_overlap_count=float("nan"))


def test_reconstruction_handoff_rejects_non_strict_json_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ReconstructionHandoff({"value": float("nan")})
