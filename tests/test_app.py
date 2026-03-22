"""End-to-end Streamlit smoke tests for the connected scientific workflow."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from cobra import Metabolite, Model, Reaction
from cobra.io import write_sbml_model
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)
    assert not app.exception
    return app


def _open_page(app: AppTest, page: str) -> AppTest:
    workspace = next(widget for widget in app.radio if widget.label == "Workspace")
    workspace.set_value(page).run(timeout=30)
    assert not app.exception
    return app


def _open_sequence_page(app: AppTest | None = None) -> AppTest:
    app = _open_page(app or _app(), "Custom Model")
    route = next(widget for widget in app.radio if widget.label == "Custom input type")
    route.set_value("Nucleotide / protein FASTA").run(timeout=30)
    assert not app.exception
    return app


def _uploader(app: AppTest, label: str):
    return next(widget for widget in app.file_uploader if widget.label == label)


def _fna_bytes(*, second_record: bool = True) -> bytes:
    records = [">contig_1 assembly fragment\nACGTACGTNNACGT\n"]
    if second_record:
        records.append(">contig_2\nGGCCATTA\n")
    return "".join(records).encode("utf-8")


def _faa_bytes() -> bytes:
    return b">protein_1 enzyme candidate\nMKTAYIAKQRQISFVKSHFSRQ*\n>protein_2\nMNNNKL\n"


def _cds_fna_bytes() -> bytes:
    return b">protein_1 coding sequence\nATGAAAACTGCTTATATTGCTAAA\n>cds_only\nATGNNNTAA\n"


def _custom_sbml_bytes(*, source_capacity: float = 10) -> bytes:
    model = Model("uploaded_app_test", name="Uploaded AppTest model")
    precursor = Metabolite("precursor_c", compartment="c")
    biomass = Metabolite("biomass_c", compartment="c")
    source = Reaction("EX_precursor", lower_bound=0, upper_bound=source_capacity)
    source.add_metabolites({precursor: 1})
    conversion = Reaction("BIOMASS", name="Biomass objective")
    conversion.add_metabolites({precursor: -1, biomass: 1})
    sink = Reaction("DM_biomass", lower_bound=0, upper_bound=1000)
    sink.add_metabolites({biomass: -1})
    model.add_reactions([source, conversion, sink])
    model.objective = conversion
    handle = StringIO()
    write_sbml_model(model, handle, f_replace={})
    return handle.getvalue().encode("utf-8")


def test_all_pages_render_with_connected_core():
    app = _app()
    for page in (
        "Custom Model",
        "Media Optimiser",
        "Sensitivity & DoE",
        "Gene–Media Explorer",
        "Interpretation & Methods",
    ):
        _open_page(app, page)


def test_custom_sbml_upload_reaches_fba_and_fva_results():
    app = _open_page(_app(), "Custom Model")
    assert app.file_uploader[0].proto.max_upload_size_mb == 5
    app.file_uploader[0].upload("uploaded-model.xml", _custom_sbml_bytes(), "application/xml").run(
        timeout=30
    )

    assert not app.exception
    inventory = {metric.label: metric.value for metric in app.metric}
    assert inventory["Reactions"] == "3"
    assert inventory["Metabolites"] == "2"

    run_button = next(
        button for button in app.button if button.label == "Run custom-model analysis"
    )
    run_button.click().run(timeout=30)

    assert not app.exception
    results = {metric.label: metric.value for metric in app.metric}
    assert results["Solver status"] == "Optimal"
    assert float(results["Objective value"]) == 10.0
    assert int(results["FVA reactions"]) >= 1


def test_custom_upload_change_clears_stale_results_and_exports():
    original = _custom_sbml_bytes(source_capacity=10)
    replacement = _custom_sbml_bytes(source_capacity=7)
    app = _open_page(_app(), "Custom Model")
    app.file_uploader[0].set_value(("same-name.xml", original, "application/xml")).run(timeout=30)
    next(
        button for button in app.button if button.label == "Run custom-model analysis"
    ).click().run(timeout=30)
    assert len(app.download_button) == 2

    app.file_uploader[0].set_value(("same-name.xml", replacement, "application/xml")).run(
        timeout=30
    )
    assert not app.exception
    assert len(app.download_button) == 0
    assert "custom_analysis" not in app.session_state.filtered_state

    next(
        button for button in app.button if button.label == "Run custom-model analysis"
    ).click().run(timeout=30)
    results = {metric.label: metric.value for metric in app.metric}
    assert float(results["Objective value"]) == 7.0


def test_bad_or_cleared_upload_cannot_resurrect_previous_analysis():
    payload = _custom_sbml_bytes()
    app = _open_page(_app(), "Custom Model")
    app.file_uploader[0].set_value(("valid.xml", payload, "application/xml")).run(timeout=30)
    next(
        button for button in app.button if button.label == "Run custom-model analysis"
    ).click().run(timeout=30)
    assert len(app.download_button) == 2

    app.file_uploader[0].set_value(("broken.xml", b"<sbml>", "application/xml")).run(timeout=30)
    assert len(app.download_button) == 0
    assert "custom_analysis" not in app.session_state.filtered_state

    app.file_uploader[0].set_value(("valid.xml", payload, "application/xml")).run(timeout=30)
    assert len(app.download_button) == 0
    assert "custom_analysis" not in app.session_state.filtered_state

    app.file_uploader[0].clear().run(timeout=30)
    assert "custom_inspection" not in app.session_state.filtered_state
    assert "custom_analysis" not in app.session_state.filtered_state


def test_oversized_sbml_is_rejected_before_model_state_is_created():
    app = _open_page(_app(), "Custom Model")
    app.file_uploader[0].set_value(("oversized.xml", b"x" * 5_000_001, "application/xml")).run(
        timeout=30
    )

    assert not app.exception
    assert "custom_inspection" not in app.session_state.filtered_state
    assert "custom_analysis" not in app.session_state.filtered_state
    assert len(app.download_button) == 0
    assert any("exceeds the 5 MB" in block.value for block in app.error)


def test_failed_custom_rerun_removes_previous_downloads():
    app = _open_page(_app(), "Custom Model")
    app.file_uploader[0].set_value(("valid.xml", _custom_sbml_bytes(), "application/xml")).run(
        timeout=30
    )
    run_button = next(
        button for button in app.button if button.label == "Run custom-model analysis"
    )
    run_button.click().run(timeout=30)
    assert len(app.download_button) == 2

    fva_picker = next(widget for widget in app.multiselect if widget.label == "FVA reactions")
    fva_picker.set_value([]).run(timeout=30)
    next(
        button for button in app.button if button.label == "Run custom-model analysis"
    ).click().run(timeout=30)

    assert len(app.download_button) == 0
    assert "custom_analysis" not in app.session_state.filtered_state


def test_custom_model_state_is_not_shared_between_app_sessions():
    first = _open_page(_app(), "Custom Model")
    first.file_uploader[0].set_value(("valid.xml", _custom_sbml_bytes(), "application/xml")).run(
        timeout=30
    )
    assert "custom_inspection" in first.session_state.filtered_state

    second = _open_page(_app(), "Custom Model")
    assert second.file_uploader[0].value is None
    assert "custom_inspection" not in second.session_state.filtered_state
    assert "custom_analysis" not in second.session_state.filtered_state


def test_fna_upload_builds_bounded_inventory_without_solver_controls():
    app = _open_sequence_page()
    assert list(_uploader(app, "Nucleotide FASTA (.fna)").proto.type) == [".fna"]
    assert list(_uploader(app, "Protein FASTA (.faa)").proto.type) == [".faa"]
    assert _uploader(app, "Nucleotide FASTA (.fna)").proto.max_upload_size_mb == 100
    assert _uploader(app, "Protein FASTA (.faa)").proto.max_upload_size_mb == 100
    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "assembly.fna", _fna_bytes(), "text/plain"
    ).run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Nucleotide records"] == "2"
    assert metrics["Total bases"] == "22"
    assert len(app.download_button) == 2
    assert all(button.label != "Run custom-model analysis" for button in app.button)
    assert "Solver status" not in metrics
    assert all(widget.label != "Objective reaction" for widget in app.selectbox)
    assert all(widget.label != "FVA reactions" for widget in app.multiselect)


def test_all_n_fna_keeps_nucleotide_labels_when_gc_is_undefined():
    app = _open_sequence_page()
    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "ambiguous.fna", b">unknown\nNNNN\n", "text/plain"
    ).run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Total bases"] == "4"
    assert "Amino acids" not in metrics
    assert any("GC: n/a" in block.value for block in app.caption)


def test_faa_upload_builds_protein_inventory_without_flux_results():
    app = _open_sequence_page()
    _uploader(app, "Protein FASTA (.faa)").upload("proteins.faa", _faa_bytes(), "text/plain").run(
        timeout=30
    )

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Protein records"] == "2"
    assert metrics["Amino acids"] == "28"
    assert any("Terminal stop markers: 1" in block.value for block in app.caption)
    assert len(app.download_button) == 2
    assert "Objective value" not in metrics
    assert all(button.label != "Run custom-model analysis" for button in app.button)


def test_co_uploaded_fna_faa_pair_is_not_claimed_as_linked():
    app = _open_sequence_page()
    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "assembly.fna", _fna_bytes(), "text/plain"
    ).run(timeout=30)
    _uploader(app, "Protein FASTA (.faa)").upload("proteins.faa", _faa_bytes(), "text/plain").run(
        timeout=30
    )

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Nucleotide records"] == "2"
    assert metrics["Protein records"] == "2"
    assert any("Co-uploaded FNA + FAA" in block.value for block in app.info)
    assert not any("Exact record-ID overlap" in block.value for block in app.caption)
    assert len(app.download_button) == 2


def test_cds_declaration_allows_literal_id_overlap_report_only():
    app = _open_sequence_page()
    declaration = next(widget for widget in app.radio if widget.label == "FNA content declaration")
    declaration.set_value("Coding sequences (CDS)").run(timeout=30)
    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "coding.fna", _cds_fna_bytes(), "text/plain"
    ).run(timeout=30)
    _uploader(app, "Protein FASTA (.faa)").upload("proteins.faa", _faa_bytes(), "text/plain").run(
        timeout=30
    )

    assert not app.exception
    assert any("Exact record-ID overlap: 1" in block.value for block in app.caption)
    assert any("does not establish" in block.value for block in app.caption)


def test_malformed_sequence_replacement_and_clear_remove_inventory_state():
    app = _open_sequence_page()
    fna = _uploader(app, "Nucleotide FASTA (.fna)")
    fna.set_value(("assembly.fna", _fna_bytes(), "text/plain")).run(timeout=30)
    assert "sequence_inspections" in app.session_state.filtered_state
    assert len(app.download_button) == 2

    _uploader(app, "Nucleotide FASTA (.fna)").set_value(
        ("assembly.fna", b"not a FASTA file", "text/plain")
    ).run(timeout=30)
    assert not app.exception
    assert "sequence_inspections" not in app.session_state.filtered_state
    assert len(app.download_button) == 0

    _uploader(app, "Nucleotide FASTA (.fna)").clear().run(timeout=30)
    assert "sequence_inspections" not in app.session_state.filtered_state
    assert "sequence_active_upload_identity" not in app.session_state.filtered_state


def test_clearing_one_co_upload_rebuilds_only_the_remaining_inventory():
    app = _open_sequence_page()
    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "assembly.fna", _fna_bytes(), "text/plain"
    ).run(timeout=30)
    _uploader(app, "Protein FASTA (.faa)").upload("proteins.faa", _faa_bytes(), "text/plain").run(
        timeout=30
    )

    _uploader(app, "Nucleotide FASTA (.fna)").clear().run(timeout=30)
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Nucleotide records" not in metrics
    assert metrics["Protein records"] == "2"
    assert set(app.session_state.filtered_state["sequence_inspections"]) == {"faa"}


def test_sequence_upload_state_is_not_shared_between_app_sessions():
    first = _open_sequence_page()
    _uploader(first, "Nucleotide FASTA (.fna)").upload(
        "assembly.fna", _fna_bytes(), "text/plain"
    ).run(timeout=30)
    assert "sequence_inspections" in first.session_state.filtered_state

    second = _open_sequence_page()
    assert _uploader(second, "Nucleotide FASTA (.fna)").value is None
    assert _uploader(second, "Protein FASTA (.faa)").value is None
    assert "sequence_inspections" not in second.session_state.filtered_state


def test_forget_sequence_uploads_removes_widget_values_and_inspections():
    app = _open_sequence_page()
    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "assembly.fna", _fna_bytes(), "text/plain"
    ).run(timeout=30)
    assert "sequence_inspections" in app.session_state.filtered_state

    next(button for button in app.button if button.label == "Forget sequence uploads").click().run(
        timeout=30
    )

    assert not app.exception
    assert _uploader(app, "Nucleotide FASTA (.fna)").value is None
    assert _uploader(app, "Protein FASTA (.faa)").value is None
    assert "sequence_inspections" not in app.session_state.filtered_state


def test_switching_custom_input_routes_clears_the_hidden_analysis_namespace():
    app = _open_page(_app(), "Custom Model")
    app.file_uploader[0].set_value(("valid.xml", _custom_sbml_bytes(), "application/xml")).run(
        timeout=30
    )
    assert "custom_inspection" in app.session_state.filtered_state

    route = next(widget for widget in app.radio if widget.label == "Custom input type")
    route.set_value("Nucleotide / protein FASTA").run(timeout=30)
    assert "custom_inspection" not in app.session_state.filtered_state
    assert "custom_analysis" not in app.session_state.filtered_state

    _uploader(app, "Nucleotide FASTA (.fna)").upload(
        "assembly.fna", _fna_bytes(), "text/plain"
    ).run(timeout=30)
    assert "sequence_inspections" in app.session_state.filtered_state

    route = next(widget for widget in app.radio if widget.label == "Custom input type")
    route.set_value("Metabolic model (SBML)").run(timeout=30)
    assert "sequence_inspections" not in app.session_state.filtered_state


def test_media_page_uses_one_consistent_core_result():
    app = _open_page(_app(), "Media Optimiser")
    metrics = {metric.label: metric.value for metric in app.metric}
    composition = app.dataframe[0].value
    alternatives = app.dataframe[1].value

    displayed_growth = float(metrics["Predicted growth flux"].split()[0])
    displayed_cost = float(metrics["Estimated medium cost index"])
    assert displayed_growth == round(float(alternatives.iloc[0]["Growth flux"]), 3)
    assert displayed_cost == round(float(composition["Estimated cost"].sum()), 2)
    assert displayed_cost == round(float(alternatives.iloc[0]["Estimated cost"]), 2)


def test_sensitivity_page_emits_exact_box_behnken_design():
    app = _open_page(_app(), "Sensitivity & DoE")
    design = app.dataframe[0].value

    assert len(design) == 15
    assert design["Design point"].value_counts().to_dict() == {
        "interaction edge": 12,
        "centre replicate": 3,
    }
    assert "Phosphate availability" in design


def test_supported_gene_rule_reaches_the_ui():
    app = _open_page(_app(), "Gene–Media Explorer")
    rac_a = next(widget for widget in app.selectbox if widget.label.startswith("racA ·"))
    rac_a.set_value("Knock-down").run(timeout=30)

    assert not app.exception
    assert any("Dispersed hyphae" in block.value for block in app.markdown)
    interactions = app.dataframe[0].value
    applied = interactions.loc[interactions["Applied"]]
    assert "raca" in set(applied["Gene"])


def test_server_key_is_not_inserted_into_browser_widget(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-secret-must-stay-server-side")
    app = _open_page(_app(), "Interpretation & Methods")
    key_widget = next(
        widget for widget in app.text_input if widget.label == "Session-only API key override"
    )

    assert key_widget.value == ""
