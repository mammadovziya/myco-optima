"""End-to-end Streamlit smoke tests for the connected scientific workflow."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)
    assert not app.exception
    return app


def _open_page(app: AppTest, page: str) -> AppTest:
    app.radio[0].set_value(page).run(timeout=30)
    assert not app.exception
    return app


def test_all_pages_render_with_connected_core():
    app = _app()
    for page in (
        "Media Optimizer",
        "Sensitivity & DoE",
        "Gene–Media Explorer",
        "AI Interpretation & About",
    ):
        _open_page(app, page)


def test_media_page_uses_one_consistent_core_result():
    app = _open_page(_app(), "Media Optimizer")
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
    app = _open_page(_app(), "AI Interpretation & About")
    key_widget = next(
        widget for widget in app.text_input if widget.label == "Session-only API key override"
    )

    assert key_widget.value == ""
