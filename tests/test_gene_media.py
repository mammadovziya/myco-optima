"""Tests for transparent gene–media morphology rules."""

from __future__ import annotations

import pytest

from myco_optima.gene_media import predict_morphology, supported_genes


def test_aspergillus_niger_raca_loss_is_traced_as_dispersed() -> None:
    result = predict_morphology("aspergillus_niger", {"glucose": 10}, {"racA": "KO"})

    assert result.predicted_morphology == "Dispersed hyphae"
    assert result.confidence == "moderate"
    assert any(item.rule_id == "an_raca_loss" and item.applied for item in result.interaction_trace)
    assert result.support_scores["Dispersed hyphae"] > result.support_scores["Compact pellets"]


def test_trichoderma_rac1_lactose_interaction_is_medium_specific() -> None:
    lactose = predict_morphology("T. reesei", {"lactose": 10}, {"rac1": "knockout"})
    glucose = predict_morphology("T. reesei", {"glucose": 10}, {"rac1": "knockout"})

    lactose_rule = next(
        item for item in lactose.interaction_trace if item.rule_id == "tr_rac1_lactose_interaction"
    )
    glucose_rule = next(
        item for item in glucose.interaction_trace if item.rule_id == "tr_rac1_lactose_interaction"
    )
    assert lactose_rule.applied is True
    assert glucose_rule.applied is False
    assert lactose.latent_scores["secretion"] == glucose.latent_scores["secretion"] + 2


def test_aspergillus_oryzae_combined_aggregation_pathway_loss_is_dispersed() -> None:
    states = {gene: "deleted" for gene in ("agsA", "agsB", "agsC", "sphZ", "ugeZ")}
    result = predict_morphology("Aspergillus oryzae", gene_states=states)

    assert result.predicted_morphology == "Dispersed hyphae"
    assert any(
        item.rule_id == "ao_gag_pathway_combined_loss" and item.applied
        for item in result.interaction_trace
    )


def test_unknown_fusarium_gene_is_not_given_an_invented_effect() -> None:
    result = predict_morphology("fusarium_venenatum", gene_states={"invented1": "KO"})

    assert result.confidence == "low"
    assert result.insufficient_evidence
    assert not [
        item for item in result.interaction_trace if item.gene == "invented1" and item.applied
    ]
    assert any("No species-specific" in warning for warning in result.warnings)


def test_low_oxygen_raises_stress_flag_without_fake_gene_rule() -> None:
    result = predict_morphology("aspergillus_niger", {"oxygen": 3}, {})

    assert result.latent_scores["stress"] == 1.0
    process_rule = next(
        item for item in result.interaction_trace if item.rule_id == "process_low_oxygen"
    )
    assert process_rule.effects == {"stress": 1.0}
    assert process_rule.gene == "(process)"


def test_scores_are_explicitly_not_probabilities() -> None:
    result = predict_morphology("aspergillus_niger")
    payload = result.to_dict()

    assert "probabilities" not in payload
    assert isinstance(result.confidence, str)
    assert sum(result.support_scores.values()) != pytest.approx(1.0)
    assert any("not probabilities" in warning for warning in result.warnings)


def test_supported_genes_and_state_validation() -> None:
    assert supported_genes("A. niger") == ("raca", "arfa")
    assert supported_genes("F. venenatum") == ()
    with pytest.raises(ValueError, match="Unknown gene state"):
        predict_morphology("aspergillus_niger", gene_states={"racA": "maybe"})
