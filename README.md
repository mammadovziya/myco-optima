<p align="center">
  <img src="assets/myco-optima-wordmark.svg" alt="Myco Optima" width="520">
</p>

# Myco Optima

Myco Optima is a small fungal fermentation modelling tool built for the Pacifico Biolabs challenge at Edinburgh BioHackathon 2026.

I wanted to make metabolic modelling easier to use for fermentation engineers who do not work with COBRA models every day. The app uses Python, COBRApy and Streamlit to turn media choices into readable FBA, FVA, sensitivity and experiment planning results.

## What it does

The app includes demonstration models for four industrial fungi:

- *Aspergillus niger*
- *Aspergillus oryzae*
- *Trichoderma reesei*
- *Fusarium venenatum*

You can change carbon, nitrogen, oxygen and mineral availability, then compare predicted growth, yield and limiting nutrients. Sensitivity analysis ranks the most useful variables and builds a 15-run Box-Behnken follow-up design. A full three-level screen of four factors would contain 81 conditions, so this gives the lab a much smaller place to start.

There is also a gene-media explorer for simple morphology hypotheses. Its results come from explicit rules and should be treated as experimental leads, not confirmed biological outcomes.

## Using your own files

The Custom Model page accepts SBML, FNA and FAA files.

An SBML model can be used for FBA and selected-reaction FVA after it passes the model checks.

FNA and FAA uploads follow a different path. The app validates the FASTA file, counts records and residues, shows a short preview, and creates an inventory and reconstruction handoff. It does not annotate sequences or build a metabolic model from them.

FBA and FVA need a curated stoichiometric model with reaction bounds and an objective. If you start with FNA or FAA files, reconstruct and curate the model in a dedicated tool, export it as SBML, then upload that SBML file here.

The included fungal models are reduced teaching models. They are useful for exploring the workflow, but they are not substitutes for strain-specific reconstructions or wet-lab validation.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/mammadovziya/myco-optima.git
cd myco-optima

python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

The modelling features work without an API key. Claude is only used for optional plain-language explanations of results that have already been calculated. To enable it, set `ANTHROPIC_API_KEY` before starting the app.

## Tests

```bash
pytest
ruff check .
```

The test suite covers optimisation, FVA, sensitivity rankings, the 15-run design, morphology rules, SBML validation, FASTA uploads and export safety. GitHub Actions runs it on Python 3.11 and 3.12.

You can also run the project with Docker:

```bash
docker build -t myco-optima .
docker run --rm -p 8501:8501 myco-optima
```

More detail about the model assumptions and biological limits is in [docs/MODEL_ASSUMPTIONS.md](docs/MODEL_ASSUMPTIONS.md).

## Project context

This project was created for the [Pacifico Biolabs challenge](https://biohackathon-edinburgh-2026.devpost.com/) at Edinburgh BioHackathon 2026.

It uses [COBRApy](https://cobrapy.readthedocs.io/) for constraint-based modelling, [Streamlit](https://streamlit.io/) for the interface, and the Anthropic Python SDK for optional explanations.

Myco Optima is an independent open-source prototype and is not an official Pacifico Biolabs product. It is available under the [MIT License](LICENSE).
