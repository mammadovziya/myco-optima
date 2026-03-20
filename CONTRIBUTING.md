# Contributing

Thanks for taking an interest in Myco Optima. The useful contributions here are
usually small and explicit: a better-tested reaction, a clearer model assumption,
or an interface change that helps a fermentation engineer understand what the
solver actually did.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
streamlit run app.py
```

Before opening a pull request, run `ruff check .` and `pytest`. If you change a
model coefficient or regulatory rule, explain the rationale, add a regression
test, and update `docs/MODEL_ASSUMPTIONS.md`. Please do not present a simulated
result as wet-lab evidence.

## Data and model contributions

Only submit model or assay data that can legally be redistributed. Include its
source, licence, organism/strain, units, and any preprocessing. Genome-scale
models should stay separate from the bundled teaching models unless their own
licence clearly permits redistribution.
