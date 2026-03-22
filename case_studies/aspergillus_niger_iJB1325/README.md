# Published model case study: *Aspergillus niger*

This case uses the published iJB1325 model for *A. niger* ATCC 1015. It is a
published genome-scale reconstruction kept separate from Myco Optima's four
small teaching models.

## Result

The source contains 2,320 reactions, 1,818 metabolites, 1,325 genes and 471
embedded consistency tests.

| Check | Published | Reproduced here |
|---|---:|---:|
| Passing tests | 373 | 373 |
| Failing tests | 98 | 98 |
| Pass rate | 79.2% | 79.2% |

The suite covers 392 growth-media cases, 73 gene-deletion phenotypes and six
system checks. The runner follows the original GEMEditor test logic, including
its boundary setup and 0.01 flux margin.

The runner also checks the model's default glucose scenario. FBA gives a biomass
objective of `0.939855`. At 95% of optimum, FVA gives a biomass range of
`0.892862` to `0.939855`. These are source-model flux values, not measured growth
rates or fermentation set-points.

Our COBRApy runner reproduces the paper's embedded test result. This is not an
independent holdout test or a measure of biological accuracy.

## Reproduce it

From the repository root:

```bash
python -m myco_optima.case_study
```

The command verifies the source checksum, runs FBA and FVA, executes all 471
cases, and prints strict JSON. The saved output is in [results.json](results.json).

Model provenance, licensing and the compatibility step are in
[SOURCE.md](SOURCE.md).
