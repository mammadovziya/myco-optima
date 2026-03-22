# Checking the 15-run design

This document describes software and statistical-design checks for Myco Optima's
15-run, three-factor Box-Behnken plan. It does not validate a fungal strain, a
metabolic model, a fermentation response, or the claim that 15 physical
experiments can always replace 81.

## Structural checks

The workflow starts with four factors at three levels, which defines 81 candidate
conditions. Sensitivity ranking retains three factors. The generated follow-up
contains:

- 12 unique edge conditions, each with two factors at `-1` or `+1` and one at
  `0`;
- three repeated centre conditions;
- four observations at `-1`, seven at `0`, and four at `+1` for each retained
  factor; and
- zero linear column sums and zero pairwise cross-products in coded units.

The second-order model matrix has ten columns: an intercept, three linear terms,
three pure quadratic terms, and three two-factor interactions. The 15-run matrix
has rank ten, leaving five residual degrees of freedom.

The repeated centre rows provide two *potential* pure-error degrees of freedom.
They become experimental pure-error information only if those rows are run as
independent replicates under the same conditions. Repeating a row in a CSV file
does not itself estimate laboratory error.

These properties support fitting a full quadratic in the three retained factors.
They are not a power calculation, and this validation does not claim that the
design is rotatable or that an excluded fourth factor is negligible.

## Deterministic synthetic benchmark

The benchmark evaluates a fixed, noise-free quadratic response in coded values
for the three retained factors. It fits the ten quadratic coefficients from the
15 Box-Behnken rows, then predicts all 81 points of the original four-factor
three-level grid.

For the aligned case, the synthetic response is independent of the excluded
fourth factor. The fitted model therefore recovers the known coefficients and
all 81 reference responses to floating-point precision. This is an algebraic
software check under deliberately favourable assumptions, not empirical
evidence about fermentation.

A negative control then adds a linear effect of `2.0` coded-response units for
the excluded factor. Because the 15-run design contains no information about
that factor, the fitted response surface cannot recover it. Across the 81-point
grid, the expected root-mean-square error is `2 * sqrt(2/3)`, approximately
`1.633`, and the maximum absolute error is `2.0`. The negative control makes the
scope of the exact-recovery result explicit.

## Run the checks

Run the focused tests:

```bash
pytest tests/test_doe.py tests/test_doe_validation.py
```

Or print the validation report and benchmark as strict JSON:

```bash
python -m myco_optima.doe_validation
```

The output is deterministic and contains both the exact aligned-case error and
the excluded-factor negative-control error.
