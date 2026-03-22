# Model assumptions and scientific boundaries

Myco Optima is useful only if it is honest about what its numbers mean. The
bundled networks are **reduced-order stoichiometric surrogates** built to make
FBA, FVA, media sensitivity and experimental prioritisation inspectable in a
hackathon-sized application. They are not genome-scale reconstructions, they are
not strain-specific, and their predicted growth flux is not a measured growth
rate.

If you want to use this for real process decisions, replace the bundled network
with a curated, strain-matched GEM; fit exchange bounds to measured uptake data;
and validate every recommendation in the lab.

## What the COBRA model represents

Each organism uses the same small network skeleton with organism-specific
biomass and substrate-efficiency coefficients. The network contains:

- exchanges and irreversible transport for carbon, nitrogen, oxygen, phosphate,
  sulfate and trace elements;
- carbon-, nitrogen-, phosphorus- and sulfur-equivalent pools;
- a respiratory reaction that couples carbon oxidation to an energy pool;
- organism-specific pseudo-biomass and optional product demand reactions; and
- finite bounds throughout, so no nutrient can appear from nowhere.

Element-equivalent pools and biomass reactions are deliberately marked as
pseudo-reactions. They keep the optimisation understandable, but should not be
mistaken for chemically complete mass balance.

The medium values shown to the solver are **maximum uptake bounds in model flux
units**. They are not g/L concentrations. Connecting a flask concentration to an
exchange flux requires kinetics or experimental uptake measurements that this
demo does not have. Temperature and starting pH are context for the morphology
rules; they do not change reaction rates in FBA.

## The four bundled organism profiles

| Organism | Demonstration context | Main carbon options | Important boundary |
|---|---|---|---|
| *Aspergillus niger* | organic acids and secreted enzymes | glucose, xylose, glycerol, maltose | not strain-specific |
| *Aspergillus oryzae* | food fermentation and enzymes | glucose, maltose, sucrose | no food-safety conclusion |
| *Trichoderma reesei* | cellulase and biorefining | glucose, xylose, glycerol | the bundled COBRA surrogate does not encode cellulase induction |
| *Fusarium venenatum* | mycoprotein and biomass | glucose, xylose, maltose | sparse gene-rule evidence; media heuristics only |

## FBA, FVA and sensitivity

FBA maximises the selected pseudo-objective at steady state. FVA then reports the
minimum and maximum feasible flux for named reactions while retaining at least
95% of the optimum. COBRApy notes that FVA ranges are calculated reaction by
reaction and do not form one simultaneous flux vector; the interface therefore
shows them as ranges, not a second solution. See the
[COBRApy flux-analysis reference](https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html).

Sensitivity analysis perturbs one open medium bound at a time around the current
baseline. It reports local elasticity together with the lower and upper biomass
responses used to calculate it. This is useful for ranking controllable factors,
but it does not capture all nonlinear or high-order interactions.

## Uploaded SBML models

The custom workspace is intentionally separate from the four bundled models. It
preserves uploaded identifiers and explicit FBC bounds, lets the user choose one
reaction objective, and runs FBA plus optional FVA on a copy. No bundled medium,
cost, organism, sensitivity or gene-rule metadata is attached to the upload.

Before COBRApy parses a file, the loader rejects external entity declarations,
missing or dangling flux-bound parameters, non-constant/non-finite bounds,
inverted bounds, missing objectives and oversized networks. That preflight is
important because a permissive SBML reader may otherwise replace an omitted
bound with a large default and silently change the feasible space. The interface
therefore accepts only explicit, auditable constraints.

Uploaded files are held in the current Streamlit session and are not forwarded
to Anthropic. FVA is limited to a user-selected set of at most 50 reactions and
uses one process. The solver is given a 30-second timeout when its interface
supports one. These are resource guardrails, not proof that every accepted model
is biologically correct or numerically well conditioned.

## Uploaded nucleotide and protein FASTA

The sequence-intake route accepts one nucleotide FASTA (`.fna`), one protein
FASTA (`.faa`), or a user-declared co-upload, with a 100 MB limit per file and a
150 MB combined accepted-processing limit. A deployment still needs enough
memory and an appropriate concurrency policy because two files exist in the
Streamlit session before their combined size can be checked. Validation is
format-specific: nucleotide records are checked as nucleotide sequences and
protein records as amino-acid sequences. The app reports only properties that
are present in the files themselves, such as record count, total residues,
length range, GC content for nucleotide input, and bounded record previews.

These uploads are held in the current Streamlit session and are not sent to
Anthropic. Replacing or clearing either member of a pair invalidates the previous
inventory. Downloadable CSV and JSON handoffs include filenames, content hashes
and inventory metadata so a downstream reconstruction can be traced back to the
exact sequence inputs.

Nucleotide GC percentage uses only `A+C+G+T+U` as its denominator; ambiguity
codes are excluded, so an all-ambiguous record has no GC percentage. A single
terminal `*` is accepted in a protein record as a stop marker and excluded from
the amino-acid length. Internal or repeated stop markers are rejected. Sequence
lines may be wrapped or unwrapped within the documented file and record limits.
The parser also caps physical lines, blank lines, record identifiers and total
retained identifier bytes so a small, pathological FASTA cannot consume an
unbounded amount of CPU or session memory.

For the optional CDS/FAA comparison, a record identifier is the first
whitespace-delimited token after `>` and comparison is case-sensitive. Duplicate
identifiers are rejected during intake. Literal overlap is reported only when the
FNA is declared as coding sequences; it is a reproducibility check, not evidence
that a nucleotide record encodes the similarly named protein.

Sequence intake is **not** genome-scale metabolic-model reconstruction. An FNA
or FAA file has no reaction stoichiometry, exchange bounds, gene–protein–reaction
rules or optimisation objective, so Myco Optima does not expose FBA/FVA controls
for it and does not infer fluxes. The next step is an external annotation and
fungal GEM-reconstruction workflow, followed by biological curation and gap
review. Only its resulting, explicitly bounded SBML model belongs in the custom
solver route. Pairing FNA and FAA improves the handoff context but does not close
that modelling gap by itself.

## How “about 80 runs to 15” is calculated

Four factors at three levels produce 3⁴ = 81 candidate conditions. Myco Optima
ranks those factors by absolute model sensitivity, retains the top three, and
creates a 15-run Box–Behnken follow-up design: 12 interaction-edge conditions and
three centre replicates. In other words, the tool narrows an 81-condition search
space to a 15-run first follow-up plan (an 81.5% reduction).

That is an experimental-prioritisation claim, not evidence that 66 biological
experiments are universally unnecessary. Replication, randomisation, blocks,
controls and later validation still belong in the real protocol.

## Gene–media morphology rules

The gene–media module is a deterministic evidence table, kept separate from the
flux solver. It produces qualitative tendencies and a contribution trace; it
does not output a probability or let the language model invent a gene effect.
Unknown or weakly supported combinations return “insufficient evidence.”

The strongest bundled rules are grounded in the following studies:

- deleting *racA* in *A. niger* produced a hyperbranching phenotype and dispersed
  cultures in the reported conditions ([Kwon et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3722221/));
- conditional *arfA* expression in *A. niger* affected pellet diameter, dispersed
  morphology and protein secretion ([Fiedler et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5952172/));
- deleting *rac1* in *T. reesei* caused hyperbranching, with a carbon-dependent
  cellulase effect on lactose ([Fitz et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6798449/)); and
- α-1,3-glucan and galactosaminogalactan both contributed to *A. oryzae* hyphal
  aggregation in liquid culture ([Miyazawa et al.](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2019.02090/full)).

These papers concern particular strains and culture conditions. The app preserves
that caveat in each evidence trace. For *F. venenatum*, the demo intentionally
does not transfer regulatory rules from another *Fusarium* species.

## The role of Claude

Claude is an optional explanation layer. It receives a bounded, structured copy
of results the deterministic code has already computed. It cannot change fluxes,
sensitivity ranks, experimental designs, morphology scores or confidence labels.
The full modelling workflow works with no Anthropic key, and API keys are read
from server-side environment variables rather than source code.
