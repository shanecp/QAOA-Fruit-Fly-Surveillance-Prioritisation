# QAOA Fruit Fly Surveillance Prioritisation

Code and data for **Evaluating Simulated Quantum-classical Hybrid Optimisation for Fruit Fly Surveillance Prioritisation in Australia**.

This repository provides the numerical pipeline for the paper:

- Source data extraction and processing
- Classical baselines and exact solutions
- Penalty diagnostics and optimiser comparisons
- Wrapped and direct QAOA comparisons
- Mixer-period checks and cost-period counterexamples
- A 20-seed comparison using COBYLA and Powell

The study uses ideal simulation to explore supplementary Queensland fruit-fly surveillance prioritisation in Victoria's Goulburn Murray Valley. It is a proof of concept. It does not establish quantum advantage, pest prevalence, operational readiness or optimal physical trap placement.

## Reproduce the results

Use Python 3.12.12 and the package versions in `uv.lock`. Run these commands from the repository directory:

```bash
uv sync --frozen
uv run python verify.py
uv run python reproduce.py
uv run python supporting.py
uv run python verify.py --results outputs/reproduced --supporting-results outputs/supporting
```

The commands perform the following steps:

1. Install the locked dependencies.
2. Verify the supplied results, mathematical encoding and simulator consistency.
3. Compute exact solutions and run the 540 optimisations in the main experiment matrix.
4. Run the 640 supporting optimisations and angle checks.
5. Compare the generated results with the supplied results.

Fresh outputs are written to `outputs/reproduced/` and `outputs/supporting/`. The optimisation scripts use the processed input. They do not use the supplied results to choose or initialise an optimisation.

All inputs and code needed for numerical reproduction are included. Package installation requires access to the package registry. The numerical experiments use local inputs once installation is complete.

All optimiser outcomes are retained, including unsuccessful runs and runs that reach the evaluation limit. The results measure performance at a fixed evaluation budget. Reaching that budget does not imply convergence.

Local optimisation can be sensitive to package versions and floating-point behaviour. Verification fails if errors exceed these absolute tolerances:

- Metrics and energies: `1e-10`
- Probability vectors: `1e-9`

## Source data and ETL

The source data were used for educational and research purposes. Original documents and raw downloads are not included. Obtain them from the publishers:

- [GMV Queensland Fruit Fly Outlook, March 2026](https://fruitflycontrol.com.au/2026/03/08/queensland-fruit-fly-outlook-march-2026/). Table 3 on PDF page 5 contains 39 locality averages for 19 December 2025 to 3 March 2026. The source is the Goulburn Murray Valley Area Wide Management Fruit Fly Program, funded by the Victorian Government.
- [Vicmap Admin Locality Polygon Aligned to Property](https://discover.data.vic.gov.au/en_AU/dataset/vicmap-admin-locality-polygon-aligned-to-property). Locality geometry is attributed to the State Government of Victoria, Department of Transport and Planning, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The authors derived the centroids and distances.

`data/sources.json` records the source versions accessed on 28 July 2026. It includes filenames, SHA-256 checksums, coordinate systems and attribution.

Refer to the publishers' reuse terms. The research-purpose statement does not assert permission to redistribute GMV-derived values. No software licence is assigned to this repository.

To rebuild the processed input, place the downloaded sources in `data/source/` and run:

```bash
uv run python etl.py \
  --report data/source/Queensland-Fruit-Fly-Outlook-March-2026.pdf \
  --geometry data/source/locality_polygon.json

uv run python reproduce.py --inputs outputs/etl/localities.csv
uv run python supporting.py --inputs outputs/etl/localities.csv
uv run python verify.py --results outputs/reproduced --supporting-results outputs/supporting
```

You can also pass both `--report` and `--geometry` to `reproduce.py` to run ETL and the main experiment matrix together.

The source checksums must match the recorded versions. Publisher downloads can change, so a current Vicmap export may not reproduce the July dataset. If the exact downloads are unavailable, use the bundled processed input to reproduce the numerical results. ETL rejects source files with different checksums.

ETL performs these steps:

1. Extract the 39 monitoring rows.
2. Match each locality name to one polygon.
3. Project the geometry from EPSG:7844 to EPSG:7855.
4. Calculate locality centroids.
5. Rank localities by distance from Shepparton, breaking ties alphabetically.
6. Join the activity values after fixing the geographic order.

The four separately reported Strathbogie localities are excluded.

## Repository structure

- `data/processed/localities.csv`: processed scientific input
- `data/sources.json`: source provenance and checksums
- `src/gmv_qaoa/`: scientific modules
- Root Python scripts: ETL, experiment execution and verification
- `results/`: main reference results
- `results/supporting/`: supporting reference results
- `data/source/`: locally downloaded source files
- `outputs/`: generated runs, rebuilt inputs and verification reports
- `.cache/uv/`: package cache

Git ignores source downloads, generated outputs and caches.

The processed input contains:

- Source rank and locality name
- Average QFF per trap
- Projected centroid coordinates in metres
- Distance from Shepparton in kilometres
- Geographic rank

The reduced instances use the nearest 6, 8, 10 and 12 localities, with budgets of 2, 3, 4 and 4. The full instance uses all 39 localities with a budget of 4.

Activity normalisation and the proximity scale are recalculated within each candidate set. Objective values from different candidate sets therefore do not share a common absolute scale.

## Code and settings

| File | Purpose |
|---|---|
| `study.json` | Main experiment matrix, seeds, objective settings, optimiser settings, angle handling, bootstrap settings and tolerances |
| `supporting-study.json` | 20-seed experiment matrix and angle-check settings, using shared scientific settings from `study.json` |
| `etl.py` | Convert source files to the processed locality CSV |
| `src/gmv_qaoa/model.py` | Define candidate sets, objectives, penalties, QUBO/Ising conversion and bit conventions |
| `src/gmv_qaoa/classical.py` | Compute exact feasible solutions, three heuristics and allocation breakpoints |
| `src/gmv_qaoa/qaoa.py` | Simulate Penalty/X and ordered-ring XY/Dicke circuits, run optimisation and check Qiskit reference circuits |
| `src/gmv_qaoa/metrics.py` | Calculate distribution metrics, sampling metrics, summaries and paired bootstrap intervals |
| `reproduce.py` | Run the main experiments and write results |
| `supporting.py` | Run the 20-seed comparison, mixer-period checks and cost-period counterexamples |
| `verify.py` | Validate supplied results and compare a fresh reproduction against them |

The numerical implementation uses:

- NumPy for statevector simulation
- SciPy for COBYLA and Powell optimisation
- Qiskit for independent reference circuits

Model arrays pass through pandas CSV serialisation and parsing in memory. This defines their floating-point values consistently. Even a difference of one unit in the last place can change an optimiser's path. No intermediate model files are needed.

## Experiments

| Experiment | Runs | Settings |
|---|---:|---|
| Penalty diagnostic | 80 | Four sizes, depths 1/2, ten seeds, direct COBYLA, 150 evaluations at both depths, independent random starts |
| Penalty sensitivity | 20 | `(12,4)`, depth 1, factors 0.5/2.0, ten seeds, direct COBYLA, 150 evaluations |
| Restricted-domain optimiser audit | 120 | Sizes 10/12, depth 1, ten seeds, COBYLA/Powell, caps 150/300/600, wrapped proposals, bounded Powell |
| Wrapped formulation comparison | 160 | Four sizes, both methods, depths 1/2, ten seeds, COBYLA caps 150/300 |
| Direct formulation comparison | 160 | Same paired design with direct, unbounded proposal evaluation |
| Supporting direct comparison | 640 | Four sizes, both methods, depths 1/2, seeds 0–19, unbounded COBYLA/Powell, caps 150/300 |

The supporting comparison and the main direct comparison share 160 COBYLA runs with seeds 0–9. These are the same observations in both result sets. Verification checks that they agree.

At depth two, COBYLA gives mixed median rankings. Powell gives XY/Dicke the higher median in all four instances. Results for the two optimisers are reported separately.

## Numerical conventions

The paired comparisons use these initialisation rules:

- Both formulations share the depth-one random start.
- Depth-two starts duplicate the same formulation's depth-one final angles.
- In the supporting comparison, the depth-one result must also match the optimiser and seed.

The circuit and parameter conventions are:

- The wrapped protocol reduces cost angles modulo `2*pi` and mixer angles modulo `pi/2`. These are search conventions, not claims of physical symmetry.
- XY edges execute in ascending nearest-neighbour order, followed by the closing ring edge.
- State-array index `z` encodes variable `i` as `(z >> i) & 1`.
- Canonical bitstrings list variable 0 first. Qiskit display strings use the reverse order.
- Parameter arrays list all cost angles, followed by all mixer angles.

The supporting angle checks use the following conditions:

- A sufficient X mixer period is `pi`, up to global phase.
- A sufficient ordered XY mixer period is `2*pi`. Its edge generator `(XX+YY)/2` has eigenvalues `-1,0,0,1`.
- The 120 mixer checks cover four sizes, both methods, both depths, five parameter vectors and each mixer layer separately. The tolerance is `1e-10`.
- A cost period requires every `T*(E_z-E_ref)` to be an integer multiple of `2*pi`. The tested `2*pi` shift fails for both formulations at all four sizes.

These checks do not establish the smallest mixer periods or cost-angle aperiodicity.

## Published results

The main results are in `results/`:

| File | Contents |
|---|---|
| `exact-solutions.csv` | Exact allocations, feasible counts, transformed-activity retention and distances for the candidate-set table and Figure 3 |
| `classical-baselines.csv` | Top-k, max-min dispersion and greedy-composite allocations and objective gaps for the classical-baseline table |
| `policy-breakpoints.csv` | Seven full-set allocation intervals, including all ties at interval boundaries |
| `qaoa-runs.csv` | Final runs, initial/final angles, optimiser status, evaluation counts and distribution metrics |
| `qaoa-probabilities.npz` | Final probability vectors keyed by `run_id`, in the documented state-index order |
| `qaoa-summary.csv` | Medians and quartiles for the diagnostic and method comparisons used in Figures 1 and 2 |
| `paired-comparisons.csv` | XY-minus-penalty differences, bootstrap intervals, pair counts and six optimiser-audit comparisons, including the no-wrap results table |
| `sampling-metrics.csv` | Exact-hit probability, expected-best normalised regret and no-feasible-sample probability at each shot budget, with uniform-valid references |
| `angle-checks.csv` | Four representative probability-vector changes under the imposed angle shifts |
| `metadata.json` | Runtime versions, input/configuration checksums, result checksums, record counts and tolerances |

The supporting results are in `results/supporting/`:

| File | Contents |
|---|---|
| `qaoa-runs.csv`, `qaoa-probabilities.npz` | All 640 run records and final distributions, including seeds, initial/final angles, optimiser statuses and evaluation counts |
| `robustness-summary.csv` | Method medians and paired 95% intervals for each optimiser, size and depth |
| `mixer-period-checks.csv` | All 120 mixer-period checks, including parameter vectors and probability errors |
| `cost-period-counterexamples.csv`, `symmetry-validation.json` | Eight rejected cost-period tests, the XY edge spectrum and counterexamples to the imposed angle shifts |
| `metadata.json` | Supporting input/configuration checksums, result checksums, versions and record counts |

Interpret the summaries using these conventions:

- Paired intervals are percentile 95% intervals for the median of paired differences, using 10,000 resamples. They are not differences between the separately displayed method medians.
- Main comparison bootstraps follow the group order, metric order and random-number stream offsets defined in `study.json` and `src/gmv_qaoa/metrics.py`.
- Supporting intervals use a separate random-number generator with seed 20261008 for each optimiser/size/depth group. Each interval summarises 20 paired XY-minus-penalty differences. There is no correction for multiple comparisons.
- Supporting runs retain all outcomes under the 150/300 evaluation caps.
- Sampling regret assigns a value of one to an infeasible draw.
- Exact ties use an objective tolerance of `1e-10`.
- Locality lists use vertical bars as separators. Multiple co-optimal sets use semicolons.
