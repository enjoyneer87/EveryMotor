# Journal paper outline — mesh-GNN IPM surrogate (2026-07-30, draft)

> Target register: **IEEE Trans. Energy Conversion / Trans. Magnetics** (sober IEEE).
> Status: living outline; numbers cite the methodology review `.github/plans/
> methodology_review_20260720.md` (§ refs).
> **Updated 2026-08-04**: the 240-geometry scale has now been run twice. Under a
> matched *gradient-step* budget it looked like a plateau (§21); under a matched
> *epoch* budget it is the campaign best, **11.638% |B| / 5.400% torque** (§21b).
> The plateau was a training-budget artifact, so the scaling claim is a genuine
> three-point curve again — but with a measured slope (−0.30 pp per doubling) that
> **cannot reach the 5% gate at any feasible dataset size**. That pair of statements,
> not either one alone, is the honest headline. The Arkkio validation number is
> **0.35%**, not 0.60% (§22).

## 1. Title + abstract skeletons

**Working title:** "Torque-Faithful Evaluation and Data Scaling of Mesh-Based GNN
Surrogates for IPM Machine Electromagnetic Analysis"

Alternates (register-consistent):
- A2: "A Torque-Faithful Benchmark and Data-Scaling Study for Graph-Neural-Network
  Magnetostatic Surrogates of Interior-PM Machines"
- A3: "Beyond Field Error: Torque-Consistent Evaluation of GNN Surrogates for IPM
  Electromagnetic Analysis and the Primacy of Training-Geometry Diversity"
- A4: "Data, Not Depth: A Controlled Study of Mesh-GNN Surrogate Accuracy for
  Interior-PM Machine Field and Torque Prediction"

**Drafted abstract skeleton (A1):**
> Machine-learning surrogates for finite-element electromagnetic analysis are usually
> judged by field-error norms, which we show can be decoupled from the torque a designer
> actually needs. We present (i) a torque-faithful, leakage-audited evaluation harness for
> mesh-based graph-neural-network (GNN) surrogates of interior-permanent-magnet (IPM)
> machines, built on an Arkkio air-gap operator validated against a commercial solver to
> 0.35% in mean torque (0.54% pointwise RMSE, waveform correlation 0.996); (ii)
> representation floors that bound achievable error for two nodal-output
> conventions; and (iii) a controlled single-axis elimination showing that model capacity,
> supervision signal, receptive-field architecture, and output representation each fail to
> move a ~12.7% air-gap |B| generalization floor, whereas expanding training-geometry
> diversity does — 40→120→240 designs drops held-out |B| 12.75%→11.96%→11.64%,
> improving every one of the six held-out geometries at every step. We further show this
> axis is easy to mis-measure: holding the gradient-step budget fixed while doubling the
> data makes the same 240-design dataset read as a plateau (12.11%), and only scaling the
> epoch budget with the data recovers the gain. Under a
> pinned, byte-reproducible protocol with a checksum-identified checkpoint registry, the
> best surrogate reaches 11.64% |B| / 5.40% torque nRMSE on held-out geometries.
> Critically, the measured slope of −0.30 pp per data doubling implies the remaining gap to
> a 5% field-error target cannot be closed by data at any tractable dataset size. The
> results reframe surrogate accuracy for electrical-machine analysis as a data-diversity
> problem, bound what that diversity can buy, and provide a reusable, torque-faithful
> benchmark.

Abstract knobs still open: none blocking. Strengthening in flight — a 100-epoch 120-design
run (R5c, launched 2026-08-04, ~24 h wall; the earlier "~11 h" note wrongly copied the
50-epoch runtime) separates "data" from "epoch budget" at the middle point as well.
Without it the 40- and 120-design points may themselves be mildly under-trained, which
would make the reported −0.30 pp/doubling slope a *lower* bound on the true one. That cuts
against our own "data cannot reach the gate" claim, so it is worth closing. Pre-registered:
<11.71% re-bases the curve at matched exposure; 11.71–12.21% confirms it as published.

## 2. Contributions (claim list)

1. **Torque-faithful evaluation harness with a leakage taxonomy.** Three leakage classes
   — *target* (fitting a quantity that caps achievable error), *shortcut* (an input
   proportional to the label), *split* (record-level vs case-level holdout) — each with a
   concrete failure it prevents. Case study: the `time_s` shortcut (proportional to
   cumulative rotor angle in a constant-speed DOE), entangled with the anti-periodic
   sector sign convention; removing it and moving to a case-level holdout changed the
   ranking of every model (§7–9).
2. **Representation floors.** Two nodal-output conventions bounded by ground-truth
   round-trips: node-resampling floor (|B| 17.13% / torque 59.54%) vs discrete-curl
   floor (|B| 5.31% / torque 1.58%). Establishes the achievable ceiling per representation
   and explains why torque and |B| errors decouple (§7, floors JSON).
3. **Arkkio air-gap operator validated against the commercial solver.** Torque from the
   surrogate's air-gap field via a segmented Arkkio operator, self-tested to FEM-identity
   0.000% and validated vs Motor-CAD virtual-work over a full electrical cycle to
   **0.35%** in mean torque / 0.54% pointwise RMSE / 0.996 correlation (§8c, re-derived
   and confirmed §22), making torque a first-class, trustworthy metric rather than a
   post-hoc estimate. The residual is a systematic mesh-faceting bias of the band area,
   same sign at every rotor position, and cancels in the relative metrics the gates read.
4. **Band spectral supervision.** The torque error lives entirely in the admissible band
   harmonic coefficients (band-projection study, §14b); a coefficient-supervision loss
   sweeps BAND_W∈{1,10,30}: torque 6.20/6.31/5.22%, |B| flat ~12.5–12.8% — a torque lever,
   not an |B| lever (§16).
5. **Controlled single-axis elimination of the |B| floor.** Capacity (width, depth),
   supervision, architecture (global attention / long-range edges), and representation
   (nodal-A + curl) each isolated as the sole variable; none moves the ~12.7% held-out |B|
   floor (§13,15–19).
6. **Data scaling as the operative lever — with a measured ceiling.** Expanding
   training-geometry count 40→120→240 (Motor-CAD LHS, test/val held fixed) drops held-out
   |B| 12.748%→11.960%→11.638%, improving all six test geometries at both steps, and every
   region (air-gap 8.21→7.30%) — the only axis to break the floor (§20, §21b). The slope
   is −0.42 then −0.30 pp per doubling: real, still open, and far too shallow to close the
   remaining 6.6 pp to the 5% gate (~22 further doublings ≈ 10⁹ designs). Stated as a
   bound on the lever, not just a win for it.

7. **A scaling confound that inverts the conclusion.** Doubling data at a *fixed
   gradient-step* budget — the natural way to keep compute constant — halves how often each
   sample is seen, and made the 240-design set score 12.106% and read as saturation, with a
   torque regression (5.252→6.349%) and a textbook under-fitting signature (per-case torque
   ripple collapsing toward the dataset mean). Re-running the identical data and config
   with the epoch budget scaled to the data reversed all of it (11.638%, torque excluding
   the degenerate case 4.01→3.94%). We report both runs, because the step-matched protocol
   is a plausible default that yields the opposite scientific conclusion (§21, §21b).

## 3. Section-by-section outline

- **I. Introduction.** ML-FEA surrogates for machine design; the field-vs-torque gap;
  why leakage and evaluation rigor are underreported; contributions.
- **II. Problem & Data.** 2D magnetostatic IPM (48-slot/8-pole, 1/8 anti-periodic sector);
  DOE design space (Ratio_Bore ±10%, Ratio_SlotDepth ±15%, PeakCurrent 10–650 A,
  PhaseAdvance 0–90°); Motor-CAD solve; OnLoadTorque 45-step rotor sweep; mesh (~6–12k
  nodes / ~12–23k elements).
- **III. Mesh-GNN Surrogate.** MeshGraphNet backbone; 9-node/4-edge V2 feature contract;
  nodal-B (average) and nodal-A (P1 curl) output heads; element-support loss.
- **IV. Torque-Faithful Evaluation Harness.** Case-level holdout; leakage taxonomy;
  Arkkio operator + validation; representation floors; element-support nRMSE + torque
  nRMSE (mean + ripple); determinism protocol.
- **V. What Does Not Move the |B| Floor (controlled elimination).** Capacity (§13,15),
  band spectral supervision (§14b,16), global-attention/long-range architecture
  (Transolver §17, Hybrid §18, attention-capacity footnote), representation (curl-A §19).
  Each: hypothesis, single-axis config, result, verdict.
- **VI. What Does: Training-Geometry Data Scaling — and its range.** DOE expansion
  pipeline; fixed test/val; the 40/120/240 curve; per-case and per-region breadth of both
  gains; generalization-gap narrowing; **the step-matched vs epoch-matched protocol
  comparison as a result in its own right** (§20, §21, §21b).
- **VII. Discussion.** Data-limited generalization with a quantified ceiling; **no
  extrapolation to G1(5%)** — the slope is real but ~22 doublings short, so the honest
  statement is that the data lever alone does not reach the 5% gate at any tractable cost,
  and the gate needs a different lever (label quality, mesh resolution, or redefining the
  target); deployment candidate; and a **measured** bar for the surrogate-as-solver-
  initialiser idea rather than a speculative one — reusing the curl-A model (§19) as a
  Newton warm start cuts iterations 15% against a zero start but loses to the trivial
  previous-rotor-angle warm start by 60%, so "seed the solver with the surrogate" is
  quantitatively premature at 18.7% |B| (§23).
- **VIII. Limitations & IX. Reproducibility.** (see §5–6 below.)
- **X. Conclusion.**

## 4. Figures / tables → existing artifacts

| # | Item | Source artifact | Status |
|---|---|---|---|
| F1 | IPM 1/8 sector mesh + regions | `eval/mesh_regions`, GIF frame `tools/make_field_gif.py` | have |
| F2 | Field error map (air-gap/magnet concentration) | `results/viz/contour_h128_vs_h256_case*.png` | have |
| F3 | Torque waveform: FEM vs surrogate (ripple) | `results/viz/torque_*_case*.png` | have |
| F4 | Leakage: time_s shortcut before/after ranking | benchmark JSONs pre/post §9 | have |
| T1 | Floors + baselines scorecard | `results/benchmark_v2_floors.json`, `gate_check.json` | have |
| T2 | Single-axis elimination summary | `benchmark_v2_nodeB_{h256,proc24,spectral_bw*,r3_transolver,r3_hybrid,r4_curl_spectral}.json` | have — numbers frozen below |
| F5 | **Data-scaling curve |B| vs #geometries (40/120/240)** | `benchmark_v2_nodeB_{spectral_bw30,doe120_bw30,doe240_bw30}.json` | numbers in hand (T3); figure **to draw** |
| T3 | Per-case |B|/torque, 40 vs 120 vs 240 | by_case in the above JSONs | have — table below |
| T4 | Motor-CAD solve-time / DOE cost | `doe_manifest.json` solve_time_s | have |
| F6 | (opt) FEM warm-start Newton-iteration count | `results/fem_warmstart_case{4,7}.json` | **have — negative** |

### T2 — single-axis elimination (frozen numbers, all on the fixed 6-geometry holdout)

Every row: same case-level split, same 270 samples, same element subset (coverage 0.8649),
scored by `eval/benchmark.py`. Only the named axis differs from the row above it.

| Axis | Config | Params | Ep | TEST \|B\| % | TEST torque % | § |
|---|---|---|---|---|---|---|
| baseline | h128 / p15, no spectral | 2,332,802 | 60 | 13.016 | 8.240 | §10 |
| capacity — width | h256 / p15 | 9,285,890 | 100 | 12.682 | 6.804 | §13 |
| capacity — depth | h256 / p24 | 14,617,346 | 100 | 12.733 | 6.557 | §15 |
| supervision | + band spectral w=1 | 9,285,890 | 100 | 12.499 | 6.204 | §16 |
| supervision | + band spectral w=10 | 9,285,890 | 100 | 12.774 | 6.306 | §16 |
| supervision | + band spectral w=30 | 9,285,890 | 100 | **12.748** | **5.219** | §16 |
| architecture — global attention | Transolver, slice 32 | 8,907,682 | 100 | 16.572 | 5.765 | §17 |
| architecture — long-range edges | HybridMGN h208, world edges | 9,485,842 | 100 | 12.829 | 5.513 | §18 |
| representation | target A → P1 curl | 9,285,633 | 100 | 18.731 | 8.720 | §19 |
| **data** | **120 geometries** | 9,285,890 | 50 | **11.960** | 5.252 | §20 |
| data | 240 geometries, step-matched | 9,285,890 | 25 | 12.106 | 6.349 | §21 |
| **data** | **240 geometries, epoch-matched** | 9,285,890 | 50 | **11.638** | **5.400** | §21b |

Run-to-run variation on this harness is ~0.25 pp in |B| (measured across repeats);
nothing inside that band is claimed as an effect. The two 240-geometry rows differ only
in epoch budget — same data, same split, same config, same seed.

### T3 / F5 — per-case data-scaling breakdown (test geometries, held identical)

|B| nRMSE %, each run trained to its own convergence:

| case | 40 geom | 120 geom | 240 geom | 40→120 | 120→240 |
|---|---|---|---|---|---|
| 4 | 10.320 | 9.652 | 9.389 | −0.67 | −0.26 |
| 7 | 11.318 | 10.586 | 10.364 | −0.73 | −0.22 |
| 18 | 12.658 | 11.948 | 11.489 | −0.71 | −0.46 |
| 32 | 22.075 | 20.778 | 20.695 | −1.30 | −0.08 |
| 37 | 15.519 | 14.282 | 13.320 | −1.24 | −0.96 |
| 39 | 10.482 | 9.980 | 9.756 | −0.50 | −0.22 |
| **pooled** | **12.748** | **11.960** | **11.638** | **−0.79** | **−0.32** |

Every geometry improves at every step — the gain is broad, not carried by one case.

By region, |B| nRMSE % (40 / 120 / 240): air-gap 8.21 / 7.62 / **7.30**, rotor-iron
12.44 / 11.30 / **10.48**, stator-teeth 12.33 / 11.73 / **11.66**, magnet 13.07 / 12.83 /
**12.83**. The air-gap is where torque is computed, so its monotone improvement is the
physically meaningful one.

Torque nRMSE %: 5.219 / 5.252 / 5.400 pooled. Case 32 dominates that figure (its FEM mean
torque is −5.0 N·m/m, near zero, so the normalization is inherently large, §13);
**excluding case 32** it is 4.151 / 4.013 / **3.938** %, i.e. the 240-geometry model is
also the campaign torque best once the degenerate normalization is set aside.

**The step-matched run, kept as a contrast (contribution 7):** same data, 25 epochs —
|B| 12.106%, torque 6.349%, excluding-case-32 torque 5.076%. Its under-fitting signature
is the per-case torque ripple collapsing toward the dataset mean (over-predicting the
three low-ripple cases 1.8–1.9×, under-predicting the three high-ripple ones, while the
pooled ripple mean stays right at 843.9 vs 862.1 true). At 50 epochs the spread recovers
(pred 847.4).

F5 as drawn (`tools/make_campaign_figures.py`, fig2): stacked |B| and torque panels vs
#training geometries (30/110/230), ±0.25 pp band, the step-matched run shown hollow on
both panels, and the epoch/step counts annotated per point.

## 5. Limitations

- Single topology family (48-slot/8-pole IPM); geometry varies only two ratios (bore ±10%,
  slot-depth ±15%) about one reference machine — not a cross-topology claim.
- 2D magnetostatic, on-load torque operating condition; no 3D end-effects, no transient
  eddy/AC-loss, fixed mechanical speed.
- Small validation/test holdout (4 val / 6 test geometries); the scaling curve has three
  points (40/120/240), which fixes a slope but not a functional form — the −0.30 pp/doubling
  extrapolation to ~10⁹ designs should be read as an order-of-magnitude infeasibility
  argument, not a prediction.
- Only the 240-design point has been trained at two epoch budgets. If the 40- and
  120-design runs are themselves mildly under-trained, the true slope is steeper than
  −0.30 pp/doubling and our "data cannot reach the gate" claim weakens accordingly.
- The one clearly-moving axis (data) is also the most expensive and the least portable:
  it presumes a licensed Motor-CAD host and ~160 s of solve per added geometry.
- Surrogate is a field/torque predictor, not a guaranteed solver. The obvious remedy —
  seed a Newton solve with it and keep FEM guarantees — was built and measured, not left
  as future work, and it does not yet pay: across two held-out geometries × 45 rotor
  positions the curl-A initial guess needs 14.8 Newton iterations against 9.3 for simply
  reusing the previous rotor angle (17.5 from zero). Reported as a bound on the idea, not
  a refutation of it (§23).

## 6. Reproducibility statement material

- **Checksum registry:** `results/checkpoints.json` — every model identified by sha256, not
  filename; metrics + scorecard path per entry.
- **Determinism:** pinned TF32/cuDNN/`use_deterministic_algorithms`; byte-identical
  scorecards across repeats on one stack, 3-decimal cross-machine contract
  (`mgn_nodeB_long` 13.324% |B| / 9.207% torque). `eval/determinism.py`.
- **Split manifests:** `eval/splits/doe{40,120,240}_case_split.json`, each carrying a DOE
  digest (`manifest_digest`) so a split provably matches its dataset; test/val held
  identical across scales for fair comparison.
- **Data pipeline:** `results/logs/doe_gen.py` (Motor-CAD LHS), `doe_ingest.py`,
  `doe_make_split.py`, `run_doe{120,240}_docker.sh`; NGC container
  `nvcr.io/nvidia/physicsnemo/physicsnemo:26.03`, no custom install.
- **Governing plan of record:** `.github/plans/methodology_review_20260720.md` §§7–21.

## 7. Open items before submission
- ~~Fill §21 (240-case |B|/torque) into abstract, T2/T3, F5.~~ **Done 2026-08-03** — §21
  landed negative under matched compute; abstract, T2, T3 now carry the frozen numbers.
- ~~Confirm the "[0.60]%" Arkkio-vs-Motor-CAD validation number and its source artifact.~~
  **Done 2026-08-03 (§22)** — the reproducible figure is **0.35%** (mean torque), 0.54%
  pointwise RMSE, 0.9955 correlation, from `results/motorcad_torque_case0004.json` +
  `backup/doe_data/case_0004` via `tools/compare_torque_waveforms.py`. The 0.60% in §8b
  does not reproduce from any committed artifact and must not be cited.
- ~~Decide the epoch-matched doe240 rerun.~~ **Done 2026-08-04 (§21b)** — ran 50 epochs,
  |B| 11.638%, clears the pre-registered 11.71% threshold. VI now reads as a three-point
  rising curve plus a measured ceiling, and the step-matched run is promoted from
  "confound to disclose" to contribution 7.
- **A 100-epoch 120-design run is RUNNING (R5c, launched 2026-08-04, ~24 h wall)** — the
  one experiment that could undercut our own ceiling claim; pre-registered decision rule
  in methodology_review §21b.
- ~~Draw F5.~~ Done (`tools/make_campaign_figures.py` fig2). F6 is optional-but-available:
  the warm-start PoC landed
  negative, and a three-bar iteration-count chart (17.5 / 14.8 / 9.3) is the cleanest way
  to state the bound if VII keeps that paragraph.
- **New reproducibility asset worth a sentence in IX:** `fem_warmstart/` is a
  licence-free reference solver (numpy + scipy only) that reproduces Motor-CAD's torque to
  0.06–0.18% mean over a full rotor sweep. Reviewers can re-derive the ground truth
  without a commercial licence, which is unusual for this literature.
- Decide single vs double column, venue (Energy Conversion favors the design-utility
  framing; Magnetics favors the operator/floor rigor).
