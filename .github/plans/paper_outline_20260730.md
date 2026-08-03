# Journal paper outline — mesh-GNN IPM surrogate (2026-07-30, draft)

> Target register: **IEEE Trans. Energy Conversion / Trans. Magnetics** (sober IEEE).
> Status: living outline; numbers cite the methodology review `.github/plans/
> methodology_review_20260720.md` (§ refs).
> **Updated 2026-08-03**: §21 (240-case) has landed and is **negative under matched
> compute** — the scaling claim is now a two-point rise plus a plateau, not an
> open-ended slope. The Arkkio validation number is **0.35%**, not 0.60% (§22).
> Remaining placeholder: whether the plateau is data saturation or training budget
> (one 50-epoch doe240 run decides it).

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
> diversity is the only axis that moves it — 40→120 designs drops held-out |B|
> 12.75%→11.96% uniformly across all six test geometries. A further doubling to 240
> designs at matched gradient-step budget does **not** improve it (12.11%, within
> run-to-run variation), placing the lever's useful range inside the first doubling for
> this design space. Under a
> pinned, byte-reproducible protocol with a checksum-identified checkpoint registry, the
> best surrogate reaches 11.96% |B| / 5.25% torque nRMSE on held-out geometries. The
> results reframe surrogate accuracy for electrical-machine analysis as a data-diversity
> problem and provide a reusable, torque-faithful benchmark.

Abstract knobs still open: whether the 240-case plateau survives an epoch-matched (rather
than step-matched) rerun (§21 "다음"); if it does not, the third sentence becomes a
three-point rising curve instead of a rise-then-plateau.

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
6. **Data scaling as the operative lever — and where it stops.** Expanding training-geometry
   count 40→120 (Motor-CAD LHS, test/val held fixed) drops held-out |B| 12.748%→11.960%,
   broadly across all six test geometries — the first and only axis to break the floor
   (§20). A second doubling 120→240 at matched gradient-step budget returns 12.106%
   (+0.15pp, inside the ~0.25pp run-to-run band) and *worsens* torque 5.252%→6.349%, so the
   measured slope is a rise then a plateau, not an extrapolable trend (§21). Reported as a
   bounded result: the useful range of the data lever for this four-parameter design space
   lies in the first doubling. The 240-case model does improve *mean* torque monotonically
   (2.95→1.17→0.71%) while compressing the case-to-case ripple spread — a regression-to-the-
   mean signature that keeps the training-budget confound explicitly open (§21).

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
  pipeline; fixed test/val; the 40/120/240 curve; per-case breadth of the 40→120 gain;
  generalization-gap narrowing; the 120→240 plateau under matched gradient-step budget
  and its confound (§20–21).
- **VII. Discussion.** Data-limited generalization within a bounded range; **no
  extrapolation to G1(5%)** — the measured curve plateaus after one doubling, so the
  honest statement is that the data lever alone does not reach the 5% gate in this design
  space; deployment candidate; and a **measured** bar for the surrogate-as-solver-
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
| **data** | **120 geometries** | 9,285,890 | 50 | **11.960** | **5.252** | §20 |
| **data** | **240 geometries** (step-matched) | 9,285,890 | 25 | 12.106 | 6.349 | §21 |

Run-to-run variation on this harness is ~0.25 pp in |B| (measured across repeats);
nothing inside that band is claimed as an effect.

### T3 / F5 — per-case data-scaling breakdown (test geometries, held identical)

|B| nRMSE %:

| case | 40 geom | 120 geom | 240 geom | 40→120 | 120→240 |
|---|---|---|---|---|---|
| 4 | 10.320 | 9.652 | 9.880 | −0.67 | +0.23 |
| 7 | 11.318 | 10.586 | 10.948 | −0.73 | +0.36 |
| 18 | 12.658 | 11.948 | 11.981 | −0.71 | +0.03 |
| 32 | 22.075 | 20.778 | 20.565 | −1.30 | −0.21 |
| 37 | 15.519 | 14.282 | 14.345 | −1.24 | +0.06 |
| 39 | 10.482 | 9.980 | 10.220 | −0.50 | +0.24 |
| **pooled** | **12.748** | **11.960** | **12.106** | **−0.79** | **+0.15** |

Torque nRMSE %: 5.219 / 5.252 / 6.349 pooled. Case 32 dominates the pooled torque figure
(its FEM mean torque is −5.0 N·m/m, near zero, so the normalization is inherently large,
§13); **excluding case 32** the pooled torque is 4.151 / 4.013 / 5.076 % — the 240-case
torque regression is real, not a normalization artifact.

Torque decomposition, which is what keeps the budget confound open:

| | 40 | 120 | 240 |
|---|---|---|---|
| mean-torque error (normalized) % | 2.948 | 1.168 | **0.709** |
| ripple error (case mean) % | 24.49 | **10.89** | 43.81 |

The 240-case model has the best mean torque of the campaign and the worst ripple; its
per-case ripple predictions collapse toward the dataset mean (over-predicts the three
low-ripple cases, under-predicts the three high-ripple ones) while the pooled ripple mean
stays right (843.9 vs 862.1 true). That is an under-training signature, so F5 must be
captioned "matched gradient-step budget", not "matched training".

F5 to draw: |B| vs #training geometries (30 / 110 / 230 on a log x-axis), two series
(|B| and torque), error band ±0.25 pp, and the 240 point annotated with its epoch count.

## 5. Limitations

- Single topology family (48-slot/8-pole IPM); geometry varies only two ratios (bore ±10%,
  slot-depth ±15%) about one reference machine — not a cross-topology claim.
- 2D magnetostatic, on-load torque operating condition; no 3D end-effects, no transient
  eddy/AC-loss, fixed mechanical speed.
- Small validation/test holdout (4 val / 6 test geometries); scaling curve has three points
  (40/120/240) — a rise then a plateau, and the plateau point is confounded with training
  budget (step-matched, so the 240-case model sees each sample half as often). One
  epoch-matched rerun would separate the two; until then the plateau is reported as
  "no further gain at equal compute", not as data saturation.
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
- **Decide the epoch-matched doe240 rerun** (~24.3 h GPU). It sets whether VI reads
  "rise then plateau" or "three-point rising curve", and whether the abstract can keep the
  bounded-range framing. Everything else in the paper is invariant to it.
- Draw F5 (spec above). F6 is now optional-but-available: the warm-start PoC landed
  negative, and a three-bar iteration-count chart (17.5 / 14.8 / 9.3) is the cleanest way
  to state the bound if VII keeps that paragraph.
- **New reproducibility asset worth a sentence in IX:** `fem_warmstart/` is a
  licence-free reference solver (numpy + scipy only) that reproduces Motor-CAD's torque to
  0.06–0.18% mean over a full rotor sweep. Reviewers can re-derive the ground truth
  without a commercial licence, which is unusual for this literature.
- Decide single vs double column, venue (Energy Conversion favors the design-utility
  framing; Magnetics favors the operator/floor rigor).
