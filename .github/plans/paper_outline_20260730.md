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
> **Updated 2026-08-09 (through §29):** the campaign now has three further measured
> results. (i) **Physics-prior injection** (linear mu_r=20 solve as per-node input
> features, §27): torque 5.400→**3.354%**, overall |B| 11.638→11.096% on the fixed
> 6-geometry holdout — but the **pre-registered primary metric (airgap < 7.05%) landed
> negative** at 7.138% (dead band 7.05–7.55), and both facts are reported. (ii) The
> **excitation-wiring defect (§24/§26) is repaired and the current axis is live**:
> mode-0 fix verified causally, an ampere-turns gate is permanent in the generation
> pipeline (80/80 pilot cases within 0.01%), and a 320-case geometry×current dataset
> (240 re-labelled fixed-excitation + 40 geometries × {0.5, 0.25}×I_ref) trained to a
> **new campaign best**. (iii) **Current generalization holds** (§29): on 12 held-out
> geometry×current cases, pooled |B| **10.856% / airgap 5.353% / torque 2.614%**;
> per-case torque 1.29–6.14% excluding the near-zero-torque geometry-32 pair. The
> legacy fixed-excitation gate did not regress (pooled 11.107/6.618/3.270 vs the prior
> model's 11.096/7.138/3.354) — **current capability came at zero cost to the original
> task**. G1' now stands 1.618 pp (airgap) and 0.270 pp (torque) away — the closest the
> campaign has been. §29 also fixes an **aggregation trap**: mean-of-per-case-nRMSE is
> not pooled nRMSE, and near-zero-torque cases explode the former (the "16.45%"
> artifact); gates use pooled, per-operating-point claims use per-case with the
> degenerate cases disclosed. R5c resolved (§21c): the 100-epoch 120-design run scored
> 11.762%, inside the pre-registered 11.71–12.21 band — the published curve stands.
> The waveform correlation figure is **0.9955** (§22), not 0.996.
> **Updated 2026-08-11 (§31):** two evaluation-side results. (i) An **independent torque
> operator** (Coulomb local virtual work, one solve) agrees with the annulus-averaged
> Maxwell-stress operator to **0.475%, spread 0.024 pp** on the same field and mesh —
> turning contribution 3's "the residual is a systematic band-discretisation bias" from an
> assertion into a measurement. (ii) An **air-gap smoothness bound**: scored on the exact
> elements the gate reads, with the gate's own pooled-|B|-nRMSE statistic, the best
> angular-harmonic description of the target reaches 9.022% (admissible orders only),
> 6.950% (all n ≤ 200) and **5.396%** (all n ≤ 350, ~interpolation in θ). The champion at
> 6.618% therefore already beats every band-limited angular representation of the field —
> which kills the "guide the network with a Fourier/AGE reconstruction of the prior" idea
> on measurement rather than opinion — and the 1.618 pp remaining to G1' splits into
> 1.22 pp of still-addressable smooth error plus 0.40 pp that needs non-smooth structure.

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
> 0.35% in mean torque (0.54% pointwise RMSE, waveform correlation 0.9955); (ii)
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
2. **Representation floors, and a second floor under the corrected gate.** Two
   nodal-output conventions bounded by ground-truth round-trips: node-resampling floor
   (|B| **16.98%** / torque 59.54%) vs discrete-curl floor (|B| 5.31% / torque 1.58%).
   Establishes the achievable ceiling per representation and explains why torque and |B|
   errors decouple (§7, floors JSON). The same instrument then retired our own *replacement*
   gate's headroom: scored on the exact elements the air-gap gate reads, the best possible
   *angular-harmonic* description of the target field reaches only **5.396%** pooled |B|
   nRMSE (700 parameters on ~750 elements per step), against a 5% target — so the last
   0.40 pp of that gate cannot be reached by any smooth-in-θ field model, only by learning
   mesh-level structure (§31). Unlike the representation floors this is a descriptive
   bound rather than a hard one, because the scatter is deterministic given the mesh and
   the surrogate sees the mesh.
3. **Arkkio air-gap operator validated against the commercial solver.** Torque from the
   surrogate's air-gap field via a segmented Arkkio operator, self-tested to FEM-identity
   0.000% and validated vs Motor-CAD virtual-work over a full electrical cycle to
   **0.35%** in mean torque / 0.54% pointwise RMSE / 0.9955 correlation (§8c, re-derived
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
   Closed loop: the pre-registered 100-epoch check at the 120-design point scored 11.762%,
   inside its 11.71–12.21 "budget was adequate" band, so the published three-point curve
   stands as-is (§21c).

8. **DOE-integrity audit: an excitation axis that never reached the solver — found,
   root-caused, repaired, and gated.** The nominal PeakCurrent axis (10–650 A) was written
   to every case file but consumed by none: the template was in RMS current-definition
   mode, where PeakCurrent is a derived display quantity, and the exported winding
   excitation is a constant 325.3 ampere-turns across a 35× nominal range (0.03% spread,
   three independent computations; causal confirmation by two counterfactual solves,
   §24/§26). Repair: mode-0 per-case injection, verified to reproduce the RMS-mode physics
   to 4–5 decimal places; a permanent post-generation ampere-turns gate now checks that
   "the axis you varied actually reached the fields" (80/80 pilot cases within 0.01%,
   §26/§29). Extends the leakage-audit framing of contribution 1 from model inputs to the
   data-generation pipeline itself.

9. **Physics-prior injection, pre-registered.** A licence-free linear magnetostatic solve
   (fixed effective mu_r=20 steel, synthetic template winding, per-node A/Bx/By) added as
   input features — the only physics-injection route not yet eliminated (loss-side and
   post-hoc projection moved nothing, §16/§14b). Outcome against pre-registered criteria
   (§27): primary metric (airgap < 7.05%) **negative** at 7.138%; secondary (overall
   < 11.39%) positive at 11.096%; torque — unregistered but the adopted gate's failing
   leg — improved 5.400→3.354%, the only lever that has ever moved it (data moves torque
   −0.08 pp/doubling). Both the negative primary and the torque gain are reported; the
   prior's current coupling (J scaled by I_pk/I_ref) was wired and verified exactly
   (superposition to 1.6e-15) before any current-axis training used it (§27, commit-level
   gates).

10. **Current-axis generalization with an aggregation caveat.** Trained on 320 cases
    (240 geometries at the true fixed excitation + 40 geometries × {0.5, 0.25}×I_ref),
    evaluated on a geometry×current double holdout: pooled |B| 10.856% / airgap 5.353% /
    torque 2.614% on 12 never-seen geometry×current cases, with the legacy
    fixed-excitation gate unharmed (11.107/6.618/3.270 vs 11.096/7.138/3.354) — current
    capability at zero cost to the original task, and the campaign's best airgap
    (6.618%). Method caveat elevated to a result: mean-of-per-case nRMSE ≠ pooled nRMSE,
    and near-zero-torque cases (FEM mean −5 to −54 N·m/m) explode the per-case form
    (a "16.45%" headline that is really two degenerate denominators); we state which
    statistic answers which question (§29).

## 3. Section-by-section outline

- **I. Introduction.** ML-FEA surrogates for machine design; the field-vs-torque gap;
  why leakage and evaluation rigor are underreported; contributions.
- **II. Problem & Data.** 2D magnetostatic IPM (48-slot/8-pole, 1/8 anti-periodic sector);
  DOE design space as *sampled*: Ratio_Bore ±10%, Ratio_SlotDepth ±15%, PhaseAdvance
  0–90°. **CORRECTION (2026-08-04, §24) + REPAIR (2026-08-05/06, §26, §29):** the nominal
  PeakCurrent axis (10–650 A) was written to the .mot files but did **not** propagate into
  the electromagnetic solve — the exported winding excitation is a constant 325.3
  ampere-turns per slot region across the entire DOE (0.03% over a 35× nominal range,
  three independent computations, two causal counterfactual solves). The pre-repair
  dataset is therefore a single-excitation slice (all cases at 650.538 A peak =
  460 A RMS), which does not affect the geometry-scaling conclusions (§20–21b) but
  bounded the deployment claim. The repair (per-case CurrentDefinition=0) was verified to
  reproduce the RMS-mode physics to 4–5 decimals; a permanent ampere-turns gate now
  audits every generated batch; and the corrected 320-case geometry×current dataset
  (240 re-labelled + 40 geometries × {325.3, 162.6} A, saturation |B| tooth p95
  1.74/1.95/2.07 T at 163/325/650 A) trains the current axis for real. Present the
  defect, the repair, and the gate together — the "DOE-integrity audit" is contribution 8.
  Motor-CAD solve; OnLoadTorque 45-step rotor sweep; mesh (~6–12k nodes / ~12–23k
  elements). (45 points/cycle is spectrally clean below order 18; a 24f slot harmonic
  folds to 21f — quantified by a 120-point re-solve, §28 — so waveform-level ripple
  metrics carry that caveat while per-position field/torque metrics do not.)
- **III. Mesh-GNN Surrogate.** MeshGraphNet backbone; 9-node/4-edge V2 feature contract;
  nodal-B (average) and nodal-A (P1 curl) output heads; element-support loss.
- **IV. Torque-Faithful Evaluation Harness.** Case-level holdout; leakage taxonomy;
  Arkkio operator + validation; representation floors; element-support nRMSE + torque
  nRMSE (mean + ripple); determinism protocol. Two additions from §31 and the
  virtual-work cross-check: (i) **an independent torque operator** — Coulomb local
  virtual work from a single solve — agrees with the annulus-averaged Maxwell stress
  operator to **0.475% with a 0.024 pp spread** over five rotor positions on the same
  field and mesh, so the residual is a systematic offset rather than operator
  unreliability, which is the evidence contribution 3 previously asserted; (ii) **an
  air-gap smoothness bound** measured the way the representation floors were, giving the
  gate's own statistic for the best band-limited description of the target (§31, table
  below). State plainly that Arkkio *is* MST — the annulus average of the single-contour
  integral — rather than a competing principle, and likewise that Coulomb's local form
  *is* virtual work, differing only in taking the derivative analytically on one solve
  instead of numerically across two.
- **V. What Does Not Move the |B| Floor (controlled elimination).** Capacity (§13,15),
  band spectral supervision (§14b,16), global-attention/long-range architecture
  (Transolver §17, Hybrid §18, attention-capacity footnote), representation (curl-A §19).
  Each: hypothesis, single-axis config, result, verdict.
- **VI. What Does: Data — geometry scaling, physics-prior injection, and the current
  axis.** DOE expansion pipeline; fixed test/val; the 40/120/240 curve (+ the §21c
  budget-adequacy check); per-case and per-region breadth; **the step-matched vs
  epoch-matched protocol comparison as a result in its own right** (§20, §21, §21b);
  the physics-prior experiment with its pre-registered split verdict (§27); the 320-case
  geometry×current training and the double-holdout current-generalization scorecard,
  including the pooled vs per-case aggregation statement (§29).
- **VII. Discussion.** Data-limited generalization with a quantified ceiling; **no
  extrapolation to G1'(5%/3%)** — the geometry slope is real but the honest statement is
  that data alone does not reach the gate at tractable cost (airgap −0.32 pp/doubling ≈
  34k designs ≈ 62 days of solve); the one lever that moved torque is input-side physics
  injection (5.400→3.354%), and the current standing is airgap 6.618% / torque 3.270%
  against 5%/3% — 1.618 and 0.270 pp away. **That 1.618 pp is not one thing** (§31): the
  best purely angular description of the target sits at 5.396%, so 1.22 pp of the gap is
  still ordinary smooth-field error that data and loss levers can address, and only the
  final 0.40 pp requires learning non-smooth, mesh-level structure. That last 0.40 pp was
  then shown to be **physics, not discretisation** (§31a): doubling the air-gap mesh raises
  the residual rather than lowering it, so the bound belongs to the field and no
  gap-meshing improvement moves it — and, read the other way, the structure is a
  deterministic function of geometry rather than noise, so it is learnable in principle. Current-axis deployment claim now measured
  (unseen geometry×current pooled |B| 10.856 / torque 2.614%), with the near-zero-torque
  normalization caveat stated. Screening claim (Spearman 1.000, n=6) as before. And a
  **measured** bar for the surrogate-as-solver-initialiser idea rather than a speculative
  one — reusing the curl-A model (§19) as a Newton warm start cuts iterations 15% against
  a zero start but loses to the trivial previous-rotor-angle warm start by 60%, so "seed
  the solver with the surrogate" is quantitatively premature at that checkpoint's 18.7%
  |B| (§23).
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
| data (check) | 120 geometries, 100 ep | 9,285,890 | 100 | 11.762 | 7.204 | §21c |
| **physics prior** | + linear-solve node features (V3) | 9,286,658 | 50 | **11.096** | **3.354** | §27 |
| **data** | **320 geometry×current (+prior)** | 9,286,658 | 50 | **11.107** | **3.270** | §29 |

The last two rows continue the single-axis chain: the prior row changes only the input
features on the 240-geometry recipe; the 320 row changes only the training data (adds the
40×{0.5,0.25}·I_ref factorial) on the prior recipe. The 320 row's airgap is the campaign
best, 6.618% (vs 7.138 prior, 7.30 champion); its current-axis scorecard is T5.

### T5 — current-axis generalization (v2 320-case model, pooled per group; §29)

| Group | n cases | pooled \|B\| % | pooled airgap % | pooled torque % |
|---|---|---|---|---|
| legacy 650.5 A (fixed-excitation gate) | 6 | 11.107 | 6.618 | 3.270 |
| unseen geometry×current (all) | 12 | 10.856 | 5.353 | 2.614 |
| — 325.3 A only | 6 | 10.690 | — | 2.404 |
| — 162.6 A only | 6 | 11.058 | — | 3.479 |
| — excluding geometry-32 pair | 10 | 10.646 | — | 2.263 |

Aggregation statement (belongs in IV): gate comparisons use pooled nRMSE (error RMS over
all samples / truth RMS over all samples); per-operating-point claims use per-case nRMSE,
whose mean is a different statistic and is dominated by near-zero-torque cases when they
exist (geometry 32: FEM mean torque −5 to −54 N·m/m → per-case 19.8–129.9%). The v2
per-case torque range excluding that geometry is 1.29–6.14%.

### T6 — air-gap smoothness bound (§31; belongs in IV beside the representation floors)

Pooled |B| nRMSE of an angular-harmonic reconstruction of the ground-truth field, scored
on the gate's own air-gap element set (region `a<k>`, sliding band and geometrically
invalid elements excluded), legacy 6-test × 45 rotor steps.

| Description of the target | params / step | pooled \|B\| nRMSE % |
|---|---|---|
| admissible orders only (4·odd, 24 orders) | 48 | 9.022 |
| all integer orders n ≤ 200 | 400 | 6.950 |
| all integer orders n ≤ 350 (≈ interpolation in θ) | 700 | **5.396** |
| — v2 champion, for reference | — | 6.618 |
| — G1′ air-gap target | — | 5.000 |

Read two ways. The champion beats every band-limited angular description, so a
Fourier-guided input cannot add information. And a hypothetical model that captured the
smooth-in-θ field perfectly would still miss the gate by 0.40 pp. Unlike the representation
floors this is descriptive, not a hard bound: the residual is radial variation across the
band plus element-level scatter, both deterministic given a mesh the surrogate can see.

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
  `doe_make_split.py`, `run_doe{120,240}_docker.sh`; current-axis pilot
  `doe_current_pilot.py` (sharded/resumable), v2 assembly `doe_build_v2.py`
  (repo-relative paths, container-verified), 480 expansion `doe_gen_480.py`; NGC
  container `nvcr.io/nvidia/physicsnemo/physicsnemo:26.03`, no custom install.
- **Post-generation gates:** ampere-turns excitation gate
  (`results/logs/doe_verify_excitation.py`, artifact `results/r8_excitation_gate_80.json`),
  path-resolution gate (`_v2_path_probe.py`), prior current-coupling gate
  (`verify_prior_current.py`, exact superposition), prior scale-invariance check
  (`_prior_scale_check.py`).
- **Training resilience:** `--resume` with run-defining-flag drift guard + atomic
  per-epoch `.last` checkpoints (`train_doe_curl_mgn.py`); the v2 run survived a host
  reboot and a session teardown losing only in-flight epochs.
- **Governing plan of record:** `.github/plans/methodology_review_20260720.md` §§7–29.

## 6b. Pending decisions surfaced by the 2026-08-04 audit (§24–25, r7 design doc)

- **Gate redefinition: ADOPTED (2026-08-04).** G1 (pooled |B| < 5%) sits *below* the curl
  representation floor (5.31% pooled / 4.77% excluding noise regions) — it was never
  attainable under this eval. New primary gate G1' = airgap-|B| < 5% AND torque < 3%:
  torque-faithful, honest (current best FAILS it at 7.30/5.40), achievable (airgap floor
  2.68%). Abstract restatement under G1': the airgap slope is −0.59 → −0.32 pp/doubling
  (epoch-matched 40/120/240), so data alone still needs ~7 further doublings ≈ 34k designs
  ≈ 62 days of continuous Motor-CAD solve — impractical, though no longer the 10⁹ absurdity
  of the ill-posed pooled gate. Present BOTH: "the original gate was unattainable by
  construction (a methodology finding), and the corrected gate remains out of data's
  practical reach (a scaling finding) — physics injection is the sanctioned lever."
- **PeakCurrent correction** (§24, factual — folded into II above).
- **Screening claim now measurable:** mean-torque ranking of the 6 held-out geometries is
  Spearman 1.000 with a 2.1× safety margin (min true gap 270 N·m vs max error 126 N·m) —
  supports a "screening-ready at current accuracy" paragraph in VII, with the n=6 caveat.

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
- ~~A 100-epoch 120-design run (R5c).~~ **Done 2026-08-05 (§21c)** — 11.762%, inside the
  pre-registered 11.71–12.21 "budget was adequate" band. The published curve stands; the
  abstract's under-training caveat can be softened to cite this check.
- **480-case expansion generating (2026-08-09):** +120 new geometries at the legacy
  excitation + 20 of them × {325.3, 162.6} A + a 6-case unseen-level (487.9 A)
  interpolation probe (eval-only). Retrain will move only the data axis on the v2 recipe.
- **G1'-closing experiments pre-registered before running** (methodology §30 when
  written): candidates are airgap-weighted loss revisited on the prior+current stack and
  longer training; decision rules to be fixed before GPU time is spent.
- ~~Fix the node-resampling floor number in contribution 2.~~ **Done 2026-08-11** — the
  committed artifact and the §7 scorecard both say 16.98%; the 17.131% figure came from
  the FNO section's separately-computed variant and must not be cited in contribution 2.
- **Stale figure asset, must not ship:** `results/viz/torque_validation_motorcad.png`
  still renders the retracted −0.60% / corr 0.99971 and a pre-repair "224.05 A" case
  label. The F3 wildcard in the figure table points at it. Delete or regenerate; the
  three-way plot from `tools/compare_torque_waveforms.py` is the asset §22 actually cites
  and carries the operator validation in the same axes.
- ~~Open branch behind the last 0.40 pp of the air-gap gate: physics or discretisation?~~
  **Settled 2026-08-11 (§31a) — physics.** Doubling the air-gap mesh (gap elements
  2550→5100, everything else fixed) *raises* the residual, 7.254→7.593%; the coarse arms
  read lower only because bigger elements average the structure away. Discretisation
  predicted the opposite, so it is rejected. Consequences: (i) the §31 bound is a property
  of the field, not of the generator, which strengthens it as a reported floor;
  (ii) improving the gap mesh — Continuum-Air-style or otherwise — would not move it, so
  that line of work is closed on evidence rather than on cost; (iii) encouragingly, the
  structure is a deterministic function of geometry rather than noise, so the last 0.40 pp
  is learnable in principle. Side finding worth a reproducibility footnote:
  `AirgapMesh_NumLayers` is a no-op on this model (Motor-CAD reports
  `AirgapMesh_NumLayers_Used = 4` however many are requested).
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
