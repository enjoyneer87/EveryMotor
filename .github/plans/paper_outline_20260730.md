# Journal paper outline — mesh-GNN IPM surrogate (2026-07-30, draft)

> Target register: **IEEE Trans. Energy Conversion / Trans. Magnetics** (sober IEEE).
> Status: living outline; numbers cite the methodology review `.github/plans/
> methodology_review_20260720.md` (§ refs). Fill §21 (240-case) when it lands.

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
> [0.60]%; (ii) representation floors that bound achievable error for two nodal-output
> conventions; and (iii) a controlled single-axis elimination showing that model capacity,
> supervision signal, receptive-field architecture, and output representation each fail to
> move a ~12.7% air-gap |B| generalization floor, whereas expanding training-geometry
> diversity (40→120→[240] designs) does — the first and only lever to break it. Under a
> pinned, byte-reproducible protocol with a checksum-identified checkpoint registry, the
> best surrogate reaches [11.96]% |B| / [5.25]% torque nRMSE on held-out geometries. The
> results reframe surrogate accuracy for electrical-machine analysis as a data-diversity
> problem and provide a reusable, torque-faithful benchmark.

Abstract knobs to finalize after §21: floor %, best %, "40→120→240" endpoint, headline
scaling slope.

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
   0.000% and validated vs Motor-CAD to [0.60]% (§14b), making torque a first-class,
   trustworthy metric rather than a post-hoc estimate.
4. **Band spectral supervision.** The torque error lives entirely in the admissible band
   harmonic coefficients (band-projection study, §14b); a coefficient-supervision loss
   sweeps BAND_W∈{1,10,30}: torque 6.20/6.31/5.22%, |B| flat ~12.5–12.8% — a torque lever,
   not an |B| lever (§16).
5. **Controlled single-axis elimination of the |B| floor.** Capacity (width, depth),
   supervision, architecture (global attention / long-range edges), and representation
   (nodal-A + curl) each isolated as the sole variable; none moves the ~12.7% held-out |B|
   floor (§13,15–19).
6. **Data scaling as the operative lever.** Expanding training-geometry count 40→120→[240]
   (Motor-CAD LHS, test/val held fixed) drops held-out |B| 12.748%→11.960%→[§21], broadly
   across all test geometries — the first and only axis to break the floor (§20–21).

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
- **VI. What Does: Training-Geometry Data Scaling.** DOE expansion pipeline; fixed
  test/val; 40/120/[240] curve; per-case breadth; generalization-gap narrowing (§20–21).
- **VII. Discussion.** Data-limited generalization; slope/extrapolation to G1(5%);
  deployment candidate; (optional) FEM warm-start reuse of the "failed" curl-A model.
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
| T2 | Single-axis elimination summary | `benchmark_v2_nodeB_{h256,proc24,spectral_bw*,r3_transolver,r3_hybrid,r4_curl_spectral}.json` | have |
| F5 | **Data-scaling curve |B| vs #geometries (40/120/240)** | `benchmark_v2_nodeB_{spectral_bw30,doe120_bw30,doe240_bw30}.json` | **to make (§21 pending)** |
| T3 | Per-case |B|/torque, 40 vs 120 [vs 240] | by_case in the above JSONs | have (240 pending) |
| T4 | Motor-CAD solve-time / DOE cost | `doe_manifest.json` solve_time_s | have |
| F6 | (opt) FEM warm-start Newton-iteration count | PoC output (pending) | future |

## 5. Limitations

- Single topology family (48-slot/8-pole IPM); geometry varies only two ratios (bore ±10%,
  slot-depth ±15%) about one reference machine — not a cross-topology claim.
- 2D magnetostatic, on-load torque operating condition; no 3D end-effects, no transient
  eddy/AC-loss, fixed mechanical speed.
- Small validation/test holdout (4 val / 6 test geometries); scaling curve has three points
  (40/120/240) — trend, not an asymptote.
- Surrogate is a field/torque predictor, not a guaranteed solver (motivates the FEM
  warm-start follow-up where the surrogate seeds a Newton solve that keeps FEM guarantees).

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
- Fill §21 (240-case |B|/torque) into abstract, T2/T3, F5.
- Confirm the "[0.60]%" Arkkio-vs-Motor-CAD validation number and its source artifact.
- Make F5 scaling curve figure; optionally F6 warm-start figure if that PoC lands.
- Decide single vs double column, venue (Energy Conversion favors the design-utility
  framing; Magnetics favors the operator/floor rigor).
