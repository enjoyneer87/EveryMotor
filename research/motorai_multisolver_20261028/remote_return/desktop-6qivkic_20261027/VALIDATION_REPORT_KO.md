# 원자료 검증 보고 — iso 회수 갱신

2026-10-27. 실행 local `desktop-6qivkic`, 원 보존 host `iso / DESKTOP-513NQ4P`. 목적은 MGN·scalar·다중 소스·Field 전이 기법 개발이다. 원고·회신·승인 AF 계수·원본 solver 파일은 변경하지 않았다. 이전 자동화는 재개하지 않았고, 열려 있던 다른 작업의 Motor-CAD를 건드리지 않았다.

## A. 출처와 요청 수

요청한 MS24 + HalfSC36 = **60개 기존 case 모두 발견**, 누락0, 중복 physical acquisition0이다. 최신 AF source commit은0034d7c24b1cc0cf41499d028733792b14815bc7이다. 이전 반환 commit156e80a의 missing 판정은 이 갱신으로 대체한다.

MS24는 INPUT_INDEX_60에서 loaded192가 실제 참조한 Ref source key만 골랐다. A·MS_MANIFEST·MS_RECEIPT와 receipt→archive-member manifest→input/mesh/log hash를 대조했고, canonical index의 mesh.nodes·solver.log hash도 확인했다. A24 원본을234,050,911 byte의 보존 bundle로 회수한 뒤 bundle SHA와 모든 member hash를 검증했다. 별도 zip은 로컬 보존물이며 Git 원시 Field 업로드가 아니다.

HalfSC32의 ARCHIVE_SHA256은 frozen HALFSC_POINTS_36의 source_raw_manifest_sha256과 일치한다. PRESERVATION_RECEIPT→MANIFEST와 읽은 metadata/member를 대조했다. exploratory4는 다른 보존 root의 실제 case/native와 FIGURE14_RESULT·derived audit를 찾았고, final scalar가 일치한다. 두 root에서 별도 PRESERVATION 문서를 찾지 못했고 frozen 행에도 archive-root hash가 없으므로 **독립 provenance root는 UNKNOWN**이다. 내부 member 해시 일치나 scalar 일치를 이 제한의 해결로 과장하지 않는다.

HalfSC native·회로·mesh·input·solver.log와 VTU1/256/512를 직접 읽고 hash 검증했다. 모든512 VTU를 다시 hash한 것은 아니다. 원래 Full240의 saved current/loss audit와 archive manifest240건 검증, latest192 연결은 이전 반환에서 유지한다. 나머지187 loaded Full native를 이번에 새로 적분한 것은 아니다.

전체 inventory는324 acquisition, 명목 조건300, geometry3, 독립 기준 template1을 유지한다. Full240과 low48 중 SC0.1A24조건은 중복 acquisition이고, 새로 회수한60case도 이미 기록된 조건이다. 독립 label 수를 추가하지 않는다.

## B. 물리량과 단위

원시 Full/HalfSC의 PARAMETERS·MODEL·INPUT_CONTRACT와 실제 mesh 면적을 사용했다. 기준 conductivity47092391.429245 S/m, active length0.15m, 기준 Cu 접선3.711×반경1.686mm 계약을 유지하며 다른 형상에 일괄 대입하지 않는다. solve-time 온도 direct readback은 UNKNOWN이다.

```text
J_total = sigma * (V_component / active_length - dAz/dt)
I_integrated = integral_area J_total dS
P_Joule = 8 * active_length / sigma * integral_area J_total^2 dS
P_DC = 8 * active_length / sigma * sum_bar(I_prescribed^2 / area_bar)
P_AC_excess = P_Joule - P_DC
J_residual = J_total - I_prescribed / area_bar
```

전류는 signed conductor current이고 symmetry8을 곱하지 않는다. W는 전체 모터 활성부 2D 적분이며 end winding을 포함하지 않는다. phase current는 RMS, gate의 분모는 sqrt(2)*Irms이다. J_residual은 분석상 residual이며 Motor-CAD Je의 이식이 아니다. 적분 전류 오차에 따른 cross term을 보존한다. Full의 skin/proximity 분리는 임의 생성하지 않는다.

B는 P1 curl `(dAz/dy,-dAz/dx)`다. E[B]²와 E[B²]를 구분하고 MS는128×600 공간 표본, Full 상세 이력은 도체별 area mean/second moment를 제공한다. Full B를 canonical MS excitation으로 대체하지 않았다.

## C. 원시 적분과 시간 구간

MS24의 raw A→P1 curl→600 sample 결과는 기존 scalar-B와 **24/24 bitwise 동일**하다. 독립 determinant curl과의 최대 차이는 4.4408921e-16 T다. raw mesh의 original node/triangle/physical ID 집합과 stationary 슬롯 좌표도 대조했다. numerical parity tolerance1e-12 T는 구현 재현 검사이며 물리 정확도 gate가 아니다.

그 B를 기존 동일 mean-square-field Hybrid evaluator에 적용해 Ref/SC192와 HalfSC36을 재현했다. 최대 절대 차이 3.6379788e-12 W, 평균 절대 차이 1.2431381e-13 W다. 기존 denominator replay의1e-6 W numerical gate를 적용했다. AF 재적합·계수 수정은 없다.

Full/HalfSC는 initial A=0, 첫 BE step 이후 BDF2로512 step을 읽었다. P2는257–512 step의 post-step arithmetic mean, 전기위상361.40625–720도다. actual time_s와 dt는 case별 raw audit에 남겼다. whole-history/trapezoid 평균으로 바꾸지 않았다.

HalfSC **36/36** raw Joule/DC/AC 재적분이 기존1e-6 W replay gate를 통과했다. AC 최대 차이 2.6193447e-10 W다. 누적42조건 AC 최대 차이는 3.4924597e-10 W다. 이 오차는 저장 결과 재현 오차이며 새 모델의 예측 오차나 solver 정확도가 아니다.

| 범위 | P2 current | whole-history current | 직접 검증 수준 |
|---|---|---|---|
| latest192 | PASS192 | PASS169 / FAIL23 | saved field-derived history 재집계; native5 직접 |
| original240 | PASS192 / FAIL48 | PASS169 / FAIL71 | saved history240; native6 직접 |
| HalfSC36 | {'PASS': 36} | {'PASS': 33, 'FAIL': 3} | native512 step36 직접 |
| 누적 raw42 | {'FAIL': 1, 'PASS': 41} | case별 audit 참조 | Ref/SC6 + HalfSC36 |

HalfSC whole-history FAIL case는 `HALFSC_FIG14_N16000_I690_B36_N256_P2_A02, HALFVAL_N16000_L1_B00_N256_P2_A01, HALFVAL_N16000_L3_B54_N256_P2_A01`다. P2 current PASS와 별개로 보존했다. 기존 Ref_P001은 P2 current0.005093805% FAIL, SC_P117은 P2 PASS/whole0.002692% FAIL을 유지한다. current gate0.001%는 기존 SCIENTIFIC_GATES와 auditor 기준이다.

저장 CURRENT_LOSS_AUDIT의 step별 max_field_current_error_per_phasepeak는 무차원 비율이고, 요약/current gate는 percent다. 최초 회수 어댑터의 expected 비교값에 누락된 ×100 변환을 바로잡고36건 전체를 원본 row와 RESULT_SUMMARY로 재검증했다. 원시 J/current 적분 및 PASS/FAIL은 변경되지 않았다. current_metric_unit_audit.json에 변환 전 비율·변환 후 percent·native 차이를 함께 보존했다.

period/time-step/mesh 수렴은 **UNKNOWN**이다. P1/P2 차이를 P2/P3 period 수렴 증거로 쓰지 않는다. HalfSC period_convergence_qualified=false, mesh_convergence_qualified=false를 유지한다. 원시 재적분 PASS를 전체 물리 검증 PASS로 확대하지 않는다.

## Field 샤드, FE 연결, 축소

원 native ASCII3 node ID/permutation을 매 step 확인하고 copper A·좌표·연결을 VTU1/256/512와 대조했다. GeometryIds는 INPUT_CONTRACT의 logical-body→physical mapping을 쓴다. 최근접 좌표로 ID를 복구하지 않는다. 이 reader는 stationary copper 범위이며 전체 회전자/antiperiodic native reader로 인증하지 않는다. J는 element-node이며 경계 값을 평균으로 합치지 않는다.

기존 Full6에 HalfSC3 공간 샤드를 추가했다. 선택은 저부하2k/L1/B0, 고주파16k/L4/B90, exploratory16k/I690/B36으로 OP 범위를 따른다. predeclared case field도 학습·튜닝 금지다. NPZ float64 압축 후 독립3점 triangle quadrature와 A→B/current 검사를 수행했고 결과는iso_recovery/shard_validation.json에 있다.

8위상 공간 샤드의 단순 평균은 P2 loss를 보존하지 않는다. 기존6 최대8.09686% 차이와 새 HalfSC case별 차이를 raw audit에 기록했다. scalar label은 전체256 P2 적분을 사용한다. 전체36×512 손실·전류를 공개하고, 상세 B moments·component voltage 및 원 native는 iso/로컬 보존 경로로 연결했다. solver time step 변경은 없다.

MS FE 예제2개는 실제6개 균질 슬롯 connectivity와128위상을 포함한다. MS에 개별 Cu/J label은 없으며, 구리 Full graph와 동일 node ID로 직접 붙일 수 없다. 후속 model 입력과 label 누출 계약은MGN_DATA_CONTRACT_KO.md에 분리했다.

## D. 마스크와 MS/Full 대응 수

Ref96: own36/excluded60. SC96: own27/excluded69, transfer-coordinate12 추가 제외57. HalfSC: predeclared32와 exploratory4. 예측/pseudo label을 새 Full 정답으로 세지 않는다.

canonical source→target 연결은 **228조건**이다. 양쪽 raw의 수치 검증까지 된 계보 쌍은 **41조건**(latest192의5 + HalfSC36), raw Full만 추가된 저전류 진단1은 이 수에 넣지 않는다. latest192의 source-current mapping exact120/bounded72와 최대0.0375A 차이를 보존했다. HalfSC current mapping도 원래 행과 계산된 실제 차이를 함께 기록했다.

이41은 동일 물리 형상·OP의 exact pair가 아니다. 균질 MS source, speed scaling, SC/HalfSC similarity transfer가 포함된다. 현재 cross-solver exact bridge는 **0**이며 Motor-CAD/GetDP의 nominal-only 대응도 승격하지 않는다. 동등 geometry·절연·공극·회로·재료·온도·파형 readback과 OP가 추가로 필요하다.

## E. 학습 가능 범위와 다음 비교

latest192 기존 P2 target은 고정 마스크를 지킨 개발 비교에 사용 가능하다. MS24는 검증된 저충실도 교사/사전학습 후보이며 Full AC truth가 아니다. original240의 P2 current FAIL48과 Ref_P001은 품질 격리 대상이다. HalfSC32는 field까지 보호된 역사적 전달 평가,4점은 exploratory 전용이다. low48은 이전 archive inventory 상태로 새 scalar truth를 만들지 않았다.

reserved R1_0199/R1_0076과 모든 alias는 학습·사전학습·튜닝·평가에서 제외한다. 과거 QC 이력은 보존하며 globally never-opened라고 부르지 않는다. geometry/alias group 외에24개 MS history의 공통 source hash도 추적해 inductive 분할 누출을 막아야 한다.

같은 Full 예산의 AF/scalar residual/다중 소스/MS pretraining·Field 전이 비교에는 frozen target budget·원천 그룹·평가 cohort, 입력-label 분리, MS 비용 회계, 독립 template와 수렴-qualified 근거가 남는다. 새 FEA·학습은 허용되었지만 범위·자원·재시도·소유 PID·수렴/중단 기준을 case_queue에 구체화한 후 필요한 case만 실행한다. 현재 결과는 원자료 회수·검증 완료이며 MGN 성능 향상을 주장하지 않는다.
