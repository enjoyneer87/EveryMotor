# 최신 JEET 보정 AF 기준선 감사

감사일은 2026-10-27이다. 감사 대상은 `C:/Users/user/.codex/worktrees/jeet-r7-final-integration/JEET-2024_rev1/manuscript/JEET_KDH_10p_EN_revision_20261017.tex`이며, 체크아웃 HEAD는 `73a1e2aff57bc5e2570dfb0c77a78c4452f027bb`이다. 원고, 정식 AF 계수, 원천 저장 결과를 변경하지 않았다. 새 FEA, AF 재적합, Git/network 변경 및 설치를 수행하지 않았다. 아래 재현은 저장된 수치/계수와 저장 scalar-B 캐시를 이용한 오프라인 검산이다.

## 판정

현재 논문 비교 기준선은 **Full-FEA로 보정한 exponent separable AF × mean-square-field Hybrid**이다. 무보정 Hybrid만을 경쟁 기준선으로 쓰면 현재 논문보다 약한 대상을 비교하게 된다. 최신 수치의 계수 평가·보정 마스크·집계는 이번 감사에서 독립 재현되었다. Hybrid 분모도 저장된 Ref scalar-B 캐시로 재현되었다. 원시 MS A에서 B를 다시 추출하거나 원시 Full-FEA Joule/DC 적분을 다시 계산한 수준의 재현은 수행하지 못했다.

| 모집단 | n | 원고 wMAE [%] | 이번 계수 재평가 [%] |
|---|---:|---:|---:|
| Ref 전체 loaded grid, 보정점 포함 | 96 | 0.66 | 0.6637418363 |
| Ref 자기 보정점 제외 | 60 | 2.38 | 2.3805027362 |
| SC 전체 loaded grid, 보정점 포함 | 96 | 0.91 | 0.9105056646 |
| SC 자기 보정점 제외 | 69 | 2.76 | 2.7571100497 |
| SC 자기 보정점 및 전달 관측 좌표 제외 | 57 | 2.64 | 2.6429784113 |
| HalfSC 사전 선언된 새 좌표, 보정 없음 | 32 | 3.15 | 3.1492795805 |
| HalfSC 전달 SC 속도가 학습 대역 내 | 24 | 3.08 | 3.0814433257 |
| HalfSC 2 kRPM, SC 1.125 kRPM로 외삽 | 8 | 6.69 | 6.6926346454 |

wMAE는 `100 * sum(abs(prediction_W - FullFEA_AC_excess_W)) / sum(FullFEA_AC_excess_W)`이다. 각 점의 상대오차를 단순 평균한 값과 다르다. 동일 원천 96점에서 무보정 Hybrid wMAE는 Ref 55.5960%, SC 42.8106%이고, HalfSC 사전 선언 32점은 47.0384%이다. 이 값들은 보정 AF 성능을 대체하지 않는다.

## 정확한 최신 출처

현재 로컬 `D:/KangDH/JEET-repro` main `a024877ad373746336c9f55a9fb5f04731b8432e`와 이를 기반으로 한 실험 checkout의 역사적 `data/e10/*`만으로 최신 논문 결과를 대표할 수 없다. 최신 원천은 `D:/KangDH/JEET-repro-appendix-review-22add80`의 **로컬 Git object database**에 이미 존재한다. 그 checkout 자체의 HEAD는 `22add8069dc196906baedb0ef89de53ef669dbd1`이지만, 읽어낸 객체는 `0034d7c24b1cc0cf41499d028733792b14815bc7`이다. 작업트리 변경 없이 `git show <commit>:<path>`로 필요한 코드/수치만 이 감사 폴더로 추출했다.

- Ref/SC 확정 모델과 행은 `handoff/20261019_AF_loaded96/AF_model_{Ref,SC}_loaded96.json`, `AF_POINTS_192.json`이다.
- 확정 보정 좌표와 전달 관측은 같은 폴더의 `SAMPLING_PLAN.json`, `REGRESSION_INPUTS.json`, `MODEL_REPORT.json`이다. MODEL_REPORT가 기록한 원래 adoption source commit은 `4aadc0d59c4427e8576cc69a8b79c97727bdd85e`이다.
- HalfSC 최종 36행은 `handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json`이다. 그중 4행은 앞서 본 exploratory 점이며 새 32점 통계에서 제외한다.
- HalfSC 사전 예측과 선택 계획은 `handoff/20261020_halfsc_direct_af/{PLAN,COMPARISON}.json`이다. PLAN의 생성 시각은 `2026-10-20T12:13:56.393446+00:00`이다. 이번 감사는 기록된 계획과 최종 행의 일관성을 확인하며 당시 실행 시점의 사전 선언을 소급 관찰한 것은 아니다.
- R7 commit `6dbef9a982883603cd86fa745ef0d3a71a17d4b2` 및 R8 `0034d7c...`는 후속 figure layout 계열이다. 최신 계수/192행/36행이 그 객체에 보존되어 있다. R7 PDF 수치를 추정해 score를 복원한 것이 아니다.

`E:/KDH/Overleaf/JEET-2024_rev1/review_response/20261018_elmer_af_rebuild/results`는 현재 기준선보다 이른 결과다. 그 `AF_RECALCULATION.json`의 Ref full/own-excluded는 1.01184/3.41850%, SC는 1.31698/3.88620%이다. `handoff/20261018_iso240/results` 및 초기 REA 240계열은 근무부하를 포함해 anchor 30점, Ref 보정 42점인 이전 설계이므로 최신 anchor24/loaded96과 혼합하면 안 된다.

## 분모 및 구현 일치

원고 177, 222, 253–255, 289–310, 638–685행과 추출 코드/저장 모델을 대조했다.

- 분모는 scalar `|B|`의 공간 가중 평균제곱을 사용한다. 6 cuboid subdivision × 2 slot sides에 걸친 총 600점이며, 128 MS phase에서 proximity loss를 평가해 산술평균한다. 각 harmonic마다 별도 kernel을 적용하는 line-mean capped-harmonic evaluator와 구분된다.
- `jeet_acloss_rbf/rea_hybrid/e10_sampling.py:evaluate_active_samples`가 이 흐름을 구현한다. 주파수는 target rpm × pole_pairs / 60이며, proximity coefficient는 target 도체 치수와 주파수에서 직접 평가된다.
- skin-excess는 actual conductor area, prescribed RMS current, scaled bare-copper dimensions와 aspect-ratio coefficient로 계산해 proximity에 더한다. AC excess를 대상으로 하므로 transport DC는 Hybrid 분모에 포함하지 않는다.
- 이번 사용 contract는 conductivity `47092391.429245 S/m`, active length `0.15 m`, geometry bare copper `3.711 × 1.686 mm`이다. 과거 Motor-CAD-MS rebuild metadata의 conductivity `46936915.6588685 S/m`와 섞지 않는다.
- 저장 행은 Ref Elmer MS만을 사용하며, 원본 full grid30 MS 중 loaded current–angle 24조건을 쓴다. SC current mapping에는 문서화된 bounded match가 있으며 이번 192행의 source-current 최대 차이는 Ref 기준 `0.0375 A`였다. 모든 행을 exact-current match로 부르면 부정확하다.
- Ref scalar-B 캐시 24개 `[128,600]`를 다시 evaluator에 넣어 192개 Hybrid 분모를 재계산했다. 저장 Hybrid와 최대 차이 `2.91e-11 W`, skin 최대 차이 `9.09e-13 W`이다. 이 검사는 캐시부터 분모까지의 replay이며 원시 A의 P1 curl 추출 재검증은 아니다.

AF는 `f(n) * kappa(I,beta)**p(n)`이다. 2D anchor RBF의 kernel은 `r² log(r+1e-12)`, 정규화 길이는 저장 current/angle 표준편차, diagonal regularization `1e-6`, affine tail과 side constraint가 없다. 속도는 코드에서 kRPM로 사용한다. 각 속도의 `log(AF)` 대 `log(kappa)` 선형회귀에서 level/spread를 얻고, `f(16)=p(16)=1`의 정확한 제약 아래 quadratic을 적합한다. loss-dependent 회귀 가중치를 쓰지 않는다. 예측 시 exponent 경로는 `kappa >= 1e-3` clipping을 포함한다.

이번 독립 replay는 canonical class를 import하지 않고 stdlib `math`만으로 저장 kernel weights, centers, length scales와 두 quadratic coefficient를 평가했다. 저장 예측과 최대 차이는 Ref `2.91e-11 W`, SC `2.91e-10 W`, HalfSC `1.46e-10 W`이다. 모델을 새로 적합하지 않았다.

## 보정·전달·평가 마스크

- Ref는 16 kRPM loaded anchor24와 2/4/8 kRPM의 4점씩, 총 own Full-FEA36점이다. 자기 보정점 제외60점은 2/4/8 kRPM만 포함하며, 전체 truth loss의 23.8303%를 차지한다.
- SC는 anchor24와 8 kRPM own3점, 총 own27점이다. 별도로 2/4 kRPM 각6점, 총12점은 frozen Ref의 전달 예측 손실로 구성한 관측이다. 따라서 SC own-excluded69점 안에도 이 전달 좌표12점이 포함된다. 그 좌표까지 제외한57점도 별도로 보고한다.
- 16 kRPM은24점 전부 anchor fit이다. 전체96점에 대한 0.66/0.91%에는 큰 손실 가중치를 가진 anchor residual이 포함된다. 16 kRPM을 빼면72점에서 Ref2.32493%, SC2.68611%이고, 나머지 own calibration도 빼면60/69점 수치가 된다.
- 원고는 SC budget 비교에 evaluated grid를 사용했음을 밝힌다. 자기 보정점 제외는 model-selection 효과까지 제거한 untouched test set이 아니다.
- HalfSC의 frozen mapping은 `AF_SC(0.5625*n_H, (4/3)*I_H, beta)`이다. 모델 JSON SHA-256 `3499e77e5789549446cf2a0486e18ea708d1754b535f3076f0210cde369a2212`가 COMPARISON의 frozen SC hash와 일치한다. HalfSC truth label을 AF fitting에 투입하지 않는다.
- HalfSC32점의 maximum absolute relative error는11.8970%, mean absolute relative error는4.52061%이다. 2 kRPM8점은 donor SC fit의 하한2 kRPM보다 아래인1.125 kRPM에 매핑된다. 연속조건 전체 또는 완전한 HalfSC map의 오차 상한이 아니다.

## 이번에 독립 확인한 범위와 남은 재현 갭

| 항목 | 이번 확인 | 남은 사항 |
|---|---|---|
| latest coefficient → AF → loss prediction | 독립 stdlib replay 완료 | 없음, 이 저장 모집단 내 |
| own/transfer masks 및 wMAE | metadata 좌표와 Boolean의 일치 및 수치 검산 | 예산 선택을 완전 미사용 test로 해석 불가 |
| 24 saved scalar-B cache → 192 Hybrid | canonical pinned evaluator replay 완료 | raw MS A → P1 curl magnitude를 독립 재추출하지 않음 |
| Full-FEA AC-excess truth | 저장 scalar 및 current flags 사용 | 원시 Joule/DC 적분 및 current waveform 재적분 미실시 |
| HalfSC 사전 예측 일치 | frozen 계수와 저장 예측의 일치 | 원시 보존 아카이브 readback 재실시 안 함 |
| numerical convergence | 기록된 P2 current pass를 확인 | period/mesh convergence를 추가 확립하지 않음 |

반드시 구분할 점은 HalfSC FINAL_AUDIT가 `period_convergence_qualified:false`, `mesh_convergence_qualified:false`를 기록한다는 것이다. 이번32점의 P2 current fail은0개지만 original whole-history fail은2개다. 4 exploratory를 포함한36행에서는3개다. 따라서 저장 P2 비교 성공을 새로운 mesh/period convergence 증명으로 부르면 안 된다.

원격에서 **추가로 필요한 입력**은 다음처럼 최소 범위로 요청할 수 있다. 현재 score/계수 replay에는 추가 입력이 필요 없다.

1. MS→B 원천 재검증에는 `INPUT_INDEX_60.json` 중 실제 baseline24개 Ref source의 `ALL_A_HISTORY.npz`, 매칭 `MS_MANIFEST.json`, `MS_RECEIPT.json` 및 solver log/mesh hash evidence가 필요하다. 기록된 원격 경로는 `D:/KDH/JEET_MS_HALFSC_20261018/<case>/cases/<case>/ALL_A_HISTORY.npz` 계열이다. baseline에 쓰인24개 원시 A 경로는 이 PC에서 모두 존재하지 않았다. 필요한 field SHA-256은192행에 이미 있다.
2. Full-FEA truth 재검증에는 Ref/SC192개 case의 저장 P2 loss/current waveform 또는 원시 field-derived Joule/DC 적분 데이터와 raw-audit receipts, geometry area 및 conductivity/symmetry/current normalization contract가 필요하다. solver를 다시 돌릴 필요 없이 저장 데이터의 read-only 재적분이면 된다.
3. HalfSC truth/preservation 재검증에는32개 새 case의 `RESULT_SUMMARY.json`, `RESULT.json`, `PRESERVATION_RECEIPT.json`, `PRESERVATION_MANIFEST.json`과 manifest가 가리키는 저장 수치/field가 필요하다. 알려진 원격 보존 root는 `D:/KDH/JEET_HALFSC_DIRECT_AF_20261020/`이다. 당시 FINAL_AUDIT의 readback PASS는 보존된 주장으로 유지하며 이번 실행에서 새로 검증했다고 쓰지 않는다.
4. 새로운 geometry/운전축에 대한 일반화 증거에는 위 replay와 별개의 사전 정의된 데이터가 필요하다. 현재 audit만으로 새로운 해석 또는 수집을 승인하지 않는다.

## 재실행과 산출물

이 폴더에서 `python extract_sources.py`, `python replay_baseline.py`, `python replay_denominator.py`를 순서대로 실행한다. 첫 script는 명시된 local Git object를 읽어 이 폴더 안에만 파일을 쓴다. score replay는 Python stdlib만 필요하고 denominator replay에는 현재 설치된 NumPy1.20.1을 사용했다. 실행 전 모든 source hash를 `source_manifest.json`과 대조할 수 있다.

- `baseline_rows_192.csv`는 후속 surrogate 비교에 쓸 truth, canonical Hybrid, frozen calibrated prediction, own/transfer 마스크를 함께 보존한다. **새 모델 baseline은 `replay_prediction_W`이다.**
- `replay_results.json`과 `denominator_replay_results.json`은 정밀 수치와 검증 한계를 기록한다.
- `source_manifest.json`은 추출한46개 numeric/code/cache 파일의 Git commit, path, SHA-256을 기록한다.
- `baseline_manifest.json`은 원고 hash/commit, 각 artifact hash 및 재현 상태를 묶는다.

공식 reviewer 질문과 Word response 내용은 이 산출물에 포함하지 않았다. 메모리 검색에서는 본 감사의 최신96/24 계열을 확인할 만한 기록이 없었으며, 결과 판단에는 현재 로컬 원고/Git 객체/저장 수치만 사용했다.
