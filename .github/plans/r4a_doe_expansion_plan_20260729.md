# R4(a) — DOE geometry 확장 실행 계획 (2026-07-29)

> 사전승인됨("DOE 진행부탁해"). **실행 순서: R4(c) 채점(§19) 완료 후 착수**(사용자
> 지시). 사용자 휴가 중 — 원격만. 로컬 개입 필요 항목은 아래 "원격 필요"에 표시.

## 목표
학습 geometry 30개(40케이스 중) → **120케이스로 확장**해 §14–18이 지목한
데이터-제한 일반화 갭(train ~13.5% / val ~23–24%)을 공략. |B| 바닥이 데이터면
일반화가 개선되어야 한다.

## DOE 공간 (4D, 기존과 동일 — densify)
| 변수 | 범위 | 단위 |
|---|---|---|
| Ratio_Bore | base×[0.90,1.10] ≈ [0.646,0.789] | 무차원 |
| Ratio_SlotDepth_ParallelSlot | base×[0.85,1.15] ≈ [0.425,0.575] | 무차원 |
| PeakCurrent | [10.0, 650.53] | A |
| PhaseAdvance | [0.0, 90.0] | deg |
토폴로지 고정(TestCAD1.mot). 넓힘 없음(외삽 방지).

## 샘플링 — 비증분 LHS 처리
- 신규 80점: `build_doe_lhs(axes, n_samples=80, seed=43, criterion=maximin)`.
- 기존 40(seed 42)과 **병합 → 120**. 최소거리 dedup(정규화 4D, thresh 작게)로
  근접 중복만 제거(연속공간이라 사실상 없음). 기존 40 solve 재사용.
- case 인덱싱: 신규는 case_0040~ 부터(기존 0000–0039 보존).

## 실행 단계 (모두 스크립트, 자율)
1. **라이선스·기동 확인** [원격 필요 가능] — PyMotorEnv_310로 Motor-CAD v261 접속.
   라이선스 미가용/대화형 활성화 요구 시 **정지·보고**(휴가 중이면 블로커).
2. **DOE 생성** — `doe_batch_run(base_mot=D:\KDH\Sim_4SolverX\TestCAD1.mot,
   parallel_workers=?)`. solve ~108s/case × 80 ≈ 2.4h 직렬(/워커). export
   (.mes→.txt→.h5) 포함 ~3–3.5h. 신규 매니페스트 병합.
3. **수집·검증** — 120케이스 doe_manifest 갱신, h5 파싱 스모크(기존 로더로
   신규 케이스 로드 확인), doe_digest 재계산.
4. **split 재생성** — 120케이스 case-holdout(train/val/test 비율 유지, 신규
   split json). 기존 40 split과 분리.
5. **재학습** — R4(c) 승자 구성(또는 bw30 target-B)로 120케이스 학습. graph
   수 3× → 에폭 시간·총시간 증가(에폭1 실측 후 조정). ~6–8h.
6. **채점·판정** — 하니스 채점, **게이트 불변 확인**. |B| 일반화 갭이 줄었는가
   (val/test |B| < 기존 12.7%?). §20 기록·커밋.

## 비용 (이 머신)
- 생성: ~3–3.5h(직렬; 병렬 워커 시 단축, 라이선스 인스턴스 수 의존).
- 재학습: ~6–8h. 총 ~10–12h, 오버나이트 자율.

## 원격 필요(휴가 중 블로커 가능) — 그 외 전부 자율
- **Motor-CAD 라이선스**: 체크아웃 불가/대화형 활성화 요구 시. (1단계에서 조기
  확인·플래그.)
- **병렬 워커 수**: 라이선스가 다중 MC 인스턴스 허용하는지 불확실 →
  기본 `parallel_workers=1`(안전) 사용, 가용 확인되면 상향.
- 그 외(생성·수집·split·재학습·채점) 로컬 개입 불필요.

## 정제 (2026-07-29, digest·비교설계)
- **데이터 격리:** `manifest_digest`가 전 케이스 geometry/electrical을 해시하므로
  40→120이면 digest가 바뀌어 40케이스 게이트 split이 예외("Regenerate the
  split") 발생. 따라서 **120는 별도 dir `backup/doe_data_120`(기존40 복사+신규80)**.
  `backup/doe_data`(40)는 불변 -> 게이트 13.324/9.207 및 §14–19 전부 재현 보존.
- **공정 비교 = test 고정:** doe120 split은 **동일 test[4,7,18,32,37,39]·
  val[5,16,25,28]**를 고정하고 train만 30→110으로 확장. 같은 6 geometry에서
  |B| 일반화를 측정 -> 40모델 12.7%와 직접 비교. split은 수동 생성(고정 test/val
  + digest=manifest_digest(120)), resolve_case_split이 재생성 않도록 사전 배치.
- 재학습: `--data-dir backup/doe_data_120 --split eval/splits/doe120_case_split.json`,
  bw30 구성(target B, h256/p15, BAND_W=30). 게이트는 기존 40 dir로 별도 확인.

## 준비도 (확인됨 2026-07-29)
- base .mot: 존재(D:\KDH\Sim_4SolverX\TestCAD1.mot) ✓
- PyMotorEnv_310: C:\Users\moa\.ansys_python_venvs\PyMotorEnv_310, pymotorcad 0.8.4 ✓
- Motor-CAD v261 설치 ✓ / doe_batch.py·sweep.py ✓
- 미확인: 라이선스(1단계에서 확인).
