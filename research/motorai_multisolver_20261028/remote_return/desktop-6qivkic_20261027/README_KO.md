# Elmer 원자료 반환 — desktop-6qivkic / iso, 2026-10-27

MGN, scalar 잔차, 다중 소스 및 Field 전이 기법 개발을 위한 반환물이다. **iso에서 MS24와 HalfSC36 원자료를 모두 찾고 읽기 전용으로 검증했다.** 이번 갱신은 기존 반환 commit `156e80a0503e02ceb64e6ce327fab6a096ddf6fc`의 결손을 보완한다. 최신 AF 원천 객체는 `0034d7c24b1cc0cf41499d028733792b14815bc7`이며 계수·분모·마스크를 변경하지 않았다.

| 대상 | 확인 결과 | 해석 범위 |
|---|---|---|
| MS Ref24 | A·manifest·receipt·입력·mesh·log 원본 hash 확인, 24/24 B 캐시 bitwise 일치 | 128위상 × 600표본; 물리 정확도 판정 아님 |
| canonical Hybrid | Ref/SC192 + HalfSC36 재현 | 최대 차이 3.6379788e-12 W |
| HalfSC36 | native512 step에서 Joule/DC/AC·36도체 전류 재적분 | 사전 선언32와 exploratory4 유지 |
| HalfSC provenance | 사전 선언32 frozen archive-root hash 일치 | exploratory4 원시 member/scalar 연결 확인, 별도 보존 receipt/root hash UNKNOWN |
| 원시 Full 직접 재적분 | 누적42 = 기존6 + HalfSC36 | latest192 중5, 저전류 진단1, HalfSC36 |
| MS/Full 원자료 계보 쌍 | 41 = latest192의5 + HalfSC36 | homogenization·speed·geometry scaling을 포함한 evaluator 연결 |
| 물리 형상·OP가 모두 exact인 교차 솔버 bridge | 0 | 원자료 회수로 exact 승격하지 않음 |

HalfSC P2 전류는 {'PASS': 36}, whole-history 전류는 {'PASS': 33, 'FAIL': 3}다. HalfSC AC 재현 최대 차이는 2.6193447e-10 W다. **period·mesh 수렴은 UNKNOWN**이며 `qualified=false`를 유지한다. 새 FEA·모델 학습은 허용되었으나 이번 결손은 기존 자료로 해결하여 둘 다 0회 실행했다. 이전 자동화와 캠페인은 재개하지 않았다.

## 바로 읽을 파일

- [comparison_ready.csv](comparison_ready.csv): 228조건의 Full/canonical Hybrid/frozen AF, 그룹·마스크·양쪽 raw 검증 여부.
- [iso_recovery/MS_Full_pairs.csv](iso_recovery/MS_Full_pairs.csv): source current 차이, 원천 speed, target scale, 41개 양쪽 raw 검증과 공통 MS 원천 그룹.
- [observations.csv](observations.csv): 1,668개 quantity별 관측. 추가72행은 HalfSC 기존36조건의 Joule/DC 물리량이며 새 독립 표본72개가 아니다.
- [iso_recovery/HalfSC36_waveforms.npz](iso_recovery/HalfSC36_waveforms.npz): 36×512 손실·전류 이력. 원 시간축과 P2 평균을 보존한다.
- [iso_recovery/MS_validation.json](iso_recovery/MS_validation.json), [HalfSC_raw_audits.json](iso_recovery/HalfSC_raw_audits.json), [HalfSC_lineage.json](iso_recovery/HalfSC_lineage.json): 수치·출처의 단계별 근거.
- [source_manifest.json](source_manifest.json), [sample_shards_manifest.json](sample_shards_manifest.json): 논리 키, 원본/추출 hash·크기, 배열 정의와 보존 위치.
- [training_policy.json](training_policy.json), [split_audit.json](split_audit.json), [MGN_DATA_CONTRACT_KO.md](iso_recovery/MGN_DATA_CONTRACT_KO.md): 사용·격리·누출 및 다음 모델 비교 계약.
- [VALIDATION_REPORT_KO.md](VALIDATION_REPORT_KO.md), [quality_summary.json](quality_summary.json), [case_queue.csv](case_queue.csv): PASS/FAIL/UNKNOWN, 남은 결손과 다음 계산 후보.

## FE graph와 재현 명령

Python 3.10+와 NumPy로 reader/샤드/반환물 테스트를 실행한다. canonical MS 표본 재현에는 matplotlib도 필요하다. 기존 iso 환경을 사용했고 패키지를 설치하지 않았다.

```powershell
python -m unittest -v test_return.py
python validate_shards.py
python -c "from pathlib import Path; from validate_shards import validate; validate(Path('iso_recovery'))"
python -c "from load_shard import load_copper_graph; d=load_copper_graph('shards/Ref_P117_field_snapshots.npz'); print(d['edge_index'].shape)"
```

MS 공개 샤드2개는 stationary 균질 슬롯의 **1,377 node / 2,191 triangle / 7,124 directed edge / 128위상**이다. Full 공개 공간 샤드는 기존6 + HalfSC3이며 실제 구리 FE 연결을 사용한다. 이전 36개 도체 중심 kNN과 다르다. 전체 모터 MGN 학습 완료를 의미하지 않는다. HalfSC predeclared32의 field도 보호된 평가 자료다.

원시 MS24 A와 원래 metadata는 이 로컬 및 iso에, 대용량 HalfSC native는 iso에 보존했다. 공개 Git에는 전체 대용량 원시 Field와 절대 경로를 넣지 않았다. `local_path_map.json`과 이번 `iso_recovery/private_paths.json`은 Git 밖의 개인 경로 매핑이다.

아래는 private request의 source 경로와 새 output을 지정하는 재현 순서다. request에는 기존 INPUT_INDEX_60의 선택24와 HALFSC_POINTS_36, 실제 source root만 넣는다. source 폴더와 output 폴더를 분리한다.

```powershell
python iso_recovery/recover_existing.py --request <private-request.json> --out <new-output-directory>
python iso_recovery/validate_ms.py --sources <baseline_audit/sources> --ms <recovered/ms> --out <new-ms-validation-directory>
python iso_recovery/merge_recovery.py --private <downloaded-private-recovery-directory>
python iso_recovery/write_recovery_docs.py
python refresh_artifacts.py
```

`merge_recovery.py`는 MS_validation 결과가 이 폴더에 놓인 후 실행한다. `build_return.py`와 `finalize_return.py`는 iso 회수 전 snapshot 재생성용이므로 갱신 후 단독 실행을 차단한다. 의도적으로 base를 다시 만들 때만 `--allow-base-rebuild`를 쓰고, 이어서 recovery merge와 artifact hash 갱신을 실행한다.

## 다음 연구

고정 AF 기준선과 같은 Full 예산의 개발 비교를 시작할 수 있다. MS pretraining 원천 그룹, 형상·OP 변환, 보호된 HalfSC32와 reserved R1_0199/R1_0076 제외를 지켜야 한다. 새 기준-template 일반화와 AF 대비 최종 우월성에는 독립 template, 품질-qualified target, 고정 예산·시험 protocol이 더 필요하다. 현재 약0.84% MS 교사 개발 오차를 Full AC 오차로 보고하지 않는다.
