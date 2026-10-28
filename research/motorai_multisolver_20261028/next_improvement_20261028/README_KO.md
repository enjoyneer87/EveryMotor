# C/G 정보 경로 개선 실험

[현재 상태](STATUS.json), [A2 완료·C/G 보고서](REPORT_KO.md), [Astra Max 설계](ASTRA_MAX_REVIEW_KO.md), [최종 A2 분기 판정](ASTRA_A2_FINAL_REVIEW_KO.md), [동결 계약](CONTEXT_CG_PLAN.json)을 따른다.

원 graph·component NPZ·checkpoint는133에 보존하며 logical source hash로 연결한다. `COMPONENT_MANIFEST.json`은360조건의 native element/edge hash와 sidecar hash를 제공한다. `artifacts/COLLECTION_RECEIPT.json`은 회수한 작은 결과/CSV/소유 receipt의 원본 hash를 제공한다. `PACKAGE_MANIFEST.json`은 공유 파일 자체의 무결성을 검증한다.

검사와 실행 코드는 Python3.12.10/Torch2.5.1+cu124/NumPy1.26.4의 기존133 환경을 사용한다. 신규 설치·새 FEA·새 prior solve는 없다. source 원본 `dual_solver_mgn.py`, 동결 split/field/pilot manifest와 소유 guard는 인접 `../phase2_multigeometry_turns_20261027/`에서 배포 시 SHA 검증했다. 경로 매핑과 `RUN_PRIVATE.json`은 Git에 넣지 않는다.

소유 원격 stage에서 사용하는 명령은 다음과 같다. 원래 실행은 동결 guard가 관리하므로 이미 실행한 stage를 다시 시작하지 않는다. 다른 로컬의 독립 재현은 같은 원자료 hash와 새 실행 계약을 먼저 만든다.

```powershell
python test_context_contract.py
python test_context_checkpoint.py
python train_context.py --bundle <admitted_train_bundle> --components <native_component_sidecars> --out <new_output> --arm C --seed 17 --phase-encoding mechanical --prefetch 8
python evaluate_context.py --bundle <frozen_dev32_all45_bundle> --components <native_component_sidecars> --checkpoint <final_last.pt> --expected-sha256 <checkpoint_sha256> --out <new_eval_output> --anchor-result <train_RESULT.json>
```

위 명령에서 C/G·seed·입력·update·LR·final checkpoint는 계약대로 대응시킨다. 파일·native ID 불일치나 누락은 UNKNOWN/FAIL로 보존한다. 개발 집합 성적을 최종시험·공극토크·AC손실 정확도로 확대하지 않는다. E1 weights/prior/teacher는 별도 admission까지 HOLD이며 이번 C/G에 섞지 않는다.
