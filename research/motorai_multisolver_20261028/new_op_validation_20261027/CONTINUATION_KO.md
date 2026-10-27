# 새 운전점 8건: 2026-10-28 08:00 KST 종료

이 폴더는 기존 연구와 A/curl 후속 모델을 수정하지 않고 새 Full-FEA 운전점에서 비교한다. PLAN.json의8조건, PREDICTIONS_FROZEN.json의8방법 예측은 새 정답을 보기 전에 동결했다. Full 보정예산은 Ref36+SC27=63, 평가예산8은 모든 방법에 동일하다. 재학습·재튜닝·best-seed 선택을 하지 않는다.

## 현재 단계와 계속할 순서

STATE.json과 실제 소유 프로세스·receipt를 먼저 확인한다. 표시된 준비/실행 상태는 실측 확인 뒤에만 바꾼다. 이미 실행된 케이스를 다시 시작하지 않는다. 기존3개 수렴 해석과 133의3seed 학습은 완료됐으므로 재실행하지 않는다.

20:08 KST에 `Codex-JEET-NewOP-20261027-Ref`와 `Codex-JEET-NewOP-20261027-SC` 일회성 Windows 작업으로 시작했다. 두 pilot 통과 후 생산2건이 실행 중이다. 트리거/자동재시작은 없다. 아래1번의 시작 절차는 이미 완료됐으므로 반복하지 말고 실제 상태 확인부터 이어간다. 현재 중단/수집 경로는 private Ref.json/SC.json 및 supervise.py를 확인한다.

1. run_queue.py와 입력 manifest를 검증한 다음 두 작업자의8-step pilot을 확인한다. 작업자별4조건, CPU8 threads, 생산 case당5400초·pilot180초·전체43560case초·재시도0·08시 UTC마감이다. pilot current/nonlinear 실패 시 해당 작업자를 중지한다. 소유 정보와 private 경로 매핑은 로컬 outputs/mgn_development_20261027/new_op_control에 둔다.
2. 각 케이스 execution.json, derived/audit.json, waveforms.npz를 회수한다. 원래 raw field는 iso에 보존한다. 모든 raw 파일의 상대경로/hash/크기와 논리 source key를 기록하고 INPUT_MANIFEST와 입력 hash를 대조한다. 입력, mesh, solver, 회로 라이브러리 hash가 틀리면 격리한다.
3. native A와 회로전압의 BE/BDF2 재적분 결과다. P2 steps257..512 post-step arithmetic mean을 정답으로 쓰고 P3는 주기 검사로 보존한다. Joule/DC/AC, prescribed 전류오차, P2/P3 민감도를 각각 보고한다. solver 자체 독립 loss채널과 같다고 표현하지 않는다. current<=0.001%, nonlinear<=1e-10의 기존 gate와 제안 period개발 threshold를 구분한다. formal mesh/time/period UNKNOWN을 유지한다.
4. 동결 예측과 P2 AC_W를 case ID로만 연결한다. AF/scalar/primary3seed/A-curl3seed의 모든 조건별 오차, Ref/SC/전체 wMAE와 seed 통계를 산출한다. 실패·누락8건모두 분모 집계에 포함하고 실제 metric에 들어간 조건 수를 명시한다. 2개MS그룹·기존형상의새속도8점이지 새템플릿 일반화나 미개봉대규모시험이 아니다.
5. 기존41개 MS/Full 쌍에 새 검증 완료8건을 무조건 더하지 말고 unique source/condition으로 확인한다. cross-solver exact bridge는0이다. reserved 두형상과 alias, HalfSC32/탐색4, 기존 마스크를 보존한다. 새8건은 평가전용이며 새학습정답이 아니다.
6. README_KO.md, source_manifest.json, observations.csv, quality_summary.json, split_audit.json, 검증보고서와 STATE.json/STATUS_KO.html을 갱신한다. 작은 파형·manifest·코드만 게시하고 원field/절대경로/자격정보는 제외한다.
7. primary/followup manifests와 verify_package도 갱신한다. 이 세 연구폴더만 명시적stage/commit. remote FETCH_HEAD를 확인하고 fast-forward push한다. 충돌이면 force 없이 별도 반환 브랜치. 이슈#1에 실제 완료·실패·누락·Full예산·과학적 한계와 링크를 남긴다.

회수용 `collect_case.py --case-dir <private-worker>/<case> --inputs-manifest INPUT_MANIFEST.json --out <new-small-export>`는 완료된 worker의 raw와 파형만 읽으며 solver를 실행하지 않는다. native audit 후의 파형 평균, frozen 입력 hash를 대조하고 원시 파일의 전체 hash/크기를 기록한다. 결과를 이 폴더의 `results/<case>/`로 복사한 뒤 `python evaluate_frozen.py`를 실행한다. 모든 시드와 미회수 수를 출력하며, 미회수를 자동으로 실패라 분류하지 않는다. 실행·중단 receipt에 근거해 별도 실패/대기 집계를 함께 완성해야 한다. 현재 작성된 후처리 코드는 구문 검사만 통과했으며 실제8건 결과로의 실행 검증은 완료 후 수행한다.

## 종료

2026-10-27T23:00:00Z 이후 새 연구 계산을 하지 않는다. 소유 작업에 STOP을 주고 PID/start-tick을 대조해 소유 process group만 종료하며 부분 파일을 보존한다. 결과 게시 후, 또는 일찍 완료되면 새 heartbeat `jeet-mgn-10-28-08`만 PAUSED로 바꾼다. 기존 PAUSED 자동화, EveryMotor 캠페인, 사용자 AEDT는 변경하지 않는다.

## 현황판

`python update_dashboard.py`는 STATE.json으로 STATUS_KO.html을 갱신한다. 실시간 자동갱신 화면이 아닌 시각이 표시된 스냅샷이다. 실행 중인 개수를 추정해서 입력하지 않는다.

앱과 별개로 iso OS의 `Codex-JEET-NewOP-20261028-0800-Cutoff` 작업이 08:00 KST에 소유 STOP 및 PID/start-tick 확인 종료를 수행한다. OS_CUTOFF_RECEIPT.json에 트리거 확인과 코드 hash를 기록했다. 조기 완료 후에는 소유 작업 Ref/SC 및 이 cutoff task만 Disabled로 바꾸고 receipt를 보존한다.

21:05 KST 갱신: iso Windows 시각이 관리 PC보다 약17.4~18.8초 늦음을 관측했다. 서버 시각을 바꾸지 않고 소유 OS cutoff만07:59 KST로1분 앞당겼다. 원래08:00 상한과 native 러너 동결은 유지한다. 최신 근거는 OS_CUTOFF_MARGIN_RECEIPT.json이며 초기 OS_CUTOFF_RECEIPT.json은 이력이다.
