# 새 운전점8조건 반환

현재 원시 적분 회수 6/8. VALIDATION_REPORT_KO.md, STATE.json, STATUS_KO.html 및 source_manifest.json을 함께 확인한다.

실행: `python evaluate_frozen.py` → `python build_return.py` → `python update_dashboard.py`. 원시 field는 보존된 source key 경로에서만 접근한다.

남은 작업: 미완료 조건 회수, 같은 frozen prediction 평가, 누락/실패 포함 최종 집계. 공식 수렴은 UNKNOWN이며 새 Phase2의 총128스텝 진단과 별개다.
