# MotorAI 다형상·전이학습·멀티모달 연구

이 폴더의 정식 저장소는 **enjoyneer87/EveryMotor**, 연구 브랜치는 `codex/motorai-multisolver-20261028`이다. 사용자 요청으로 MotorAI 계열4개 연구 폴더를 JEET-repro의 작업에서 이전했다. 기존 JEET 커밋·원자료 반환 이력은 보존하며 신규 모델 개발은 여기서 이어간다.

- [개선 요청·아이디어 통합 대장](phase2_multigeometry_turns_20261027/IMPROVEMENT_REGISTER_KO.md)
- [Astra 전이학습·멀티모달 판정](phase2_multigeometry_turns_20261027/ASTRA_TRANSFER_MULTIMODAL_KO.md)
- [현재 현황판](phase2_multigeometry_turns_20261027/STATUS_KO.html)
- [확대 학습 결과](phase2_multigeometry_turns_20261027/SCALE360_REPORT_KO.md)
- [기존 MotorAI 점수·모델 계보](phase2_multigeometry_turns_20261027/PRIOR_MOTORAI_COMPARISON_KO.md)
- [E1 대비 최근 모델 성능 차이 진단 · Astra xHigh](performance_diagnosis_20261028/REPORT_KO.md)
- [A2 최종 판정·C/G 첫 개선 실험 · Astra Max](next_improvement_20261028/REPORT_KO.md)
- [동결 새OP 손실 비교](new_op_validation_20261027/VALIDATION_REPORT_KO.md)

이전 범위: `mgn_development_20261027`, `mgn_followup_20261027`, `new_op_validation_20261027`, `phase2_multigeometry_turns_20261027`의 코드·작은 모델/파형·검증·요청 대장과 필요한 baseline/reader/관측 참조. 각 파일의 원본·이전본 SHA-256은 MIGRATION_MANIFEST.json으로 대조한다.

JEET 마지막 공개 스냅샷은21afcb4016aa735d18a997a141849ef75df8c14f다. 그 이후 로컬 사전선언·reader/평가 코드와 미게시 감사 결과도 파일 hash로 포함했다. 소스의 옛 commit/issue 링크는 역사적 provenance로 남긴다. 원본 결과의 scope·미확정 상태를 변경하지 않는다.

기존 리더는 동일한 상대 폴더 구조를 유지했다. `remote_return`의 원시/샘플 binary는 중복 이관하지 않았으므로 해당 입력이 필요한 원시 재현은 pinned JEET source 또는 private source-key mapping에서 확보해야 한다. 대용량 HDF5/graph zip/E1 weights도 외부에 보존한다. 누락 dependency 목록은 manifest의 omitted_reference_artifacts에 있다.

현재:320형상그룹360조건의133 학습30epoch/88565updates 완료,dev32 B벡터18.0765%,구리34.6131%. 이전E1의전체|B|11.1%/공극4.96%와는지표·분할이달라직접비교하지않는다. 가중치전이선행감사는prior/전체donor reserved노출이UNKNOWN이므로아직HOLD다. 새OP8중6회수,wholecurrent PASS5/FAIL1,SC2미실행. formal수렴UNKNOWN,exactbridge0.

업스트림 모델·평가 코드는 source commit65dda206e28770b242fa0dd888fd394f7a155ca0 위에 별도 연구 스냅샷으로 추가했다. 기존 EveryMotor canonical 작업 트리와 캠페인을 수정하지 않는다. 새 canonical model benchmark 통합은 root eval/계약을 따른다.
