# 새 운전점 원자료 검증 반환

요청8건 중 6건 회수. 회수 중 품질FAIL 1건, 선행 gate 때문에 미실행 2건, 실행중 0건, 순차대기 0건이다. 실행 완료와 품질 적합성을 구분하며 미회수를 자동으로 실패로 세지 않는다. 입력·예측은7f04178에서 해석 전에 동결했다. 보정 Full63(Ref36+SC27), 새 평가8건은 모든 방법이 공유하며 재학습·튜닝·best-seed 선택은 없다.

| 조건 | 채택 P2 AC excess W | 전체 전류 최대오차 % | P2/P3 평균차이 % |
|---|---:|---:|---:|
| Ref_I230.05_B36_N6000 | 2229.977374 | 0.000126224 (PASS) | 0.00000306 |
| Ref_I230.05_B36_N12000 | 7664.773513 | 0.000255802 (PASS) | 0.00000225 |
| Ref_I460_B36_N6000 | 7276.482893 | 0.000083029 (PASS) | 0.00000135 |
| Ref_I460_B36_N12000 | 25376.616002 | 0.000163858 (PASS) | 0.00000102 |
| SC_I460.1_B36_N6000 | 24497.041125 | 0.000515081 (PASS) | 0.00008200 |
| SC_I460.1_B36_N12000 | 70881.439991 | 0.001101197 (FAIL) | 0.01179092 |

native A와 회로전압에서 BE 첫 스텝/BDF2 이후의 dA/dt, P1 삼각형 내부 J 적분, prescribed transport DC를 재계산했다. 전체모터 활성부 Joule에서 DC를 뺀 AC excess이며 end winding은 포함하지 않는다. P2는257–512 post-step 산술평균이다. P1과P3는 진단이며 채택구간을 바꾸지 않았다. 저장 파형 재평균과 audit의 오차는 quality_summary.json에 기록했다. 이는 동일 native evaluator의 산술 일관성으로 독립 solver loss채널 검증은 아니다.

전류<=0.001%와 비선형<=1e-10은 기존 gate다. P2/P3 개발 threshold와 공식 주기·시간간격·mesh 수렴 UNKNOWN을 구분한다. conductivity는47092391.429245 S/m, active length0.15m, symmetry8이다. 온도는 receipt에 명시되지 않아UNKNOWN이며 conductivity로 임의 역추정하지 않았다.

| 방법 | 회수 조건 전체 wMAE % |
|---|---:|
| latest_AF | 3.264468 |
| scalar | 1.781971 |
| MS_MGN_seed17 | 1.004644 |
| MS_MGN_seed29 | 1.007066 |
| MS_MGN_seed43 | 0.885627 |
| A_curl_seed17 | 1.090199 |
| A_curl_seed29 | 0.767532 |
| A_curl_seed43 | 1.256347 |

품질FAIL을 포함한 부분 진단 결과다. P2 전류가통과해도전체이력의FAIL을승격하지않는다. CURRENT_GATE_DIAGNOSTICS.json에초과step과P2를구분했다. 조건수와 MS source group이 적어 우월성이나 새로운 형상 일반화를 주장하지 않는다. 모든3시드의 결과와 평균/표준편차는 EVALUATION.json에 보존한다. Ref는 원래 Ref MS의 주파수 재사용, SC는 Ref MS를 스케일링한 분모 연결이므로 native SC MS/Full 쌍으로 세지 않는다. exact cross-solver bridge는0이다.

재현 명령: `python evaluate_frozen.py`, `python build_return.py`, `python update_dashboard.py`. 이 명령들은 solver와 학습을 실행하지 않는다. 대용량 native field는 iso 원본을 보존하고 GitHub에는 파형·hash·코드만 둔다. 상세 source key와 파일별 hash는 각 results의 raw_source_manifest.json에 있다.

별도 Phase2의4/6/8 슬롯당 도체, 총128스텝 요청은 ../phase2_multigeometry_turns_20261027/에서 수행한다. 기존8건의 동결된 N256x3주기 비교 계약은 변경하지 않는다. 신규 Phase2 labels와 기존 평가 labels를 혼용하지 않는다.
