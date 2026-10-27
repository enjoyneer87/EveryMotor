# Ref_P117 수치 민감도와 원시 적분

상태: **COMPLETE**, 계획한3개 수치 조건 중3개 완료. 실제 물리 운전점은1개이며 새 독립 학습 정답은0개다.

기존 보정점 Ref_P117, 16,000rpm·460Arms·beta36°에서 원래 mesh/N256, 원래 mesh/N512, RefFine/N256을 각각3주기로 계산한다. 초기 A=0, 첫 스텝 BE 이후 BDF2, sigma47092391.429245S/m, 활성길이0.15m,8sector를 유지한다. P2는 원래의 균일 post-step 평균이며 전체 이력 평균으로 바꾸지 않는다.

**판정 범위:** current0.001%와 nonlinear1e-10은 기존 채택 규칙이다. 주기 평균0.1%·파형L2 0.5%, 시간/메시 평균1%는 계산 전에 기록한 개발용 기준이다. 별도의 승인된 수렴 계약이 없으므로 공식 주기·시간·메시 qualification은UNKNOWN으로 남긴다. 아래 개발 기준 충족을 전체 물리 검증PASS로 확대하지 않는다.

| 수치 조건 | 스텝 | P2 AC(W) | P3 AC(W) | P2/P3 평균 차(%) | P3 전류 최대오차(%) | 계산시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| N256_P3 | 768 | 41600.733263414 | 41600.733870889 | 1.4602489e-06 | 1.6589473e-05 | 3789.83 |
| N512_P3 | 1536 | 41492.406176488 | 41492.406905650 | 1.7573401e-06 | 4.9403728e-05 | 7777.29 |
| fine_N256_P3 | 768 | 41574.961288405 | 41574.961899533 | 1.4699419e-06 | 1.6606412e-05 | 4986.23 |

전류 오차는 순간 전류가 아닌 prescribed phase peak(sqrt(2)×Irms)에 대한 최대 비율이다. P2/P3 waveform L2와 전체 이력 current 상태는 CONVERGENCE_RESULTS.json에 별도 보존한다.

원래 P2 저장 scalar의 새 계산 재현 최대 차이는 **1.18e-08W**, 기존1e-6W gate에 대해PASS다. 이는 고정 label 재현 검사이며 다른 수치 해상도의 차이를 오류로 취급하지 않는다.

첫 주기 AC 평균은 P2보다21.729% 높다. startup이 포함된 whole-history 평균을 채택 P2 대신 쓰지 않는다.

time_step: P3 평균 차이 **0.261077%**, 위상을 맞춘 파형L2 차이0.280138%. 개발 기준PASS; 공식 qualification UNKNOWN.

mesh: P3 평균 차이 **0.061989%**, 위상을 맞춘 파형L2 차이0.097283%. 개발 기준PASS; 공식 qualification UNKNOWN.

새 scalar/current 시간 이력은 native A와 회로 전압에서 BE/BDF2를 적용해 재구성했다. 저장8위상의 J 재적분과 이 이력의 일치는 shard·reader 일관성 검사이며 별도의 독립 solver 손실 채널 검증이라고 주장하지 않는다. 독립적인 과거 채택 scalar와의 연결은 위 원래 P2 재현 검사이며, 각 시각의 전류는 prescribed current와 직접 비교했다.

## 메시·필드·보존

RefFine는 기존 검증된 input의 물성·회로·solver를 복사하고 메시만 교체했다. 구리36영역의 면적 차이 최대1.65e-13%, 중심차1.25e-16m, 외접범위 차0m다. 전체 physical domain 면적 차이는 최대0.00794%로 보고하며 연속 곡면의 완전 동일성을 증명했다고 하지 않는다. 좌표 최근접으로 ID를 복구하지 않았다.

대용량 native/VTU/solver log는 원 계산 위치에 보존하고 모든 파일의 SHA256·크기를 raw_source_manifest.json으로 연결했다. 공개 shard는 원래 Cu node/element/회로 ID, 연결 관계, 좌표, 면적/체적, P3의8위상 A/B/total J와 전체 시간 scalar/current 파형을 포함한다. float64 NPZ 압축은 무손실이다. sparse8의 평균은 Full period loss label로 사용하지 않는다.

N256_P3:8위상 J 재적분과 원 scalar의 최대 차이0W, current 차이0A. 8위상만 평균하면 AC 평균이 전체 P3와1.8039% 다르므로 전체 파형을 함께 제공한다.

N512_P3:8위상 J 재적분과 원 scalar의 최대 차이0W, current 차이0A. 8위상만 평균하면 AC 평균이 전체 P3와1.7886% 다르므로 전체 파형을 함께 제공한다.

fine_N256_P3:8위상 J 재적분과 원 scalar의 최대 차이0W, current 차이0A. 8위상만 평균하면 AC 평균이 전체 P3와1.8103% 다르므로 전체 파형을 함께 제공한다.

## 실행 이력과 한계

첫8스텝 pilot은 MPI 초기화 단계에서 멈췄고 물리 step을 계산하지 않았다. 원본 실패 기록을 보존하고 single-rank MPI 환경을 단독 검증한 뒤 새 root에서 pilot8을 통과했다. 이후 생산 case는3개로 한정했고 재시도하지 않았다. 실제 속도·메모리 측정 후 CPU영역을 분리해 실행했으며 누적 case 경과시간의 최종 상한은17,000초다. 원래 주행 프로세스는 중단·변경하지 않았고 이 작업의 coordinator만 소유 PID 확인 후 교체했다.

후속 독립 템플릿, SC/HalfSC 고부하와 저전류, 모든 운전점의 수렴은 남아 있다. 과거 HalfSC의 period_convergence_qualified=false와 mesh_convergence_qualified=false는 그대로다. 현재 원시 검증이 완료된 canonical MS/Full 대응41개 고유 조건에 이 동일 앵커의 수치 변형을 더해 표본 수를 늘리지 않는다.
