# 과학 축 재베이스라인 게이트 — 판정 (2026-09-02 실행 / 2026-09-08 판정)

> **실행 머신: `DESKTOP-513NQ4P` / `192.168.0.234`** (격리 PC, RTX 3090 24 GB).
> PhysicsNeMo 컨테이너 `physicsnemo:26.03`, `/workspace/app` 바인드 마운트.
> 트리: `iso/integrate-cli-native-20260901` @ `d23f6a9` + 미추적 게이트 스크립트.
> 판정 작성: 외부망 PC 에서 `ssh iso` 로 산출물을 읽어 수행.

## 1. 이 게이트가 답하는 질문

`5e2a97b` 로 병합한 `feature/cli-native-training`(HPC_134 작성)은 두 축을 함께 가져왔다.

| 축 | 내용 | 처리 |
|---|---|---|
| 런타임 축 | 네이티브 실행 전환, DGL 경고 억제, `.gitattributes`, CLI 배선 | 그대로 병합, 켜둠 |
| **과학 축** | 채널 z-score 정규화, Je 크기인식 focal 가중 | `c4acab4` 로 **기본값 보수형**(`normalization=none`, `alpha=0.0`), 채택은 이 게이트로 판정 |

즉 질문은 하나다 — **패키지 트레이너에서 z-score 정규화가 실제로 도움이 되는가.**

## 2. 무엇을 돌렸나

`_gate_chain.sh` (이 커밋에 동봉). 두 arm 은 아래 `COMMON` 을 공유하고 두 플래그만 다르다.

```
COMMON = --input-format doe --data-dir backup/doe_data_120
         --case-indices <doe120 split 의 train 110개>
         --epochs 50 --batch-size 1 --seed 42 --ckpt-interval 10 --weight-current 0

arm A (대조)  --target-normalization none   --current-focus-alpha 0.0
arm B (처리)  --target-normalization zscore --current-focus-alpha 2.0 --current-focus-gamma 1.0
```

평가는 학습에 쓰지 않은 **test 6 케이스**(4, 7, 18, 32, 37, 39)를 `infer_phase1_pbc.py` 로 추론한 뒤
`_gate_aggregate.py` 로 풀링. 지표는 채널별 `rmse` 와 **`rel = RMSE / std(gt)`** 다.
**`rel = 1.0` 은 "평균을 예측한 것과 같다" = 스킬 0 을 뜻한다.**

소요: arm A 22,580 s / arm B 22,948 s, 둘 다 `rc=0`. 체인 완료 `2026-09-02T16:32:36+00:00`.

### Je 채널에 대한 주의 — 두 arm 모두 무효다

이 DOE 의 Motor-CAD 내보내기에는 `fields/je` 가 없어 로더가 0 으로 채운다. 그래서
`--weight-current 0` 을 **`COMMON` 에 두어 양쪽 arm 에서 Je 항을 손실에서 제거**했다.
arm B 의 `--current-focus-alpha 2.0` 은 그 제거된 항 안쪽의 가중이라 **무연산**이다.
컨테이너 로그가 `current=` 를 따로 찍지만 `total` 에는 들어가지 않는다
(arm A: `total=0.309326 = a 0.000138 + b 0.309189`, `current=7.804323` 는 별도).

**따라서 이 게이트는 Je focal 가중을 판정하지 않는다.** `fields/je` 가 실린 내보내기가 생기기
전까지 그 축은 미판정으로 남는다.

## 3. 결과

### 3.1 학습 손실 (각 arm 자신의 공간에서)

| arm | epoch 50 train_total | train_a | train_b | val_total | best_val |
|---|---|---|---|---|---|
| A | 0.309326 | 0.000138 | 0.309189 | 0.308665 | 0.308604 (ep49) |
| B | 2.012097 | 1.014505 | **1.001782** | 1.990207 | 1.985066 (ep48) |

### 3.2 test 6 케이스 풀링 (2,005,843 절점, 물리 단위)

| | bx rmse / rel | by rmse / rel | a rmse / rel | \|B\| rmse / rel |
|---|---|---|---|---|
| **A** | 0.5136 / **0.760** | 0.4995 / **0.686** | 0.01055 / 0.977 | 0.6910 / 1.197 |
| **B** | 0.6761 / **1.0002** | 0.7280 / **1.0000** | 0.01082 / 1.0023 | 0.9091 / 1.575 |

원시 데이터: `results/gate_summary.json`(케이스별 포함), `results/gate_{A,B}_summary.json`.

## 4. 판정

### z-score 는 채택하지 않는다. `c4acab4` 의 보수형 기본값을 유지한다.

근거가 **서로 독립인 두 줄**로 일치한다.

1. **평가 쪽** — arm B 의 `rel` 이 bx 1.0002 / by 1.0000 / a 1.0023 으로 스킬 0 선에 정확히 붙었다.
2. **학습 쪽** — arm B 의 `train_b` 가 **정규화 공간에서 1.0018** 이다. z-score 공간의 MSE 1.0 은
   그 자체로 "평균을 예측 중"이라는 뜻이므로, 평가 경로를 전혀 거치지 않고도 학습이 실패했음을 말한다.

두 번째가 중요하다. 평가만 봤다면 "정규화된 출력을 원시 타깃과 비교해서 생긴 착시" 가능성이 남는데,
학습 손실이 자기 공간에서 이미 1.0 이므로 그 설명은 배제된다. **arm B 는 진짜로 학습에 실패했다.**

## 5. 이 게이트가 **말하지 않는** 것

같이 나온 숫자를 과잉 해석하지 않도록 명시한다.

- **arm A 도 쓸 수 없다** (\|B\| rel 1.197). 하지만 이것을 "패키지 트레이너가 측정 계보보다 못하다"로
  읽으면 안 된다. **비교 조건이 다르다**:

  | | arm A (패키지 트레이너) | `doe120_bw30` (측정 계보) |
  |---|---|---|
  | 스크립트 | `phase1_static/train.py` | `train_doe_curl_mgn.py` |
  | 폭/깊이 | h**128**/기본 (미지정 → default) | h**256**/p15 |
  | 감독 | `[Bx, By, A, Je]` 직접 | curl(A) vs B_fem + 게이지 + airgap + **band-spectral(bw30)** + PBC |
  | 결과 | \|B\| rel 1.197 | \|B\| nRMSE **11.96%** / 토크 5.25% |

  용량도 손실 구성도 다르므로 이 표의 두 열은 **비교 대상이 아니다.** 두 트레이너의 우열을 말하려면
  같은 폭·같은 손실로 다시 재야 한다.

- **`train_a`(=A 채널) 는 arm A 에서 0.000138 로 매우 낮은데 평가 `rel` 은 0.977 이다.** 손실이
  작은 것과 스킬이 있는 것이 다르다는 전형적인 예다 — A 의 절대 크기가 작아 MSE 가 작을 뿐,
  `std(gt)` 대비로는 설명력이 없다. 다음 실험 설계 시 A 채널 손실 가중을 근거로 삼지 말 것.

- 50 epoch / batch 1 / lr 1e-3 고정이고 LR 스케줄·조기종료가 없다. arm A 의 `val_total` 은
  ep43~50 구간에서 0.3086~0.3117 사이를 오르내리며 **평탄**하다(과적합이 아니라 미적합).
  더 돌리면 나아질 여지는 남아 있으나, 그것은 이 게이트의 질문이 아니다.

## 6. 후속

| # | 항목 | 비고 |
|---|---|---|
| 1 | **`_gate_aggregate.py` 의 정규화 arm 처리 확인** | `infer_phase1_pbc.py` 가 정규화 arm 출력을 역변환하는지 코드로 확정할 것. 이번 판정은 학습 손실로 독립 확인했으므로 영향 없지만, 이 하니스를 정규화 arm 에 재사용하면 문제가 된다 |
| 2 | **Je focal 가중 판정** | `fields/je` 가 실린 내보내기 이후. 설계 노트 §4.5 (고조파 주입 케이스)와 짝 |
| 3 | **트레이너 이원화** | 추적 중인 러너 `results/logs/*.sh` **19개 중 16개가 `train_doe_curl_mgn.py` 를 부르고, `phase1_static.train` 을 부르는 것은 0개**(2026-09-08 실측). §5 의 비교 불가 문제의 뿌리 |
| 4 | 네이티브 arm 동등성 | `HPC_134` 몫 (AGENTS.md: 134 는 Docker 불가) |

## 7. 재현

```bash
# 컨테이너 안, /workspace/app
bash _gate_chain.sh          # arm A → 평가 → arm B → 평가 → 집계, 약 12.6 h
```

체인은 `results/gate_{A,B}.pt`, `results/gate_{A,B}_case<k>.npz`, `results/gate_{A,B}.log` 를
남기지만 전부 gitignore 대상이라 격리 PC 로컬에만 있다. 이 커밋이 담는 것은
**스크립트와 집계 JSON** — 즉 판정을 재검토하는 데 필요한 최소 집합이다.
