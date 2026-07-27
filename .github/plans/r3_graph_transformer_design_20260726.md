# R3 설계 초안 — 전역 수용영역 아키텍처 (2026-07-26, 승인 대기)

> 상태: **초안, 승인 대기.** 스펙트럼 스윕(실험3) 결론 후 착수 예정.
> 가설(§15): 남은 |B| 오차는 용량이 아니라 **메시지패싱의 국소성**이다 —
> 에어갭 밴드에서 슬롯 피치가 ~65홉 떨어져 있어 15~24층 MGN이 슬롯 스케일을
> 한 번에 결합하지 못한다. 전역 수용영역이 원리적 해법.

## 0. 설계 원칙 — 단일 축 교체

실험1(폭)·2(깊이)와 동일한 규율: **아키텍처 하나만 바꾼다.** 나머지는 스윕
승자 구성을 그대로 고정한다 —
- `--target B`(nodal Bx,By → 요소 평균, out_dim=2, PBC/gauge 항 없음). 스펙트럼
  스윕이 쓴 구성.
- `--band-spectral-weight 30`(스윕 확정: §16. bw30이 |B| 동률에서 토크 최고
  5.219%, 실험1 대비 -1.59pp. bw1/bw10은 6.2~6.3%).
- `--no-wrap-rotor --no-anti-periodic-edges`, batch 1, 100 epochs, step-stride 1.
- 9-node / 4-edge 피처 계약 불변.

이렇게 하면 R3의 성패가 "전역 수용영역"이라는 단일 변수에만 귀속된다.

## 1. 접속점 (코드 조사 결과)

파이프라인은 모델을 블랙박스로 취급한다. 계약:
```
model(x_nodes:(N,9), edge_attr:(E,4), graph) -> raw:(N, out_dim)
```
- 손실(base 요소 MSE + **밴드 스펙트럼** + PBC)은 전부 `raw`/`b_pred`에서 계산 →
  **아키텍처 독립.** R3가 위 계약만 지키면 손실 코드 변경 0.
  (`train_doe_curl_mgn.py:628-659`, curl/평균은 `raw`만 받음.)
- **문제: 아키텍처 선택 seam이 없다.** trainer(`train_doe_curl_mgn.py:601`)와
  eval(`eval/predictors.py:404`)이 `MeshGraphNet`을 하드코딩. 체크포인트는
  하이퍼파라미터(`processor_size`,`hidden_dim`)와 I/O 폭만 읽고 **클래스는
  고정**해 재구성한다.
- 체크포인트는 리치 dict(`model_state_dict` + 정규화 통계 `x_mean/x_std/...` +
  `a_scale/out_scale` + `output_kind` + `node_features`/`edge_features` + `args`).

### 필요한 배선 (3곳, 최소 침습)
1. **trainer 플래그** `--model {mgn,transolver,bistride,hybrid}` (기본 mgn).
   구성 분기점 `train_doe_curl_mgn.py:601`.
2. **체크포인트 키** `"model_arch": args.model` 추가(`:720`, `output_kind` 옆).
3. **eval 분기** `CurlMeshGraphNetPredictor.from_checkpoint`
   (`eval/predictors.py:404`)에서 `ckpt["model_arch"]`로 빌더 선택. 하위(curl,
   de-scale, 요소 배치)는 전부 불변.

그래프 빌더(`build_curl_graph`)는 eval이 trainer에서 import해 재사용 →
R3도 동일 `CurlData`(curl 연산자·밴드 연산자·cut_pairs 부착)를 그대로 받는다.

## 2. 아키텍처 옵션 — 유지보수 구현 3종 + 커스텀

26.03 이미지에서 import 확인:
`physicsnemo.models`: `meshgraphnet`(MeshGraphNet, **BiStrideMeshGraphNet**,
**HybridMeshGraphNet**, MeshGraphKAN), **`transolver`**(Transolver).
국소성 병목을 공략하는 정도 순(작음→큼):

### (A) HybridMeshGraphNet — 장거리 "world" 엣지 추가 [보수적]
`forward(node_feats, mesh_edge_feats, world_edge_feats, graph)`. 메시 엣지와
별도로 **장거리 엣지 인코더**를 둔다. MGN을 유지하고 슬롯 피치 스케일의 명시적
장거리 결합만 추가 → 국소성을 직접 완화. 배선 비용: world 엣지 구성(예: 에어갭
밴드 노드를 슬롯 피치 간격으로 연결, 또는 반경/각도 기반 k-NN)을 그래프 빌더에
추가. 파라미터·비용 MGN과 유사. **가장 낮은 위험, 가장 작은 개념 도약.**

### (B) BiStrideMeshGraphNet — U-Net 다중스케일 MGN [중간]
`forward(node_feats, edge_feats, graph, ms_edges, ms_ids)`. bi-stride 풀링으로
그래프를 계층적으로 coarsen → 먼 노드가 적은 홉으로 연결(수용영역이 스케일마다
2배씩). 메시지패싱 패러다임 유지. 배선 비용: `ms_edges`/`ms_ids`(bi-stride BFS
풀링) 사전계산을 그래프 빌더에 추가(`num_mesh_levels=2` 기본). 어텐션 없이
국소성 해결. `bistride_pos_dim`에 pos(x,y) 사용.

### (C) Transolver — 물리-어텐션 그래프 트랜스포머 [헤드라인, §15의 "원리적 해법"]
`Transolver(functional_dim, out_dim, embedding_dim, n_layers=4, n_hidden=256,
n_head=8, slice_num=32, unified_pos=False, use_te=True, ...)`,
`forward(fx:(B,N,C_in), embedding:(B,N,C_emb), time) -> (B,N,C_out)`.
- **물리-어텐션**: N개 노드를 `slice_num`개 소프트 슬라이스(토큰)로 사영해
  어텐션 → **O(N·slice_num)** 선형, N²의 전역 어텐션 폭발 없음. 매 층이 전역
  수용영역. 비정형 메시에 설계된 PDE 연산자(arXiv:2402.02366).
- 엣지 미사용. 대신 어댑터로 계약을 맞춘다(§3).

### (D) 커스텀: 어텐션 증강 MGN [가장 통제된 실험, 최다 코드]
MGN 프로세서(국소 메시지패싱) 사이에 **전역 어텐션 층**을 삽입. "MGN + 전역
수용영역"이라는 단일 변수만 추가 → 방법론적으로 가장 깨끗하지만 커스텀 코드가
가장 많고, flash-attn을 써야 N² 메모리를 피함.

### 권고
**1차 = (C) Transolver** — 가설이 지목한 "전역 어텐션"의 직접 검정이자 유지보수
구현이라 커스텀 위험이 없다. **폴백/대조 = (A) Hybrid** — Transolver가 메시 구조
상실로 애매한 null을 내면, MGN을 유지한 채 장거리 결합만 더해 "국소성 자체가
병목인가"를 저비용으로 분리. (B)/(D)는 (C)·(A) 결과를 보고 판단.

## 3. Transolver 어댑터 (계약 정합)

```python
class R3Transolver(nn.Module):          # forward 계약을 MGN과 동일하게
    def __init__(self, in_node=9, out_dim=2, n_hidden=256, n_layers=8,
                 n_head=8, slice_num=32, emb_dim=64):
        self.pos_embed = nn.Linear(2, emb_dim)          # pos_x,pos_y → 위치임베딩
        self.core = Transolver(functional_dim=in_node, out_dim=out_dim,
                               embedding_dim=emb_dim, n_hidden=n_hidden,
                               n_layers=n_layers, n_head=n_head,
                               slice_num=slice_num, unified_pos=False,
                               use_te=False)             # 결정성 계약 위해 TE off
    def forward(self, x, edge_attr, graph):             # edge_attr 미사용
        fx  = x.unsqueeze(0)                            # (1,N,9)  이미 정규화됨
        emb = self.pos_embed(x[:, :2]).unsqueeze(0)     # (1,N,emb_dim)
        return self.core(fx, emb).squeeze(0)            # (N,out_dim)
```
- 정규화된 `x`(파이프라인이 `x_mean/x_std` 적용)를 그대로 fx로 사용 → MGN과
  동일한 입력 통계. de-scale(`raw*out_scale`)도 불변.
- **`use_te=False` 필수(초기).** AGENTS.md의 비트단위 재현 계약(결정성 게이트)
  때문에 Transformer Engine(fp8) 경로는 초기엔 끈다. 속도 손해는 감수, R3
  수용 조건에 "동일 ckpt 2회 채점 바이트 동일" 포함.

## 4. 파라미터 예산 (h256 MGN = 9.29M 기준)

용량 축은 §15에서 소진됨 → R3은 **동급 파라미터**로 맞춰 순수 아키텍처 검정.
Transolver 대략치(블록당 ≈11·d² : 물리어텐션 ~3d² + MLP 8d², d=256 → ~0.72M):
- n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4
- n_layers=8 → ≈6M / n_layers=10 → ≈7.5M / n_layers=12 → ≈9M(+임베딩/디코더)

→ **n_hidden=256, n_layers=10~12, slice_num=32**로 **~7.5–9M**, h256와 동급.

## 5. 3090 24GB 메모리 타당성 (batch 1, N≈6–8k 노드)

- **Transolver 물리-어텐션**: 활성 지배항 = N·d ≈ 8000·256 = 2.0M float =
  8MB/텐서/층. 슬라이스 어텐션 가중 N·slice_num = 8000·32 = 256k(무시가능).
  전 층·역전파 포함 수백 MB 규모. **N² 폭발 없음.** batch 1에서 넉넉.
- 참고: MGN h256/p15/batch1 실측 상주 ≈ 9.3GB(하트비트 관측, CUDA 컨텍스트
  포함) → 15GB 여유. Transolver는 그보다 가벼울 전망.
- (커스텀 D의 전역 어텐션은 N²=64M 스코어 → **flash-attn 필수**(이미지에 존재,
  O(N) 메모리). Transolver는 슬라이스로 이 문제를 아예 회피.)

## 6. 학습 비용 추정

- 기준: MGN h256/p15/batch1 = **189s/epoch**, 100 epoch = **5.3h**.
- Transolver 8~12층 선형 어텐션(N≈8k): forward가 MGN 15층 희소 메시지패싱과
  대략 동급~1.5배로 추정 → **~190–320s/epoch → 100 epoch ≈ 5–9h**.
- 불확실 → **스윕과 동일 프로토콜**: epoch 1에서 실측 타이밍·손실 확인 후 총
  epoch/step-stride 조정. GPU 경합(ANSYS) 시 하트비트가 플래그.

## 7. 스펙트럼 손실 포함 여부 → **예**

- 스펙트럼 손실은 `b_pred`에서 계산되어 **아키텍처 독립** → R3에 무변경 적용.
- 근거: §14b가 "실패 지점 = 밴드 고조파 계수"임을 측정으로 확정했고, bw1이 그
  계수 감독으로 |B|·토크를 동시에 갱신(12.499%/6.204%). R3의 전역 수용영역이
  슬롯 스케일 계수를 결합할 수 있는지 검정하려면 **그 계수를 직접 감독하는
  스펙트럼 손실과 함께** 쓰는 것이 가장 강한 실험.
- BAND_W = **30**(스윕 확정 §16: |B| 동률에서 토크 최고).

## 8. 수용 조건 (R3 판정 규칙)

1. 게이트: `mgn_nodeB_long` 재채점 13.324%/9.207% 유지(환경 불변 확인).
2. 결정성: R3 ckpt 2회 채점 바이트 동일(use_te=False 확인).
3. 판정 지표(TEST 6 case-holdout):
   - **성공** = |B| nRMSE가 스펙트럼 승자(bw*) 대비 유의하게 하락(런간 변동
     ~0.25pp를 넘는 폭), 즉 용량축이 못 깬 ~12.5% 바닥을 전역 수용영역이 깬다.
   - **부분** = 토크만 개선(|B| 정체) → 국소성은 토크 리플에만 유효.
   - **음성** = 둘 다 정체 → 병목은 수용영역이 아니다. R4(입력/데이터·물리
     제약 축)로 전환.

## 부록 — 조사 앵커
- 모델 구성: `train_doe_curl_mgn.py:595-611` / eval 재구성: `eval/predictors.py:383-416`
- 체크포인트 저장: `train_doe_curl_mgn.py:708-748`
- 손실(base/spectral/pbc): `train_doe_curl_mgn.py:628-671`
- 9-node/4-edge 계약: `eval/feature_guard.py:134-149`; 구성 `train_doe_curl_mgn.py:168-263`
- 빌딩블록 시그니처: 본 세션 컨테이너 probe(Transolver/BiStride/Hybrid)
