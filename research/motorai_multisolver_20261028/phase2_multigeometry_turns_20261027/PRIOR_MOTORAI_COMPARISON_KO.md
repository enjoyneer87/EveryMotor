# 이전 MotorAI 점수와 현재 모델 비교의 경계

사용자가 기억한5% 이하 성과를 iso 원점수와 `em.sh scores --all`로 확인했다. 원본 source commit은 `65dda206e28770b242fa0dd888fd394f7a155ca0`이다.

| E1 airgap5 평가 집단 | 전체 \|B\| | 공극 자계 | 토크 |
|---|---:|---:|---:|
| legacy6 | 11.096697% | 4.956319% | 3.214573% |
| new12 | 10.737% | 3.303% | 2.050% |
| all18 | 10.859% | 4.071% | 2.982% |

legacy6의공극은5%이하지만전체자계약11.1%도사용자요청의개선대상이다. all18혼합수치를legacy6의기존gate로바꾸지않는다.

원점수 파일: `results/benchmark_v2_nodeB_doe_v3_bw30_prior_e1airgap5{,_legacy6,_new12}.json`. legacy6 SHA-256은 `74cfb960df989da640e6104d501ac19dc9f549a434c7e7b610296962e87ccfa8`, all18은 `b84c1e4653a96085a4a7c2982688a2458a77624121feb24ce74ee18baa4bc8e3`이다. recipe `run_e1_airgap_docker.sh` SHA-256은 `16b00f06e2486dd22d4870d55a9e7facf193b5f7f79375cb9ad874d0ff830132`이다. source/code/점수의읽기전용사본과전체hash는private evidence mapping으로보존했다. weights는이감사를위해재사용하지않았다.

E1은h256/p15 약9.29M의기존학습체계이고현재다형상추가모델은width32/depth3의native FE요소graph다. 현재dev32의18.0765%는a2를제외한요소면적가중B벡터nRMSE로서위표와평가집단·출력위치·metric weighting이달라직접우열비교할수없다. 현재구리영역B벡터오차는34.6131%다. 전체 \|B\|와벡터,공극,구리,B²및AC손실을공통평가계약으로연결해야한다.

**모델 계보 정정:** E1 원본recipe `--target B`와저장checkpoint감사의`target=B`를확인했다. 실제경로는nodal Bx/By를요소로평균한다. trainer이름`curl`만으로E1을nodal A→P1curl head라고단정한초기의견은철회한다. 실제이식대상은원래B head/normalizer/inputfeature/processor이며조건부전략은[아스트라 전이·멀티모달 판정](ASTRA_TRANSFER_MULTIMODAL_KO.md)을따른다.

가중치전이,기존구조/recipe재사용,새솔버adapter학습을구분한다. 기존E1이현재innerdev32를pretrain에서보았을가능성이있으므로그가중치로새독립형상시험이라고평가하지않는다. donorFull450train비용과새증분Full예산도별도로기록한다. 실행가능한후속전략은사용자요청Astra검토후개선대장의MI-004/005/006/012에반영한다.
