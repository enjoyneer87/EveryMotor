# 공통 공극·토크 평가와 a1 손실 실험 · 완료 증거 통합

2026-10-30 KST 통합. 원계산은2026-10-29 HPC_133에서사용자지시로Astra세션이별도작업폴더에서완료했다. 현재주연구Sol이원본을보존하며검증·EveryMotor게시를담당한다. 통합관리PC는DESKTOP-6QIVKIC이며실제원계산환경은동결receipt의provenance를따른다.

**공통평가12모델과후속a1손실6모델이이미완료돼있었으나canonical저장소대기표에는미실행으로남아있었다.** 이전주연구상태안내의미실행표현을정정하고,중복GPU/FEA없이완료증거를회수했다. [Astra 동기화 판정](ASTRA_SYNC_REVIEW_KO.md).

## 기존 모델의 같은 a1·Arkkio 연산자 진단

고정dev32×45=1,440 snapshot을A0/A2/C/G각3seed로평가해총17,280 graph를계산했다. nRMSE단위는%,각seed의집계지표를먼저계산한뒤3seed산술평균이다. 자계는native면적가중,토크는동일연산자로구한1,440값의RMSE/RMS(true)다.

| arm | 전체B벡터 | a1 B벡터 | a1 B크기 | 동일Arkkio토크 |
|---|---:|---:|---:|---:|
| A0 | 20.5993 | 44.1012 | 34.1790 | 25.0523 |
| A2 | 22.8470 | 44.3011 | 33.7756 | 30.1498 |
| C | 20.4250 | 43.2646 | 33.5306 | 29.3607 |
| G | 20.4728 | 43.0230 | 33.3849 | 33.1428 |

G의공극오차가조금낮아져도토크개선을보장하지않았다. A2/C-G의기존NO_IMPROVEMENT_CLAIM은불변이다. E1가중치를읽지않았으며legacy6의공극4.9563/토크3.2146%와같은시험으로순위를매기지않는다. [공통평가전체보고서](COMMON_A1_REPORT_KO.md), [수치](COMMON_A1_SUMMARY.json).

## 후속 손실 배분 단일축 L0/L05

동일A0구조·기계각·train328/phase9·seed17/29/43·88,565update/finalcheckpoint를고정하고, Lα=(1−α)Lall+αLa1의α만0/0.5로변경했다. 새6개학습과8,640개평가를완료했으며원래계약과전체실패이력을보존했다.

| arm | 전체B벡터 | a1 B벡터 | a1 B크기 | 동일Arkkio토크 |
|---|---:|---:|---:|---:|
| L0 | 20.5993 | 44.1012 | 34.1790 | 25.0523 |
| L05 | 22.9209 | 33.3613 | 27.9948 | 21.5005 |

a1 개선폭은10.7399pp로primary를통과했지만전체자계·seen9·구리·PA≥70의악화한도를넘었다. 토크평균은낮아졌으나seed변동기준을통과하지못했고리플오차도증가했다. **최종NO_IMPROVEMENT_CLAIM이며A0/L0기준을유지한다.** α나seed를사후선택하지않는다. [a1최종판정](../a1_loss_allocation_20261029/ASTRA_FINAL_REPORT_KO.md), [동결판정JSON](../a1_loss_allocation_20261029/A1_ALLOCATION_VERDICT.json).

## 검증과 공유 경계

원공통회수50개/a1회수57개의원본SHA·크기를검증했다. 원공통요약기와원동결판정기를저장JSON만으로별도실행해기존결과와일치했다. portable검사기는공유파일과원collection핀을대조하고공통1,140개·a1 650개지표분포와결정식을재계산한다. [이번검증receipt](INTEGRATION_VERIFICATION.json).

공유JSON의절대경로는logical-path키로정규화했고숫자leaffingerprint는동일하다. source_manifest의source_sha256는불변원본, published_sha256는정규화된공유파일이다. 원래receipt안SHA는원본을가리키며공유본의새hash와혼동하지않는다. 개인경로매핑·원본대형maskmanifest·graph/PT/HDF5/예측NPZ는private/원격에남긴다. 코드의고정source-stage식별자는실행했던원천신원을뜻하며개인접속정보를포함하지않는다.

반환검사명령: `python -B common_airgap_torque_20261029/verify_return.py` (연구루트기준). Torch·solver·binarypayload·네트워크없이실행한다. frozen_code와원평가기/판정기는원데이터receipt의신원검사를유지하므로정규화공유JSON을원본으로오인해강제로검사하지않는다.

32조건의a1는nativeID·45시각실제좌표/면적·기준면·입력치수와결합했다. case44에서a1 node와moving후보102개가겹치지만실제45좌표는기준과같았으므로후보flag만으로임의제외하지않았다. Geometry R1/R2와bootstrap실패는원기록으로남겼다. 길이0.15m와sector8근거가있어N·m로진단하지만같은B기반Arkkio참값이며별도solver토크/실측물리검증은아니다.

dev32는E1oldTRAIN노출innerdev다. 독립최종시험·AC손실·실제native연결복원성과가아니다. R1_0199/R1_0076/alias와oldval/test제외,HalfSC32/4,exactbridge0,canonicalAF분모계약을유지한다. E1prior/weights/teacher는admission까지HOLD다. 이번통합의새학습·추론·FEA는0이고다음실험은별도구체계약전HOLD다.
