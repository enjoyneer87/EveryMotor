# Astra 검토와 반영

사용자 요청으로 별도 Astra 채팅에서 읽기 전용 과학·코드 검토를 진행했다. 검토 의견은 solver 증거나 실행 검증을 대신하지 않는다.

- 입력 불일치80행 제외 후 기존val/test가 남지 않아, 기존TRAIN에서 입력만으로20그룹을 선택해16/4 내부 분할을 사전 동결했다.
- B-only pretraining은 J/Je 손실 정의와 exact bridge가 없어도 명시된 범위로 진행할 수 있다. AC/paired 학습은 비활성이다.
- native ID 공유 변 인접성,45스텝 moving좌표 복원,a2제외,그룹 분할은 정적 검토에서 즉시 차단 결함이 없었다. reader20/20과 두GPU smoke는 별도 receipt로 증명한다.
- 준비script 실패가 exit0으로 끝날 수 있어 trainer에서 failed0·16/4그룹·180파일 hash를강제했다. 후속 reader source는실패시exit2로보완했다. 이미완료bundle은재생성하지않았다.
- 준비외부timeout1200초,실제34.8초완료. 학습에는3600초/STOP/절대마감guard를적용했다.
- 배포guard는Docker cleanup실패에도baton해제가능성이있었다. 후속source는container종료재확인시에만idle해제로보완했다. 이미실행중인코드를바꿨다고주장하지않으며배포원본hash를DEPLOY_RECEIPT에보존한다. 이번run은종료후별도로immutablecontainerID상태를확인한다.
- 10GiB는PyTorch peak allocated의소프트검사이며전체GPU하드제한은아니다.
- 총128스텝은한해석history상한이다. N128P1과N64P2각128은준수한다. 초기누적256위반의견은원래사용자질문·답변확인후철회됐다.

검토채팅ID: `01a17d9e-8a81-7262-9eae-e086c1957892`. 계산·GitHub게시소유권은주채팅에있다.

## 충분한 샘플 확대 판정

사용자 추가지시에 따라 후보A(eligible360조건×9시각,30epoch)를우선하고45시각전수확대는후순위로정했다.360조건은320형상그룹이며288train/32innerdev그룹,실제328train/32dev조건이다.기존pilot16/4역할과oldval/test미사용을검토했다. 확대trainer의88565updates,cosine일정,그룹/파일검증,외부guard는정적검토에서즉시차단결함이없었다. 실제준비360/360PASS가GPU진입선행조건이다.

Pilot의train오차도62–63%로높아소표본일반화문제만으로단정할수없다.3-hop국소전달과a2제외로원거리상호작용표현은UNKNOWN이다. 데이터확대와update수·LR일정변화가함께있음을명시한다. epoch0평가는smoke5회후이며무학습baseline이아니다. 이번dev32는모두460Arms조건으로115/230Arms의독립형상일반화평가를포함하지않는다. 같은pilotdev4,전체dev32,형상/전류/영역별평가를구분한다.
