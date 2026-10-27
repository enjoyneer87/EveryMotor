"""Render the canonical request/idea JSON; no execution or automatic approvals."""
from pathlib import Path
import json
R=Path(__file__).parent
def main():
 d=json.loads((R/'IMPROVEMENT_REGISTER.json').read_text(encoding='utf-8'));ids=[x['id'] for x in d['items']];assert len(ids)==len(set(ids))
 lines=['# MotorAI 개선 요청·아이디어 통합 대장','',f"갱신일: {d['updated_KST']} KST. 원본은 `IMPROVEMENT_REGISTER.json`이며 이 문서는 `python update_improvement_register.py`로 갱신한다.",'',
 '사용자 요청, Astra 판정, 실행 계획과 실제 결과를 같은 항목 ID로 추적한다. 제안·판정·실행 완료는 서로 다른 상태다. 새 계산은 이 대장만으로 시작하지 않고 별도 bounded 실행 계약과 원자료/분할 gate를 통과해야 한다.','',
 'Astra 검토 채팅: `'+d['Astra_review_thread_id']+'`. 전체 형상·OP·모달리티는 geometry/alias group을 상속하고, reserved 및 기존 동결시험 계약을 유지한다.','',
 '| ID | 우선순위 | 개선 요청 / 아이디어 | 상태 |','|---|---|---|---|']
 for x in d['items']:lines.append(f"| [{x['id']}](#{x['id'].lower()}) | {x['priority']} | {x['title']} | {x['status']} |")
 labels={'origin':'요청 출처','request':'요청과 목표','confirmed':'확인된 근거와 현재 결과','Astra_decision':'Astra 판정','admission_gate':'실행·해석 관문','next_action':'다음 행동'}
 for x in d['items']:
  lines+=['',f"<a id=\"{x['id'].lower()}\"></a>",f"## {x['id']} · {x['title']}",'',f"**{x['priority']} · {x['status']}**",'']
  for k,label in labels.items():
   if x.get(k):lines+=[f"**{label}:** {x[k]}",'']
  links=[]
  for path in x['evidence']:
   assert (R/path).exists(),path
   links.append(f'[{Path(path).name}]({path})')
  lines+=['근거 파일: '+' · '.join(links)]
 lines+=['','## 갱신 규칙','','새 요청은 기존 항목을 찾아 내용·근거·다음 행동을 갱신한다. 별도 연구 축이면 새 MI 번호를 추가한다. 판정이 바뀌면 변경 이유와 근거를 보존하고, 코드 존재·reader PASS·학습 완료를 물리 정확도 PASS로 확대하지 않는다. 결과에는 실험 ID, 분할, 지표, 시드, Full donor 비용과 남은 UNKNOWN을 연결한다.']
 (R/'IMPROVEMENT_REGISTER_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'items':len(ids),'unique_ids':True}))
if __name__=='__main__':main()
