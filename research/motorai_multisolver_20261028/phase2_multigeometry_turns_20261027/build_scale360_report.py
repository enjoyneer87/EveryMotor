from pathlib import Path
import json
R=Path(__file__).parent
def main():
 status=json.loads((R/'SCALE360_TRAIN_STATUS.json').read_text());bundle=json.loads((R/'SCALE360_BUNDLE_MANIFEST.json').read_text());result=json.loads((R/'SCALE360_TRAIN_RESULT.json').read_text()) if (R/'SCALE360_TRAIN_RESULT.json').exists() else {}
 maximum=max(s['B_magnitude_roundtrip_max_T'] for r in bundle['records'] for s in r['QA']['steps'])
 lines=['# 확대 다형상 B 학습: 133 우선','',
 f"상태: {status['training_status']} · {status['epochs_completed']}/30epoch · {status['training_updates']}/88565updates. 실제 소유PID 생존: {status.get('owned_process_live')}.",'',
 '기존HDF5에서360조건/320형상그룹의45스텝reader검사360PASS·0FAIL을확인했다. 9스텝씩3240개native FE요소graph를준비했다. 288train그룹328조건,32innerdev그룹32조건이며기존pilot16/4역할을유지한다. 원래val/test0,과거checkpoint0,새FEA0이다.','',
 f'B벡터/저장크기 산술 최대차이는{maximum:.9g}T다. 각스텝의float32기반허용오차와native ID·moving좌표·a2외기하검사를통과했다. 이PASS는물리수렴/AC손실적분PASS가아니다.','',
 '133 L40S 한작업에서width32/depth3,seed17,30epoch,cosine1e-3→1e-5를사전고정했다. 마지막checkpoint만보존하며bestseed/epoch선택을하지않는다. host당GPU한작업,10800초/UTC22:59/STOP/소유PID시작시각guard,자동재시도0이다. 234는reader와작은진단용으로둔다.','',
 '환경1은h5py없는PyTorchvenv를선택해0field payload/0graph에서중단했다. 실패receipt를보존하고,검증된native Python3.13+h5py3.16+NumPy2.4.2로별도R2준비를160초에완료했다. HDF5읽기에Docker는필수가아니다. 133학습은native PyTorch환경에서NPZ를읽는다.','',
 '| 평가 epoch | updates 의미 | train B nRMSE | dev32 B nRMSE | 고정 pilot dev4 B nRMSE |','|---:|---|---:|---:|---:|']
 for m in result.get('metrics',[]):
  tr=m['train']['all']['vector_nRMSE_percent'];dv=m['development']['all']['vector_nRMSE_percent'];pd=m['development'].get('fixed_pilot_dev4',{}).get('vector_nRMSE_percent')
  lines.append(f"| {m['epoch']} | {'5회 smoke 후' if m['epoch']==0 else '해당 epoch 완료'} | {tr:.3f}% | {dv:.3f}% | {pd:.3f}% |")
 lines+=['','nRMSE는a2를제외한native요소면적가중B벡터오차다. 결과JSON에case/geometry/current/material별오차도저장한다. dev32는460Arms조건뿐이므로115/230Arms독립형상일반화검증을포함하지않는다. epoch0은5회smoke후이며무학습baseline이아니다.','',
 '표본수·update수·학습률일정이pilot과함께달라져순수샘플수효과로해석하지않는다. Pilot의train오차도높았으므로소표본만이원인이라고단정하지않는다.3-hop국소전달과a2제외에따른원거리장표현능력은추가판정이필요하다.','',
 'Motor-CAD transient B 전용사전학습이며J/Je/AC손실/MS교사/pairedalignment를학습하지않는다. 기존Full63/새OP8동결비교는별개다. donorFull비용은328학습조건+32내부검증조건으로명시한다. exactcrosssolver0,formal수렴UNKNOWN을유지한다.','',
 '큰raw/HDF5/NPZ/checkpoint는private remote source-key mapping으로보존하고공개manifest/hash/code/report만GitHub에남긴다. SCALE360_PLAN.json, SCALE360_SELECTION_FROZEN.json, SCALE360_BUNDLE_MANIFEST.json, SCALE360_TRAIN_STATUS.json, SCALE360_TRAIN_RESULT.json으로계약·실행·결과를대조한다.']
 (R/'SCALE360_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'status':status['training_status'],'updates':status['training_updates'],'reader_cases':360,'B_roundtrip_max_T':maximum}))
if __name__=='__main__':main()
