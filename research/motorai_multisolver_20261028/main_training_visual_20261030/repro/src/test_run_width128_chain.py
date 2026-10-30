"""Synthetic CPU tests only: fake processes, JSON, and text checkpoints; no datasets/GPU."""
from contextlib import ExitStack
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_width128_chain as chain


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


class Missing(Exception):pass
class Process:
    def __init__(self,pid,created,alive=True):self.pid=pid;self.created=created;self.alive=alive;self.terminated=0;self.killed=0
    def create_time(self):
        if not self.alive:raise Missing()
        return self.created
    def children(self,recursive=True):return []
    def terminate(self):self.terminated+=1;self.alive=False
    def kill(self):self.killed+=1;self.alive=False
    def is_running(self):return self.alive
class FakePsutil:
    NoSuchProcess=Missing
    def __init__(self):self.processes={os.getpid():Process(os.getpid(),1.)}
    def Process(self,pid):
        if pid not in self.processes or not self.processes[pid].alive:raise Missing()
        return self.processes[pid]
    @staticmethod
    def wait_procs(processes,timeout):return processes,[]


def fixture(root):
    def js(name,obj):
        p=root/name;p.write_text(json.dumps(obj));return chain.file_pin(p)
    names=['train_width128.py','a1_allocation_loss.py','dual_solver_mgn.py','evaluate_width128.py','capacity_contract.py']
    sources=[]
    for name in names:
        p=root/name;p.write_text(name);sources.append(chain.file_pin(p))
    byname={Path(s['path']).name:s for s in sources}
    selection=js('selection.json',{});pilot=js('pilot.json',{})
    traces={str(n):{str(seed):digest('order'+str(n)+str(seed)) for seed in (17,29,43)} for n in (2957,88565)}
    widths={arm:dict(initial_state_sha256_by_seed={str(seed):digest(arm+str(seed)) for seed in (17,29,43)},expected_data_order_LR_trace_sha256=traces) for arm in chain.ARMS}
    bridge=dict(status='PASS_BASELINE_BRIDGE_CPU_ONLY',synthetic_W64={'status':'PASS_BIT_EXACT_SYNTHETIC_CPU'},**{key:True for key in ('trainer_full_module_AST_equivalent','model_source_identical','numerical_function_AST_equivalent','outer_main_AST_equivalent','initial_states_match_historical','all45_order_LR_traces_match_historical')})
    reference=dict(bridge=bridge,parameters=273478,field_path_parameters=118530,initial_state_sha256_by_seed={str(seed):digest('W64'+str(seed)) for seed in (17,29,43)},expected_data_order_LR_trace_sha256=copy.deepcopy(traces))
    cpu=dict(data_arm='A45',experiment_id='W128_PHASE45_EQUAL_UPDATE_20261030',reference_W64=reference,status='PASS',torch='2.5.1+cu124',numpy='1.26.4',widths=widths,source_files=sources)
    masks=dict(status='ADMITTED',records=[{}]*328,development_records=[{}]*32,selection_sha256=selection['sha256'],bundle_manifest_sha256=digest('bundle'))
    mask=js('masks.json',masks);deadline=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=12)).isoformat()
    plans={};eng={};full={}
    for arm,width in chain.ARMS.items():
        obj=dict(status='RELEASED_ENGINEERING_ONLY',training=dict(width=width,depth=3,device='cuda'),
                 arms={'W128':{'alpha':0.,'width':128}},fixed_seeds=[17,29,43],
                 data_arm='A45',experiment_id='W128_PHASE45_EQUAL_UPDATE_20261030',train_steps=list(range(45)),seen_steps=list(range(0,45,5)),selection=selection,pilot_manifest=pilot,bundle_manifest_sha256=digest('bundle'),mask_manifest_sha256=mask['sha256'],
                 guard_deadline_UTC=deadline,initial_state_sha256_by_seed=widths[arm]['initial_state_sha256_by_seed'],source_files=sources)
        eng[arm]=js(arm+'_engineering.json',obj);plans[arm]=obj
        obj=copy.deepcopy(obj);obj['status']='RELEASED_FIXED_WIDTH128_RUNS';full[arm]=js(arm+'_training.json',obj)
    template=dict(release_status='RELEASED_FROZEN_EVALUATION',max_seconds=1800,workers=1,retry_limit=0,training_updates=0,new_FEA=0,new_prior_solves=0,
                  phase_encoding='mechanical',selection=selection,training_mask_manifest=mask,geometry_manifest=js('geometry.json',{}),source_files=sources,deadline_UTC=deadline)
    plan=dict(status='RELEASED_ENGINEERING_ONLY',stage_root=str(root),runs_root=str(root/'runs'),python_executable=sys.executable,pythonpath=str(root),
              guard=chain.file_pin(Path(chain.__file__)),trainer=byname['train_width128.py'],evaluator=byname['evaluate_width128.py'],source_files=sources,
              engineering_plans=eng,training_plans=full,mask_manifest=mask,cpu_admission=js('cpu.json',cpu),evaluation_template=js('eval_template.json',template),train_bundle_dir=str(root/'bundle'),
              stop_files=[str(root/'STOP'),str(root/'runs'/'STOP')],deadline_UTC=deadline,workers=1,retry_limit=0,global_max_seconds=28800,engineering_max_seconds=900,
              engineering_updates=2957,train_max_seconds=10800,eval_max_seconds=1800,disk_free_min_bytes=10*1024**3,torch_allocated_max_bytes=10*1024**3,
              projection_safety_factor=1.25,collection_reserve_seconds=300,run_order=list(chain.RUN_IDS))
    path=root/'chain.json';path.write_text(json.dumps(plan));return plan,path,cpu


class FakeWorkers:
    def __init__(self,plan,cpu,psutil,fail=None):self.plan=plan;self.cpu=cpu;self.psutil=psutil;self.commands=[];self.fail=fail
    def popen(self,command,**kwargs):
        self.commands.append(command);index=len(self.commands);pid=50000+index
        proc=Process(pid,1000.+index);self.psutil.processes[pid]=proc
        out=Path(command[command.index('--out')+1]);out.mkdir()
        if '--job' in command:
            job=json.loads(Path(command[command.index('--job')+1]).read_text())
            result=dict(status='COMPLETED',graphs_processed=1440,training_updates=0,job_sha256=command[command.index('--job-sha256')+1],checkpoint_sha256=job['checkpoint']['sha256'],
                        reproduction_gate={'status':'PASS_ARITHMETIC_ONLY'},data_arm='A45',experiment_id='W128_PHASE45_EQUAL_UPDATE_20261030',arm=job['arm'],width=job['width'],depth=3,seed=job['seed'])
        else:
            arm=command[command.index('--arm')+1];seed=int(command[command.index('--seed')+1]);engineering='--max-updates' in command
            updates=2957 if engineering else 88565;epochs=[0,1] if engineering else [0,1,5,10,20,30]
            p=json.loads(Path(command[command.index('--plan')+1]).read_text());expected=chain.expected_hashes(self.plan,p,self.cpu,arm,seed,engineering)
            checkpoint=out/'last.pt';checkpoint.write_text('synthetic text checkpoint')
            result=dict(status='SMOKE_COMPLETED' if engineering else 'COMPLETED',training_updates=updates,arm=arm,seed=seed,alpha=0.,width=chain.ARMS[arm],depth=3,execution_device='cuda',torch='2.5.1+cu124',numpy='1.26.4',
                        parameters=chain.PARAMETERS[arm][0],field_path_parameters=chain.PARAMETERS[arm][1],phase_encoding='mechanical',data_arm='A45',experiment_id='W128_PHASE45_EQUAL_UPDATE_20261030',train_steps=list(range(45)),train_graphs=14760,train_phase9_diagnostic_graphs=2952,old_checkpoint_reused=False,
                        new_FEA=0,new_prior_solves=0,original_val_test_used=0,GPU_smoke='PASS',peak_torch_allocated_bytes=100,max_updates=2957 if engineering else None,
                        checkpoint_sha256=chain.sha(checkpoint),elapsed_seconds=1.,assessment_wall_seconds_total=.1*len(epochs),
                        metrics=[dict(epoch=epoch,training_updates=updates if epoch==epochs[-1] else 5,assessment_wall_seconds=.1,development={'all':{'snapshots':288,'vector_nRMSE_percent':20.}}) for epoch in epochs],**expected)
        (out/'RESULT.json').write_text(json.dumps(result));outer=self
        class Worker:
            def __init__(self):self.pid=pid;self.polls=0
            def poll(self):
                self.polls+=1
                if self.polls>=2:proc.alive=False;return 1 if index==outer.fail else 0
                return None
            def wait(self,timeout=None):proc.alive=False;return 1 if index==outer.fail else 0
        return Worker()


class GuardTests(unittest.TestCase):
    def release(self,root,plan,path):
        cpu=json.loads(Path(plan['cpu_admission']['path']).read_text());controls={};trains={}
        oldcpu=dict(status='PASS',widths={'W64':cpu['reference_W64']})
        p=root/'oldcpu.json';p.write_text(json.dumps(oldcpu));oldpin=chain.file_pin(p)
        for seed in (17,29,43):
            meta=dict(initial_tensor_state_sha256=cpu['reference_W64']['initial_state_sha256_by_seed'][str(seed)],
                      data_order_LR_trace_sha256=cpu['reference_W64']['expected_data_order_LR_trace_sha256']['88565'][str(seed)])
            p=root/f'control_{seed}.json';p.write_text(json.dumps(dict(seed=seed,status='COMPLETED',arm='W64',width=64,depth=3,graphs_processed=1440,training_updates=0,checkpoint_sha256='c'*64,metadata_hashes=meta,data_arm='A45',experiment_id='W64_PHASE45_EQUAL_UPDATE_20261030')));controls[str(seed)]=chain.file_pin(p)
            train=dict(status='COMPLETED',arm='W64',seed=seed,width=64,depth=3,data_arm='A45',alpha=0.,training_updates=88565,train_steps=list(range(45)),parameters=273478,field_path_parameters=118530,checkpoint_sha256='c'*64,experiment_id='W64_PHASE45_EQUAL_UPDATE_20261030',**meta)
            p=root/f'train_control_{seed}.json';p.write_text(json.dumps(train));trains[str(seed)]=chain.file_pin(p)
        release=dict(status='RELEASED_FIXED_WIDTH128_RUNS',chain_plan_sha256=chain.sha(path),cpu_admission_sha256=plan['cpu_admission']['sha256'],
                     training_plan_sha256_by_arm={arm:plan['training_plans'][arm]['sha256'] for arm in chain.ARMS},
                     engineering_receipt=chain.file_pin(root/'runs/ENGINEERING_RECEIPT.json'),phase45_controls_by_seed=controls,phase45_control_train_results_by_seed=trains,phase45_cpu_admission=oldpin)
        p=root/'release.json';p.write_text(json.dumps(release));return chain.file_pin(p)
    def setup_fake(self,root,fail=None,gpu=None):
        plan,path,cpu=fixture(root);psutil=FakePsutil();workers=FakeWorkers(plan,cpu,psutil,fail);clock=[0.]
        def mono():clock[0]+=1.;return clock[0]
        stack=ExitStack();stack.enter_context(patch.dict(sys.modules,{'psutil':psutil}))
        stack.enter_context(patch.object(chain,'gpu_compute_pids',return_value=[] if gpu is None else gpu))
        stack.enter_context(patch.object(chain.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
        stack.enter_context(patch.object(chain.subprocess,'Popen',side_effect=workers.popen))
        stack.enter_context(patch.object(chain.time,'monotonic',side_effect=mono));stack.enter_context(patch.object(chain.time,'sleep'))
        return stack,plan,path,workers,psutil
    def test_engineering_exits_after_one_and_cannot_auto_full(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root)
            with stack:result=chain.run_guard(plan,chain.sha(path),path)
            self.assertEqual(result['status'],'ENGINEERING_COMPLETED_AWAITING_RELEASE');self.assertEqual(len(workers.commands),1)
            self.assertEqual(result['engineering_updates'],2957);self.assertEqual(result['full_training_updates'],0)
            self.assertEqual([c[c.index('--width')+1] for c in workers.commands],['128'])
            self.assertTrue(all('--max-updates' in c for c in workers.commands));self.assertFalse((root/'WIDTH128_GUARD_LOCK.json').exists())
    def test_scientific_without_release_cannot_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root)
            with stack,self.assertRaisesRegex(ValueError,'separate release'):chain.run_guard(plan,chain.sha(path),path,'scientific')
            self.assertEqual(workers.commands,[])
    def test_complete_separate_release_only_three_W128_trains_evals(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root)
            with stack:
                chain.run_guard(plan,chain.sha(path),path)
                release=self.release(root,plan,path)
                result=chain.run_guard(plan,chain.sha(path),path,'scientific',release)
            self.assertEqual(result['status'],'COMPLETED');self.assertEqual(len(workers.commands),7)
            self.assertEqual(result['full_training_updates'],265695);self.assertEqual(result['evaluation_graphs'],4320)
            self.assertEqual((result['completed_train_runs'],result['completed_eval_runs']),(3,3))
            self.assertTrue(all(c[c.index('--width')+1]=='128' for c in workers.commands[1:] if '--width' in c))
    def test_failed_control_binding_stops_without_full_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root)
            with stack:
                chain.run_guard(plan,chain.sha(path),path);spec=self.release(root,plan,path)
                release=json.loads(Path(spec['path']).read_text());p=Path(release['phase45_controls_by_seed']['17']['path']);control=json.loads(p.read_text())
                control['metadata_hashes']['initial_tensor_state_sha256']='f'*64;p.write_text(json.dumps(control));release['phase45_controls_by_seed']['17']=chain.file_pin(p)
                p=Path(spec['path']);p.write_text(json.dumps(release));spec=chain.file_pin(p)
                with self.assertRaisesRegex(ValueError,'CPU binding'):chain.run_guard(plan,chain.sha(path),path,'scientific',spec)
            self.assertEqual(len(workers.commands),1)
    def test_optional_baseline_bridge_requires_current_CPU_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root)
            with stack:
                chain.run_guard(plan,chain.sha(path),path);spec=self.release(root,plan,path)
                release=json.loads(Path(spec['path']).read_text())
                bridge=dict(status='PASS_W64_PHASE45_BASELINE_BRIDGE',cpu_admission_sha256='f'*64,control_results_by_seed=release['phase45_controls_by_seed'])
                p=root/'bridge.json';p.write_text(json.dumps(bridge));release['baseline_bridge']=chain.file_pin(p)
                p=Path(spec['path']);p.write_text(json.dumps(release));spec=chain.file_pin(p)
                with self.assertRaisesRegex(ValueError,'current CPU admission'):chain.run_guard(plan,chain.sha(path),path,'scientific',spec)
            self.assertEqual(len(workers.commands),1)
    def test_foreign_gpu_no_output_no_kill(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,psutil=self.setup_fake(root,gpu=[999])
            with stack,self.assertRaisesRegex(ValueError,'foreign GPU'):chain.run_guard(plan,chain.sha(path),path)
            self.assertFalse((root/'runs').exists());self.assertEqual(workers.commands,[]);self.assertEqual(sum(p.terminated for p in psutil.processes.values()),0)
    def test_existing_output_retry_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root);(root/'runs').mkdir()
            with stack,self.assertRaisesRegex(ValueError,'retry0'):chain.run_guard(plan,chain.sha(path),path)
            self.assertEqual(workers.commands,[])
    def test_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root,fail=1)
            with stack,self.assertRaisesRegex(ValueError,'nonzero'):chain.run_guard(plan,chain.sha(path),path)
            status=json.loads((root/'runs/ENGINEERING_RECEIPT.json').read_text());self.assertEqual(len(workers.commands),1)
            self.assertEqual(status['status'],'INCOMPLETE_NO_CLAIM')
    def test_stop_before_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root);(root/'STOP').touch()
            with stack,self.assertRaisesRegex(InterruptedError,'STOP_FILE'):chain.run_guard(plan,chain.sha(path),path)
            self.assertEqual(workers.commands,[])
    def test_lock_declines_second_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.setup_fake(root);(root/'WIDTH128_GUARD_LOCK.json').write_text('{}')
            with stack,self.assertRaises(FileExistsError):chain.run_guard(plan,chain.sha(path),path)
            self.assertEqual(workers.commands,[])
    def test_recycled_pid_untouched(self):
        ps=FakePsutil();ps.processes[123]=Process(123,2.);ps.processes[456]=Process(456,4.)
        chain.terminate_owned(ps,[{'pid':123,'create_time':1.},{'pid':456,'create_time':4.}])
        self.assertEqual(ps.processes[123].terminated,0);self.assertEqual(ps.processes[456].terminated,1)
    def test_identity_tree_rejects_recycled_root(self):
        ps=FakePsutil();ps.processes[123]=Process(123,2.)
        with self.assertRaisesRegex(ValueError,'PID changed'):chain.capture_owned_tree(ps,{'pid':123,'create_time':1.})
    def test_exact_bounds_and_absolute_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_,_=fixture(Path(folder));chain.validate_plan(plan)
            for key,value in [('workers',2),('retry_limit',1),('global_max_seconds',57600),('train_max_seconds',7200),('torch_allocated_max_bytes',11*1024**3),('stop_files',['STOP'])]:
                bad=copy.deepcopy(plan);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):chain.validate_plan(bad)
    def test_deadline_global_engineering_worker(self):
        now=dt.datetime.now(dt.timezone.utc);later=now+dt.timedelta(hours=12)
        for args,expected in [({'stop':True},'STOP_FILE'),({'deadline':now},'UTC_DEADLINE'),({'elapsed':28800},'GLOBAL_28800_SECONDS'),({'engineering_elapsed':900},'ENGINEERING_900_SECONDS'),({'worker_elapsed':10800,'worker_limit':10800},'WORKER_WALL_LIMIT')]:
            kw=dict(now=now,deadline=later,elapsed=0);kw.update(args);self.assertEqual(chain.limit_reason(**kw),expected)
    def test_profile_uses_real_phase45_timing_and_only_three_W128_runs(self):
        profiles={arm:(dict(status='SMOKE_COMPLETED',training_updates=2957,arm=arm,width=width,execution_device='cuda',elapsed_seconds=101.,assessment_wall_seconds_total=20.,metrics=[{'assessment_wall_seconds':10.}]*2),102.) for arm,width in chain.ARMS.items()}
        projection=chain.profile_projection(profiles,40000)
        expected=1.25*(1+81/2957*88565+60)
        self.assertAlmostEqual(projection['projected_remaining_seconds'],3*expected+5400+300)
        for change in ('missing','assessment','cuda','remaining'):
            p=copy.deepcopy(profiles);remaining=40000
            if change=='missing':del p['W128']
            if change=='assessment':p['W128'][0]['assessment_wall_seconds_total']=101.
            if change=='cuda':p['W128'][0]['execution_device']='cpu'
            if change=='remaining':remaining=100
            with self.subTest(change=change),self.assertRaises(ValueError):chain.profile_projection(p,remaining)
    def test_input_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_,_=fixture(Path(folder));chain.admit_inputs(plan,'engineering')
            Path(plan['source_files'][0]['path']).write_text('modified')
            with self.assertRaisesRegex(ValueError,'SHA differs'):chain.admit_inputs(plan,'engineering')
    def test_cpu_reference_order_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_,_=fixture(Path(folder));spec=plan['cpu_admission'];p=Path(spec['path']);cpu=json.loads(p.read_text());cpu['reference_W64']['expected_data_order_LR_trace_sha256']['2957']['17']=digest('wrong')
            p.write_text(json.dumps(cpu));plan['cpu_admission']=chain.file_pin(p)
            with self.assertRaisesRegex(ValueError,'order/LR'):chain.admit_inputs(plan,'engineering')
    def test_failed_cpu_bridge_gate_rejected_before_engineering(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_,_=fixture(Path(folder));spec=plan['cpu_admission'];p=Path(spec['path']);cpu=json.loads(p.read_text())
            cpu['reference_W64']['bridge']['outer_main_AST_equivalent']=False
            p.write_text(json.dumps(cpu));plan['cpu_admission']=chain.file_pin(p)
            with self.assertRaisesRegex(ValueError,'bridge gate failed'):chain.admit_inputs(plan,'engineering')
    def test_unknown_gpu_response_rejected(self):
        self.assertEqual(chain.parse_gpu_pids('123\n123\n456\n'),[123,456])
        self.assertEqual(chain.parse_gpu_pids(''),[])
        with self.assertRaises(ValueError):chain.parse_gpu_pids('N/A')


if __name__=='__main__':unittest.main()
