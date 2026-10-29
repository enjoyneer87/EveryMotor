"""Fake-process CPU guard tests: no GPU commands, model imports or remote access."""
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
import run_a1_chain as chain


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


class FakeNoSuchProcess(Exception):pass


class Process:
    def __init__(self,pid,created,alive=True):self.pid=pid;self.created=created;self.alive=alive;self.terminated=0;self.killed=0
    def create_time(self):
        if not self.alive:raise FakeNoSuchProcess()
        return self.created
    def children(self,recursive=True):return []
    def terminate(self):self.terminated+=1;self.alive=False
    def kill(self):self.killed+=1;self.alive=False
    def is_running(self):return self.alive


class FakePsutil:
    NoSuchProcess=FakeNoSuchProcess
    def __init__(self):self.processes={os.getpid():Process(os.getpid(),1.)}
    def Process(self,pid):
        if pid not in self.processes or not self.processes[pid].alive:raise FakeNoSuchProcess()
        return self.processes[pid]
    @staticmethod
    def wait_procs(processes,timeout):return processes,[]


def fixture(root):
    source_names=['train_a1_allocation.py','a1_allocation_loss.py','dual_solver_mgn.py','evaluate_a1_allocation.py']
    sources=[]
    for name in source_names:
        path=root/name;path.write_text(name);sources.append(chain.file_pin(path))
    specs={Path(spec['path']).name:spec for spec in sources}
    initial={str(seed):digest('init'+str(seed)) for seed in (17,29,43)}
    traces={str(n):{str(seed):digest('trace'+str(n)+str(seed)) for seed in (17,29,43)} for n in (2957,88565)}
    cpu={'status':'PASS','initial_state_sha256_by_seed':initial,'expected_data_order_LR_trace_sha256':traces,'source_files':sources}
    deadline=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=16)).isoformat()
    selection={'path':str(root/'selection.json'),'sha256':digest('selection')}
    training={'selection':selection,'bundle_manifest_sha256':digest('bundle'),'initial_state_sha256_by_seed':initial}
    engineering=copy.deepcopy(training)
    mask={'path':str(root/'mask.json'),'sha256':digest('mask')}
    template={'release_status':'RELEASED_FROZEN_EVALUATION','max_seconds':1800,'workers':1,'retry_limit':0,
              'training_updates':0,'new_FEA':0,'new_prior_solves':0,'phase_encoding':'mechanical',
              'selection':selection,'training_mask_manifest':mask,'deadline_UTC':deadline}
    plan={'status':'RELEASED_ENGINEERING_THEN_FIXED_SIX_RUNS','stage_root':str(root),'runs_root':str(root/'runs'),
          'python_executable':sys.executable,'pythonpath':str(root),'guard':chain.file_pin(Path(chain.__file__)),
          'trainer':specs['train_a1_allocation.py'],'evaluator':specs['evaluate_a1_allocation.py'],'source_files':sources,
          'training_plan':{'path':str(root/'TRAINING_PLAN.json'),'sha256':digest('fullplan')},
          'engineering_plan':{'path':str(root/'ENGINEERING_PLAN.json'),'sha256':digest('engplan')},
          'mask_manifest':mask,'cpu_admission':{},'evaluation_template':{},'train_bundle_dir':str(root/'bundle'),
          'stop_files':[str(root/'STOP')],'deadline_UTC':deadline,'workers':1,'retry_limit':0,'global_max_seconds':57600,
          'engineering_max_seconds':900,'engineering_updates':2957,'train_max_seconds':7200,'eval_max_seconds':1800,
          'disk_free_min_bytes':10*1024**3,'projection_safety_factor':1.25,'collection_reserve_seconds':300,'run_order':list(chain.RUN_IDS)}
    path=root/'CHAIN_PLAN.json';path.write_text(json.dumps(plan))
    return plan,path,(training,engineering,{},cpu,template)


class FakeWorkers:
    def __init__(self,plan,admission,psutil,fail_index=None):
        self.plan=plan;self.admission=admission;self.psutil=psutil;self.fail_index=fail_index;self.commands=[]
    def popen(self,command,**kwargs):
        self.commands.append(command)
        index=len(self.commands);pid=50000+index
        process=Process(pid,1000.+index);self.psutil.processes[pid]=process
        out=Path(command[command.index('--out')+1]);out.mkdir()
        if '--job' in command:
            job=json.loads(Path(command[command.index('--job')+1]).read_text())
            result={'status':'COMPLETED','graphs_processed':1440,'training_updates':0,
                    'job_sha256':command[command.index('--job-sha256')+1],'checkpoint_sha256':job['checkpoint']['sha256'],
                    'reproduction_gate':{'status':'PASS_ARITHMETIC_ONLY'}}
        else:
            arm=command[command.index('--arm')+1];seed=int(command[command.index('--seed')+1]);engineering='--max-updates' in command
            updates=2957 if engineering else 88565;epochs=[0,1] if engineering else [0,1,5,10,20,30]
            expected=chain.expected_hashes(self.plan,self.admission[1 if engineering else 0],self.admission[3],arm,seed,engineering)
            checkpoint=out/'last.pt';checkpoint.write_bytes((arm+str(seed)+str(updates)).encode())
            result={'status':'SMOKE_COMPLETED' if engineering else 'COMPLETED','training_updates':updates,'arm':arm,'seed':seed,'alpha':chain.ARMS[arm],
                    'parameters':70182,'phase_encoding':'mechanical','data_arm':'A0','old_checkpoint_reused':False,
                    'new_FEA':0,'new_prior_solves':0,'original_val_test_used':0,'GPU_smoke':'PASS','peak_torch_allocated_bytes':100,
                    'max_updates':2957 if engineering else None,'checkpoint_sha256':chain.sha(checkpoint),'elapsed_seconds':1.,
                    'assessment_wall_seconds_total':.1*len(epochs),
                    'metrics':[{'epoch':epoch,'training_updates':updates if epoch==epochs[-1] else 5,'assessment_wall_seconds':.1,
                                'development':{'all':{'snapshots':288,'vector_nRMSE_percent':20.}}} for epoch in epochs],**expected}
        (out/'RESULT.json').write_text(json.dumps(result))
        class Worker:
            def __init__(self):self.pid=pid;self.polls=0
            def poll(self):
                self.polls+=1
                if self.polls>=2:process.alive=False;return 1 if index==self_outer.fail_index else 0
                return None
            def wait(self,timeout=None):process.alive=False;return 1 if index==self_outer.fail_index else 0
        self_outer=self
        return Worker()


class GuardTests(unittest.TestCase):
    def run_fake(self,root,fail_index=None,gpu_pids=None):
        plan,path,admission=fixture(root);psutil=FakePsutil();workers=FakeWorkers(plan,admission,psutil,fail_index)
        clock=[0.]
        def monotonic():clock[0]+=1.;return clock[0]
        stack=ExitStack()
        stack.enter_context(patch.dict(sys.modules,{'psutil':psutil}))
        stack.enter_context(patch.object(chain,'admit_inputs',return_value=admission))
        stack.enter_context(patch.object(chain,'gpu_compute_pids',return_value=[] if gpu_pids is None else gpu_pids))
        stack.enter_context(patch.object(chain.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
        stack.enter_context(patch.object(chain.subprocess,'Popen',side_effect=workers.popen))
        stack.enter_context(patch.object(chain.time,'monotonic',side_effect=monotonic))
        stack.enter_context(patch.object(chain.time,'sleep',return_value=None))
        return stack,plan,path,workers,psutil

    def test_complete_fourteen_jobs_no_retry_and_exact_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            stack,plan,path,workers,psutil=self.run_fake(root)
            with stack:result=chain.run_guard(plan,chain.sha(path),path)
            self.assertEqual(result['status'],'COMPLETED')
            self.assertEqual((result['completed_train_runs'],result['completed_eval_runs']),(6,6))
            self.assertEqual((result['full_training_updates'],result['engineering_updates'],result['evaluation_graphs']),(531390,5914,8640))
            self.assertEqual(len(workers.commands),14)
            self.assertEqual(len(list((root/'runs').glob('*_GUARD_RECEIPT.json'))),14)
            self.assertEqual(sum(p.terminated for p in psutil.processes.values()),0)
            self.assertTrue((root/'runs/L05_s43/eval_job.json').is_file())
            self.assertTrue(all(row['status']=='COMPLETED' for row in result['jobs']))

    def test_first_training_failure_preserves_unstarted_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,psutil=self.run_fake(root,fail_index=3)
            with stack,self.assertRaisesRegex(ValueError,'nonzero'):chain.run_guard(plan,chain.sha(path),path)
            status=json.loads((root/'runs/CHAIN_STATUS.json').read_text())
            self.assertEqual(status['status'],'INCOMPLETE_NO_CLAIM')
            self.assertEqual(len(workers.commands),3)
            self.assertEqual(status['jobs'][2]['status'],'INCOMPLETE_NO_CLAIM')
            self.assertTrue(all(row['status']=='NOT_STARTED' for row in status['jobs'][3:]))
            self.assertTrue((root/'runs/L0_s17/train/RESULT.json').exists())

    def test_foreign_gpu_declines_without_output_or_process_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,psutil=self.run_fake(root,gpu_pids=[999])
            with stack,self.assertRaisesRegex(ValueError,'foreign GPU'):chain.run_guard(plan,chain.sha(path),path)
            self.assertFalse((root/'runs').exists());self.assertEqual(len(workers.commands),0)
            self.assertEqual(sum(p.terminated for p in psutil.processes.values()),0)

    def test_existing_runs_declined_without_job_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stack,plan,path,workers,_=self.run_fake(root);(root/'runs').mkdir()
            with stack,self.assertRaisesRegex(ValueError,'retry0'):chain.run_guard(plan,chain.sha(path),path)
            self.assertEqual(len(workers.commands),0)

    def test_limits_stop_utc_global_engineering_and_worker(self):
        now=dt.datetime.now(dt.timezone.utc);future=now+dt.timedelta(hours=16)
        self.assertEqual(chain.limit_reason(now,future,0,stop=True),'STOP_FILE')
        self.assertEqual(chain.limit_reason(now,now,0),'UTC_DEADLINE')
        self.assertEqual(chain.limit_reason(now,future,57600),'GLOBAL_57600_SECONDS')
        self.assertEqual(chain.limit_reason(now,future,0,engineering_elapsed=900),'ENGINEERING_900_SECONDS')
        self.assertEqual(chain.limit_reason(now,future,0,7200,7200),'WORKER_WALL_LIMIT')

    def test_recycled_pid_never_terminated(self):
        psutil=FakePsutil();psutil.processes[123]=Process(123,2.);psutil.processes[456]=Process(456,4.)
        chain.terminate_owned(psutil,[{'pid':123,'create_time':1.},{'pid':456,'create_time':4.}])
        self.assertEqual(psutil.processes[123].terminated,0)
        self.assertEqual(psutil.processes[456].terminated,1)

    def test_projection_complete_profile_and_time_caps(self):
        result={'status':'SMOKE_COMPLETED','training_updates':2957,'elapsed_seconds':101.,'assessment_wall_seconds_total':20.,
                'metrics':[{'assessment_wall_seconds':10.},{'assessment_wall_seconds':10.}]}
        profiles={arm:(result,105.) for arm in chain.ARMS}
        projection=chain.profile_projection(profiles,57600)
        self.assertEqual(projection['status'],'PASS_RESOURCE_PROJECTION_ONLY')
        with self.assertRaises(ValueError):chain.profile_projection(profiles,100)
        with self.assertRaises(ValueError):chain.profile_projection({'L0':profiles['L0']},57600)
        bad=copy.deepcopy(result);bad['elapsed_seconds']=500.
        with self.assertRaises(ValueError):chain.profile_projection({arm:(bad,505.) for arm in chain.ARMS},57600)

    def test_unknown_gpu_probe_and_boolean_or_nan_bounds_rejected(self):
        self.assertEqual(chain.parse_gpu_pids('12\n12\n34\n'),[12,34])
        for output in ('N/A','error','0'):
            with self.assertRaises(ValueError):chain.parse_gpu_pids(output)
        for value in (True,float('nan'),-1):
            with self.assertRaises(ValueError):chain.finite_positive(value,'test')

    def test_cpu_trace_is_required_not_just_pair_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_,admission=fixture(Path(folder));cpu=admission[3]
            expected=chain.expected_hashes(plan,admission[0],cpu,'L0',17)
            self.assertEqual(expected['data_order_LR_trace_sha256'],cpu['expected_data_order_LR_trace_sha256']['88565']['17'])
            self.assertNotEqual(expected['data_order_LR_trace_sha256'],cpu['expected_data_order_LR_trace_sha256']['2957']['17'])

    def test_empty_cpu_cuda_mask_removed_for_approved_gpu_worker(self):
        env=chain.worker_environment({'CUDA_VISIBLE_DEVICES':'','X':'yes'},'source')
        self.assertNotIn('CUDA_VISIBLE_DEVICES',env);self.assertEqual(env['X'],'yes')
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')


if __name__=='__main__':unittest.main()
