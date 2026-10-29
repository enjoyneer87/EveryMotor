"""Pure JSON/synthetic tests; no numpy, torch, native files, or remote tools."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import judge_a1_allocation as j


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode()


def digest(text):
    return j.sha(text.encode())


def simple_runs():
    keys = {j.PRIMARY, j.TORQUE} | {k for k, _ in j.NO_HARM.values()}
    return {f"{arm}_s{seed}": {"metrics": {key: (10. if arm == "L0" or key != j.PRIMARY else 8.) for key in keys}}
            for arm in j.ARMS for seed in j.SEEDS}


def evaluation(job, job_sha, training):
    errors = {"all":20., "a1":10. if job["arm"] == "L0" else 8., "copper":15.}
    fields, stats, pa = {}, {}, {}
    for phase, steps in j.PHASES.items():
        for region, nodes in (("all",4), ("a1",2), ("copper",1)):
            groups = [("", list(range(32)))]+[(f"/case:{c}", [c]) for c in range(32)]
            if region != "copper": groups += [("/PA_ge_70", list(range(16))), ("/PA_lt_70", list(range(16,32)))]
            for group, selected in groups:
                for weight in ("area", "unweighted"):
                    count = len(selected)*steps*nodes; ws = count*(.25 if weight == "area" else 1.)
                    key = f"{phase}{group}/{region}/{weight}"
                    ve, me = ws*(errors[region]/100)**2, ws*(errors[region]/200)**2
                    stats[key] = dict(count=count, weight_sum=ws, vector_sse=ve, magnitude_sse=me, truth_energy=ws)
                    fields[key] = dict(count=count, vector_nrmse_percent=errors[region], magnitude_nrmse_percent=errors[region]/2,
                        vector_rmse_T=errors[region]/100, magnitude_rmse_T=errors[region]/200, normalization_status="DEFINED")
            for label, selected in (("PA_ge_70",list(range(16))), ("PA_lt_70",list(range(16,32)))):
                pa[f"{phase}/{label}/{region}"] = dict(case_ids=selected, case_count=16,
                    mean_case_Bvector_nrmse_percent=errors[region], normalization_status="DEFINED")
    conductors, copper = [], {}
    for phase, count in j.PHASES.items():
        for c in range(32):
            conductors.append(dict(case=c,native_reg_code=3,phase_scope=phase,snapshots=count,mean_area_m2=.25,
                pred_spatial_mean_B2_T2=1.1,true_spatial_mean_B2_T2=1.,pred_square_spatial_mean_B_T2=.8,
                true_square_spatial_mean_B_T2=.8,B2_signed_relative_percent=10.))
        copper[phase] = dict(area_weighted_relative_MAE_percent=10.,signed_bias_percent=10.,
            weighted_absolute_error_T2_m2=.8,weighted_signed_error_T2_m2=.8,weighted_reference_T2_m2=8.,normalization_status="DEFINED")
    samples = [dict(case=c,step=s,pred_Nm=1.1,truth_Nm=1.) for c in range(32) for s in range(45)]
    result = dict(status="COMPLETED",id=job["id"],arm=job["arm"],alpha=job["alpha"],seed=job["seed"],
        job_sha256=job_sha,checkpoint_sha256=job["checkpoint"]["sha256"],graphs_processed=1440,
        training_updates=0,new_FEA=0,new_prior_solves=0,automatic_retries=0,phase_encoding="mechanical",
        metadata_hashes=job["metadata_hashes"],model_sha256=job["metadata_hashes"]["model_sha256"],evaluator_sha256=digest("evaluate_a1_allocation.py"),
        selection_sha256=job["selection"]["sha256"],bundle_manifest_sha256=job["bundle_manifest_sha256"],
        geometry_manifest_sha256=job["geometry_manifest"]["sha256"],training_mask_manifest_sha256=job["training_mask_manifest"]["sha256"],
        original_train_sha256=job["original_train"]["sha256"],
        reproduction_gate=dict(status="PASS_ARITHMETIC_ONLY",tolerance_pp=.0001,absolute_differences_pp={"seen9_training_anchor_pp":0.}),
        field_metrics=fields,field_sufficient_statistics=stats,PA_mean_case_metrics=pa,conductor_B2=conductors,copper_B2=copper,
        torque_samples=samples,case_torque={str(c):j.torque_summary([r for r in samples if r["case"]==c],True) for c in range(32)},
        pooled_torque=j.torque_summary(samples),mean_case_torque_nrmse_percent=j.torque_summary(samples)["nrmse_percent"],
        snapshot_errors=[dict(case=c,step=s,phase_scope="seen9" if s%5==0 else "unused36",vector_nrmse_percent=20.,magnitude_nrmse_percent=10.) for c in range(32) for s in range(45)],
        prediction_files=[dict(case_index=c,file=f"a1_case{c:04d}.npz",sha256=digest("prediction"+str(c)),bytes=100) for c in range(32)],
        historical_A0_reference=dict(status="NOT_REQUESTED"),verdict="PENDING_ALL_SIX_PREREGISTERED_JUDGMENT")
    return result


class Fixture:
    def __init__(self, root):
        self.package = root/"package"; self.results = root/"collection"
        self.package.mkdir(); self.results.mkdir(); self.files = []
        self.deadline = "2026-10-30T00:00:00+00:00"
        names = ["dual_solver_mgn.py","a1_allocation_loss.py","train_a1_allocation.py","evaluate_a1_allocation.py","run_a1_chain.py"]
        sources = [dict(path=j.STAGE+"/src/"+n,sha256=digest(n),bytes=100) for n in names]
        source_map = j.source_map(sources)
        selection = {"records": [dict(case_index=1000+c,split="train",geometry_group=f"train{c%288}",source_sha256=digest(str(1000+c))) for c in range(328)]
            +[dict(case_index=c,split="development",geometry_group=f"dev{c}",source_sha256=digest(str(c)),native_input={"PhaseAdvance":80 if c<16 else 60}) for c in range(32)]}
        selection_pin = self.static("metadata/selection.json",selection)
        geometry = dict(status="ADMITTED",selection_sha256=selection_pin["sha256"],bundle_manifest_sha256=digest("devbundle"),
            records=[dict(case_index=c,status="ADMITTED",steps_checked=45,source_sha256=digest(str(c)),a1_native_element_ids=[1,2]) for c in range(32)])
        geometry_pin = self.static("metadata/geometry.json",geometry)
        masks = dict(status="ADMITTED",selection_sha256=selection_pin["sha256"],bundle_manifest_sha256=digest("trainbundle"),
            dev_geometry_manifest_sha256=geometry_pin["sha256"],common_dev_bundle_sha256=digest("devbundle"),
            train_steps=list(range(0,45,5)),train_graphs=2952,development_graphs=288,records=[],development_records=[])
        for row in selection["records"]:
            item = dict(row,status="ADMITTED",operator_sha256=digest("op"+str(row["case_index"])),a1_native_element_ids=[1,2],
                graphs=[dict(step=s,sha256=digest("graph"),native_element_id_sha256=digest("ids"),a1_mask_sha256=digest("mask"),
                    a1_count=2,all_count=4,area_a1_m2=.5,area_all_m2=1.,q=.5,effective_a1_allocation_alpha05=.75) for s in range(0,45,5)])
            masks["records" if row["split"]=="train" else "development_records"].append(item)
        mask_pin = self.static("metadata/masks.json",masks)
        cpu = dict(status="PASS",source_files=sources[:3],initial_state_sha256_by_seed={str(s):digest("initial"+str(s)) for s in j.SEEDS},
            expected_data_order_LR_trace_sha256={str(u):{str(s):digest(f"{u}/{s}") for s in j.SEEDS} for u in (2957,88565)})
        cpu_pin = self.static("CPU_CONTRACT_RECEIPT.json",cpu)
        proposal = dict(arms={"L0":{"alpha":0.},"L05":{"alpha":.5}},
            fixed=dict(seeds=list(j.SEEDS),updates_per_run=88565,full_runs=6,width=32,depth=3,angle="mechanical",selection_sha256=selection_pin["sha256"]),
            proposed_decision_rule_to_freeze_before_release=dict(mean_no_harm_L05_minus_L0_pp={k:v[1] for k,v in j.NO_HARM.items()},
                primary_improvement="delta>=1.0pp AND delta>2*spread",
                torque_improvement_claim="Requires its own delta>=1pp AND delta>2*torque_spread; primary a1 success alone does not imply torque success"))
        proposal_pin = self.static("proposal.json",proposal); self.proposal_sha = proposal_pin["sha256"]
        training = dict(status="RELEASED_FIXED_SIX_RUNS",arms=proposal["arms"],fixed_seeds=list(j.SEEDS),seen_steps=list(range(0,45,5)),
            source_files=sources,guard_deadline_UTC=self.deadline,initial_state_sha256_by_seed=cpu["initial_state_sha256_by_seed"],
            selection=selection_pin,bundle_manifest_sha256=digest("trainbundle"),mask_manifest_sha256=mask_pin["sha256"],
            training=dict(width=32,depth=3,expected_optimizer_updates=88565,updates_per_A0_epoch=2952,smoke_updates=5,
                learning_rate={"start":.001,"end":.00001},evaluation_updates={"0":5,"1":2957,"5":14765,"10":29525,"20":59045,"30":88565},
                retry=0,wall_seconds_max=7200,torch_allocated_soft_limit_GiB=10))
        training_pin = self.static("TRAINING_PLAN.json",training)
        engineering = dict(training,status="RELEASED_ENGINEERING_ONLY")
        engineering_pin = self.static("ENGINEERING_PLAN.json",engineering)
        template = dict(source_files=sources,deadline_UTC=self.deadline,selection=selection_pin,training_mask_manifest=mask_pin,
            release_status="RELEASED_FROZEN_EVALUATION",max_seconds=1800,workers=1,retry_limit=0,phase_encoding="mechanical",
            training_updates=0,new_FEA=0,new_prior_solves=0,geometry_manifest=geometry_pin,bundle_manifest_sha256=digest("devbundle"))
        template_pin = self.static("EVALUATION_TEMPLATE.json",template)
        self.plan = dict(status="RELEASED_ENGINEERING_THEN_FIXED_SIX_RUNS",workers=1,retry_limit=0,global_max_seconds=57600,
            engineering_max_seconds=900,engineering_updates=2957,train_max_seconds=7200,eval_max_seconds=1800,
            projection_safety_factor=1.25,collection_reserve_seconds=300,disk_free_min_bytes=10*1024**3,
            run_order=list(j.PAIRED_ORDER),stage_root=j.STAGE,runs_root=j.STAGE+"/runs",stop_files=[j.STAGE+"/STOP"],deadline_UTC=self.deadline,
            source_files=sources,trainer=sources[2],evaluator=sources[3],guard=sources[4],training_plan=training_pin,
            engineering_plan=engineering_pin,cpu_admission=cpu_pin,mask_manifest=mask_pin,evaluation_template=template_pin)
        plan_pin = self.static("CHAIN_PLAN.json",self.plan); self.plan_sha = plan_pin["sha256"]
        owner=dict(pid=100,create_time=100.)
        projection=dict(status="PASS_RESOURCE_PROJECTION_ONLY",arms={a:dict(safety_factor=1.25,projected_train_seconds=1000.) for a in j.ARMS},
            projected_remaining_seconds=17100.,available_remaining_seconds=57000.,evaluation_reserve_seconds=10800,collection_reserve_seconds=300)
        status=dict(status="COMPLETED",chain_plan_sha256=self.plan_sha,plan_sha256=self.plan_sha,owner=owner,started_UTC=self.time(0),completed_UTC=self.time(30),
            elapsed_seconds=30.,scientific_verdict="PENDING_PREREGISTERED_JSON_JUDGMENT",completed_train_runs=6,completed_eval_runs=6,
            completed_engineering_runs=2,full_training_updates=531390,engineering_updates=5914,evaluation_graphs=8640,new_FEA=0,new_prior_solves=0,
            automatic_retries=0,engineering_projection=projection,jobs=[])
        trains={}
        alljobs=[("L0_s17","engineering"),("L05_s17","engineering")]+[(r,k) for r in j.PAIRED_ORDER for k in ("train","eval")]
        for i,(run,kind) in enumerate(alljobs):
            arm,seed=run.split("_s");seed=int(seed);alpha=0. if arm=="L0" else .5
            ident="engineering_"+run if kind=="engineering" else run+"_"+kind
            sub="engineering/"+run if kind=="engineering" else run+"/"+kind
            entry=dict(id=ident,run_id=run,kind=kind,arm=arm,seed=seed,alpha=alpha,status="COMPLETED",returncode=0,
                worker_status="SMOKE_COMPLETED" if kind=="engineering" else "COMPLETED",out=j.STAGE+"/runs/"+sub,
                source_sha256_by_name=source_map,owner=dict(pid=101+i,create_time=101.+i),started_UTC=self.time(i*2+1),completed_UTC=self.time(i*2+2),
                elapsed_seconds=1.,checkpoint_sha256=digest("checkpoint"+run+kind))
            expected=j.expected_metadata(self.plan,training,cpu,source_map,seed,kind=="engineering")
            if kind!="eval":
                epochs=[0,1] if kind=="engineering" else [0,1,5,10,20,30];updates=2957 if kind=="engineering" else 88565
                result=dict(status=entry["worker_status"],arm=arm,seed=seed,alpha=alpha,training_updates=updates,**expected,
                    checkpoint_sha256=entry["checkpoint_sha256"],parameters=70182,field_path_parameters=30594,phase_encoding="mechanical",data_arm="A0",
                    old_checkpoint_reused=False,new_FEA=0,new_prior_solves=0,original_val_test_used=0,train_steps=list(range(0,45,5)),train_graphs=2952,
                    train_conditions=328,development_conditions=32,train_groups=sorted({r["geometry_group"] for r in selection["records"] if r["split"]=="train"}),
                    development_groups=sorted({r["geometry_group"] for r in selection["records"] if r["split"]=="development"}),GPU_smoke="PASS",
                    peak_torch_allocated_bytes=100,max_updates=2957 if kind=="engineering" else None,
                    metrics=[dict(epoch=e,training_updates=training["training"]["evaluation_updates"][str(e)],assessment_wall_seconds=.01,
                        development={"all":dict(snapshots=288,vector_nRMSE_percent=20.)}) for e in epochs],assessment_wall_seconds_total=.01*len(epochs))
                entry.update(plan_sha256=expected["plan_sha256"],training_updates=updates,
                    initial_tensor_state_sha256=expected["initial_tensor_state_sha256"],data_order_LR_trace_sha256=expected["data_order_LR_trace_sha256"])
                pin=self.collect("runs/"+sub+"/RESULT.json",result)
                if kind=="train":trains[run]=(result,pin)
            else:
                result_train,train_pin=trains[run]
                job=dict(template,id=run,arm=arm,seed=seed,alpha=alpha,metadata_hashes=expected,original_train=train_pin,
                    checkpoint=dict(path=j.STAGE+"/runs/"+run+"/train/last.pt",sha256=result_train["checkpoint_sha256"],bytes=100))
                job_pin=self.collect("runs/"+run+"/eval_job.json",job)
                result=evaluation(job,job_pin["sha256"],result_train)
                pin=self.collect("runs/"+sub+"/RESULT.json",result)
                entry.update(job_sha256=job_pin["sha256"],checkpoint_sha256=result_train["checkpoint_sha256"],graphs_processed=1440)
            entry["result_sha256"]=pin["sha256"]
            status["jobs"].append(entry);self.collect("runs/"+ident+"_GUARD_RECEIPT.json",entry)
        self.collect("runs/CHAIN_STATUS.json",status)
        self.collect("runs/CHAIN_RECEIPT.json",{k:v for k,v in status.items() if k!="jobs"})
        self.collect("runs/GUARD_OWNER.json",dict(owner,chain_plan_sha256=self.plan_sha))
        self.collect("runs/ENGINEERING_PROJECTION_RECEIPT.json",projection)
        self.refresh_collection()

    @staticmethod
    def time(second):return f"2026-10-29T00:00:{second:02d}+00:00"

    def static(self,relative,value):
        return self.write(self.package,relative,value)

    def collect(self,relative,value):
        pin=self.write(self.results,relative,value)
        self.files.append(dict(relative=relative,sha256=pin["sha256"],bytes=pin["bytes"]))
        return pin

    @staticmethod
    def write(root,relative,value):
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);data=encoded(value);path.write_bytes(data)
        return dict(path=j.STAGE+"/"+relative,sha256=j.sha(data),bytes=len(data))

    def refresh_collection(self):
        snapshot=dict(status="TERMINAL_READY",terminal_status="COMPLETED",guard_plan_sha256=self.plan_sha,live_owned=[],binary_payloads_opened_or_hashed=0)
        receipt=dict(status="VERIFIED",terminal_status="COMPLETED",guard_plan_sha256=self.plan_sha,raw_or_checkpoint_files_transferred=0,
            automatic_retries=0,hash_size_failures=0,files=self.files,remote_snapshot_before=snapshot,remote_snapshot_after=snapshot)
        self.write(self.results,"COLLECTION_RECEIPT.json",receipt)

    def run(self):
        with patch.object(j,"PROPOSAL_SHA",self.proposal_sha):
            return j.judge(self.results,self.package,self.package/"CHAIN_PLAN.json",self.plan_sha,self.package/"proposal.json")


class JudgmentTests(unittest.TestCase):
    def test_01_preregistered_boundaries(self):
        runs=simple_runs()
        for seed in j.SEEDS:runs[f"L05_s{seed}"]["metrics"][j.PRIMARY]=9.
        self.assertEqual(j.decide(runs)[0]["verdict"],"IMPROVEMENT_CLAIM_CRITERIA_MET")
        runs["L0_s17"]["metrics"][j.PRIMARY]=9.5
        runs["L0_s43"]["metrics"][j.PRIMARY]=10.5
        self.assertFalse(j.decide(runs)[0]["primary"]["spread_pass"])
        for seed in j.SEEDS:runs[f"L05_s{seed}"]["metrics"][j.PRIMARY]=8.
        self.assertEqual(j.decide(runs)[0]["primary"]["delta_L0_minus_L05_pp"],2.)
        self.assertFalse(j.decide(runs)[0]["primary"]["spread_pass"])
        for seed in j.SEEDS:runs[f"L05_s{seed}"]["metrics"][j.PRIMARY]=7.999
        self.assertTrue(j.decide(runs)[0]["primary"]["spread_pass"])
        for seed in j.SEEDS:
            runs[f"L0_s{seed}"]["metrics"][j.PRIMARY]=10.
            runs[f"L05_s{seed}"]["metrics"][j.PRIMARY]=9.001
            runs[f"L05_s{seed}"]["metrics"][j.TORQUE]=8.
        self.assertEqual(j.decide(runs)[0]["verdict"],"NO_IMPROVEMENT_CLAIM")
        self.assertTrue(j.decide(runs)[0]["torque_metric_improvement_gate"]["pass_"])

    def test_02_inclusive_mean_noharm_and_separate_torque(self):
        runs=simple_runs();key=j.NO_HARM["seen9_all_Bvector"][0]
        for seed in j.SEEDS:runs[f"L05_s{seed}"]["metrics"][key]=10.5
        decision=j.decide(runs)[0]
        self.assertTrue(decision["no_harm_pass"]);self.assertFalse(decision["torque_metric_improvement_gate"]["pass_"])
        runs["L05_s17"]["metrics"][key]=11.5;runs["L05_s29"]["metrics"][key]=10.;runs["L05_s43"]["metrics"][key]=10.
        self.assertTrue(j.decide(runs)[0]["no_harm_pass"])
        runs["L05_s17"]["metrics"][key]+=0.001
        self.assertEqual(j.decide(runs)[0]["verdict"],"NO_IMPROVEMENT_CLAIM")

    def test_03_missing_or_undefined_never_reduces_seed_count(self):
        for corruption in ("missing","null","nan"):
            runs=simple_runs()
            if corruption=="missing":del runs["L05_s43"]
            else:runs["L05_s43"]["metrics"][j.PRIMARY]=None if corruption=="null" else float("nan")
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):j.decide(runs)

    def test_04_full_json_completion_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(Path(tmp));result=fixture.run()
            self.assertEqual(result["errors"],[])
            self.assertEqual(result["passed_run_count"],6)
            self.assertEqual(result["verdict"],"IMPROVEMENT_CLAIM_CRITERIA_MET")
            self.assertFalse(result["decision"]["torque_metric_improvement_gate"]["pass_"])
            self.assertEqual(result["arms"]["L0"][j.PRIMARY]["expected_count"],3)
            self.assertEqual(len(result["paired"][j.PRIMARY]["paired"]),3)
            for key in ("NPZ_opened","checkpoints_loaded","model_imports","training_updates","new_evaluation_runs","remote_calls"):
                self.assertEqual(result["operations"][key],0)

    def test_05_hash_mutation_and_missing_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(Path(tmp));path=fixture.results/"runs/L05_s43/eval/RESULT.json";original=path.read_bytes()
            path.write_bytes(original+b" ")
            result=fixture.run();self.assertEqual(result["verdict"],"INCOMPLETE_NO_CLAIM");self.assertNotIn("decision",result)
            path.write_bytes(original);(fixture.results/"runs/L05_s43_eval_GUARD_RECEIPT.json").unlink()
            self.assertEqual(fixture.run()["verdict"],"INCOMPLETE_NO_CLAIM")

    def test_06_provenance_initial_order_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(Path(tmp));inventory,_=j.collection_inventory(fixture.results,fixture.plan_sha)
            evidence=j.Evidence(fixture.results,fixture.package,inventory)
            trainplan=evidence.pinned(fixture.plan["training_plan"]);cpu=evidence.pinned(fixture.plan["cpu_admission"])
            selection=evidence.pinned(trainplan["selection"]);sources=j.source_map(fixture.plan["source_files"])
            result,_=evidence.result("runs/L0_s17/train/RESULT.json");entry,_=evidence.result("runs/L0_s17_train_GUARD_RECEIPT.json")
            expected=j.expected_metadata(fixture.plan,trainplan,cpu,sources,17)
            for key in ("initial_tensor_state_sha256","data_order_LR_trace_sha256","trainer_sha256","mask_manifest_sha256","checkpoint_sha256"):
                bad=copy.deepcopy(result);bad[key]="f"*64
                with self.subTest(key=key),self.assertRaises(ValueError):j.check_training(bad,entry,expected,selection)
            bad=copy.deepcopy(entry);bad["source_sha256_by_name"]["dual_solver_mgn.py"]="e"*64
            with self.assertRaises(ValueError):j.check_receipt(bad,bad,"L0_s17","train",sources,j.utc_value(fixture.deadline))

    def test_07_duplicate_snapshot_and_arithmetic_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(Path(tmp));inventory,_=j.collection_inventory(fixture.results,fixture.plan_sha)
            evidence=j.Evidence(fixture.results,fixture.package,inventory)
            result,_=evidence.result("runs/L0_s17/eval/RESULT.json");job,pin=evidence.result("runs/L0_s17/eval_job.json");train,_=evidence.result("runs/L0_s17/train/RESULT.json")
            for corruption in ("duplicate","PA","B2","anchor"):
                bad=copy.deepcopy(result)
                if corruption=="duplicate":bad["torque_samples"][-1]=bad["torque_samples"][0]
                elif corruption=="PA":bad["PA_mean_case_metrics"]["all45/PA_ge_70/all"]["mean_case_Bvector_nrmse_percent"]+=1.
                elif corruption=="B2":bad["copper_B2"]["all45"]["signed_bias_percent"]+=1.
                else:bad["reproduction_gate"]["tolerance_pp"] = .1
                with self.subTest(corruption=corruption),self.assertRaises(ValueError):j.check_new_result(bad,job,pin,train,list(range(32)),set(range(16)),{c:2 for c in range(32)})

    def test_08_no_source_execution_or_raw_read_imports(self):
        import ast
        tree=ast.parse(Path(j.__file__).read_text(encoding="utf-8"))
        imports={n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
        self.assertFalse(imports & {"torch","numpy","subprocess","paramiko","requests"})
        for value in ("../escape.json","runs/../escape.json","C:/escape.json","/absolute.json","runs\\bad.json"):
            with self.assertRaises(ValueError):j.safe_relative(value)

    def test_09_json_rejects_duplicates_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.json"
            for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
                path.write_text(text,encoding="utf-8")
                with self.assertRaises(ValueError):j.read_json(path)


if __name__=="__main__":unittest.main()
