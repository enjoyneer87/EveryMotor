"""Hold an unnamed Windows Job Object around the one read-only recovery worker."""
from pathlib import Path, PureWindowsPath
import _winapi
import ctypes as C
from ctypes import wintypes as W
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
K=C.WinDLL('kernel32',use_last_error=True)


class BasicLimits(C.Structure):
    _fields_=[('PerProcessUserTimeLimit',C.c_longlong),('PerJobUserTimeLimit',C.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',C.c_size_t),('MaximumWorkingSetSize',C.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',C.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class ExtendedLimits(C.Structure):
    _fields_=[('BasicLimitInformation',BasicLimits),('IoInfo',IO),('ProcessMemoryLimit',C.c_size_t),('JobMemoryLimit',C.c_size_t),('PeakProcessMemoryUsed',C.c_size_t),('PeakJobMemoryUsed',C.c_size_t)]
class Accounting(C.Structure):
    _fields_=[(n,C.c_longlong) for n in ('TotalUserTime','TotalKernelTime','ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime')]+[(n,W.DWORD) for n in ('TotalPageFaultCount','TotalProcesses','ActiveProcesses','TotalTerminatedProcesses')]


def api(name,args,result=W.BOOL):
    f=getattr(K,name);f.argtypes=args;f.restype=result;return f
create_job=api('CreateJobObjectW',[C.c_void_p,W.LPCWSTR],W.HANDLE)
set_job=api('SetInformationJobObject',[W.HANDLE,C.c_int,C.c_void_p,W.DWORD])
query_job=api('QueryInformationJobObject',[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p])
assign_job=api('AssignProcessToJobObject',[W.HANDLE,W.HANDLE])
terminate_job=api('TerminateJobObject',[W.HANDLE,W.UINT])
terminate_process=api('TerminateProcess',[W.HANDLE,W.UINT])
resume_thread=api('ResumeThread',[W.HANDLE],W.DWORD)
close_handle=api('CloseHandle',[W.HANDLE])


def run_in_job(command,seconds,memory_bytes):
    """Own all descendants before resuming the suspended first instruction."""
    assert seconds>0 and memory_bytes>0
    started=time.monotonic();job=create_job(None,None);assert job
    hp=ht=None;assigned=False;timed_out=False
    try:
        limits=ExtendedLimits();limits.BasicLimitInformation.LimitFlags=0x2000|0x200|0x100
        limits.JobMemoryLimit=limits.ProcessMemoryLimit=memory_bytes
        assert set_job(job,9,C.byref(limits),C.sizeof(limits)),C.get_last_error()
        hp,ht,pid,tid=_winapi.CreateProcess(None,subprocess.list2cmdline(command),None,None,False,0x4|subprocess.CREATE_NO_WINDOW,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),str(ROOT),subprocess.STARTUPINFO())
        assert assign_job(job,hp),C.get_last_error();assigned=True
        assert resume_thread(ht)!=0xFFFFFFFF,C.get_last_error();_winapi.CloseHandle(ht);ht=None
        while True:
            accounting=Accounting();assert query_job(job,1,C.byref(accounting),C.sizeof(accounting),None)
            if accounting.ActiveProcesses==0:break
            if time.monotonic()-started>=seconds:
                timed_out=True;assert terminate_job(job,124);break
            time.sleep(.025)
        if timed_out:
            _winapi.WaitForSingleObject(hp,2000)
            until=time.monotonic()+2
            while time.monotonic()<until:
                accounting=Accounting();assert query_job(job,1,C.byref(accounting),C.sizeof(accounting),None)
                if accounting.ActiveProcesses==0:break
                time.sleep(.025)
        peak=ExtendedLimits();assert query_job(job,9,C.byref(peak),C.sizeof(peak),None)
        return dict(pid=pid,returncode=_winapi.GetExitCodeProcess(hp),timed_out=timed_out,all_owned_job_processes_exited=accounting.ActiveProcesses==0,processes_created=int(accounting.TotalProcesses),peak_job_commit_bytes=int(peak.PeakJobMemoryUsed),elapsed_seconds=time.monotonic()-started,job_memory_limit_bytes=memory_bytes,suspended_before_job_assignment=True,retained_job_handle_scope_only=True)
    finally:
        if hp is not None:
            if assigned:terminate_job(job,125)
            else:terminate_process(hp,125)
            _winapi.WaitForSingleObject(hp,2000)
        if ht is not None:_winapi.CloseHandle(ht)
        if hp is not None:_winapi.CloseHandle(hp)
        close_handle(job)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    started=time.monotonic();contract_path=ROOT/'CONTOUR_LR20_RECOVERY_EXECUTION_CONTRACT.json';contract=json.loads(contract_path.read_bytes())
    assert contract['guard_source_sha256']==sha(Path(__file__)) and contract['max_seconds']==180
    worker=ROOT/'recover_closed_lr20_artifacts.py';assert sha(worker)==contract['recovery_source_sha256']
    guard_attempt=ROOT/'CONTOUR_LR20_RECOVERY_GUARD_ATTEMPT.json'
    with guard_attempt.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),UTC=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),contract_sha256=sha(contract_path),retry=0),f,indent=2)
    failure=ROOT/'CONTOUR_LR20_RECOVERY_GUARD_FAILURE.json'
    try:
        assert not (ROOT/'CONTOUR_LR20_RECOVERY_POINTER.json').exists() and not (ROOT/'CONTOUR_LR20_RECOVERY_FAILURE.json').exists()
        result=run_in_job([sys.executable,'-B','-X','utf8',str(worker)],180-(time.monotonic()-started)-6,1024**3)
        assert not result['timed_out'] and result['returncode']==0 and result['all_owned_job_processes_exited']
        assert result['peak_job_commit_bytes']<1024**3 and not (ROOT/'CONTOUR_LR20_RECOVERY_FAILURE.json').exists()
        pending_path=ROOT/'CONTOUR_LR20_RECOVERY_PENDING_POINTER.json';pending=json.loads(pending_path.read_bytes())
        assert PureWindowsPath(pending['path'])==PureWindowsPath(contract['receipt_path'])
        assert pending['worker_source_sha256']==contract['recovery_source_sha256'] and pending['execution_contract_sha256']==sha(contract_path)
        assert len(pending['sha256'])==64 and 0<pending['bytes']<=1024**2
        assert time.monotonic()-started<180
        receipt=dict(status='PASS_OWNED_JOB_BOUNDED_READ_ONLY_RECOVERY',source_sha256=sha(Path(__file__)),contract_sha256=sha(contract_path),pending_pointer_sha256=sha(pending_path),elapsed_seconds=time.monotonic()-started,worker=result,original_operational_status='INCOMPLETE_NO_CLAIM',process_ancestry='UNKNOWN',new_inference=0,remote_mutations=0,retry=0)
        rp=ROOT/'CONTOUR_LR20_RECOVERY_GUARD_RECEIPT.json'
        with rp.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
        pending['guard_receipt']=dict(path=str(rp),sha256=sha(rp),bytes=rp.stat().st_size)
        pending['status']='PENDING_EXTERNAL_GUARD_NORMAL_EXIT_OBSERVATION'
        with (ROOT/'CONTOUR_LR20_RECOVERY_POINTER.json').open('x',encoding='utf-8') as f:json.dump(pending,f,indent=2)
        assert time.monotonic()-started<180
        print(json.dumps(dict(status=receipt['status'],elapsed_seconds=receipt['elapsed_seconds'],new_inference=0)))
    except BaseException as e:
        with failure.open('x',encoding='utf-8') as f:json.dump(dict(status='INCOMPLETE_NO_CLAIM',error=repr(e),elapsed_seconds=time.monotonic()-started,retry=0),f,indent=2)
        raise


if __name__=='__main__':main()
