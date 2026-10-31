"""Field-agnostic retained-handle child supervision, with an independent child timer."""
import subprocess
import time


def run_owned_child(command, *, admit, check, identify, identity_matches, cwd, stdout, stderr,
                    runner_seconds=360, clock=time.monotonic, sleep=time.sleep, popen=subprocess.Popen):
    child=None
    result=dict(status='STARTING',child_started=False,child_exit_verified=True)
    try:
        admit()  # STOP, time, completed training identities and GPU checks precede Popen.
        child=popen(command,cwd=cwd,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result.update(child_started=True,spawned_pid=child.pid,status='RUNNING')
        child_started=clock();result['child']=identify(child)
        while child.poll() is None:
            check()
            if clock()-child_started>=runner_seconds:raise TimeoutError('owned runner360 deadline')
            if not identity_matches(result['child']):
                if child.poll() is not None:break
                raise RuntimeError('owned child identity lost')
            sleep(1)
        result['returncode']=child.wait()
        assert result['returncode']==0,'owned child exited nonzero'
        result['status']='CHILD_COMPLETED'
    except BaseException as exc:
        result.update(status='INCOMPLETE_NO_CLAIM',error=repr(exc))
    finally:
        errors=[]
        if child is not None and child.poll() is None:
            try:child.terminate()
            except BaseException as exc:errors.append('terminate: '+repr(exc))
            try:child.wait(timeout=10)
            except BaseException as exc:
                errors.append('wait_after_terminate: '+repr(exc))
                try:child.kill()
                except BaseException as failure:errors.append('kill: '+repr(failure))
                try:child.wait(timeout=10)
                except BaseException as failure:errors.append('wait_after_kill: '+repr(failure))
        if errors:result.update(status='INCOMPLETE_NO_CLAIM',cleanup_errors=errors)
        result['child_exit_verified']=child is None or child.poll() is not None
        if not result['child_exit_verified']:result['status']='INCOMPLETE_NO_CLAIM'
    return result
