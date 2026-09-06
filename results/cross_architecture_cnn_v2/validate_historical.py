"""Run historical verifiers unchanged, redirecting only newly written reports."""
import sys, json, runpy, traceback, contextlib, io, os, hashlib, subprocess
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def snapshot():
    result={}
    excluded={'.git','.venv-gpu','cross_architecture_cnn_v1','cross_architecture_cnn_v2','__pycache__'}
    for folder,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in excluded]
        for name in files:
            p=Path(folder)/name; st=p.stat()
            record={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
            if st.st_size<10_000_000 or p.suffix in {'.csv','.json','.md','.txt','.py','.pdf','.tex'}:
                record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            result[str(p.relative_to(ROOT))]=record
    return result
if __name__=='__main__':
    before=snapshot()
    with (OUT/'historical_file_inventory.json').open('x') as f: json.dump(before,f,indent=2,sort_keys=True)
    reports=OUT/'historical_validation'; reports.mkdir(exist_ok=False)
    original=Path.write_text
    def redirected(self,data,*args,**kwargs):
        if self.is_relative_to(ROOT/'results') and not self.is_relative_to(OUT):
            target=reports/self.relative_to(ROOT/'results')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise FileExistsError(target)
            return original(target,data,*args,**kwargs)
        raise RuntimeError('Unexpected verifier write: '+str(self))
    results=[]
    for name in ['verify_epsilon_sensitivity.py','verify_adamw_state_coherence.py','verify_reviewer2026_v5_final_eval.py']:
        stream=io.StringIO(); status='PASS'
        try:
            Path.write_text=redirected
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                runpy.run_path(str(ROOT/'scripts'/name),run_name='__main__')
        except BaseException as exc:
            if isinstance(exc,SystemExit) and exc.code in (None,0): pass
            else: status='FAIL'; stream.write(traceback.format_exc())
        finally: Path.write_text=original
        with (reports/(name+'.log')).open('x') as f: f.write(stream.getvalue())
        results.append({'verifier':name,'status':status,'source_sha256':hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest(),'output':stream.getvalue()})
        print(name,status,flush=True)
    assert before==snapshot(),'Historical files changed!'
    with (OUT/'historical_validation_results.json').open('x') as f: json.dump({'results':results,'historical_files_unchanged':True},f,indent=2)

