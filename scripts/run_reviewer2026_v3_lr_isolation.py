from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, platform, random, subprocess, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'experiments'/'scripts'), str(ROOT/'experiments')]
import run_reviewer2026_v2 as v2  # noqa: E402
import reviewer2026_v2_heavy_campaigns as heavy  # noqa: E402
import run_opt2026_10m_gpu_final as base  # noqa: E402
OUT=ROOT/'results'/'reviewer2026_v3_lr_isolation'; V2=ROOT/'results'/'reviewer2026_v2'; SRC=ROOT/'results'/'opt2026_10m_gpu_final'
H=100; H20=20; EPS=1e-12; SOURCE_STARTING_COMMIT='a127083830cc7ac6f7ade865397362df048c0a05'
PRIMARY_SEEDS=[1101,1102,1103]; PRIMARY_AGES=[75,150,300]; CONTROLLED_SEEDS=[1201,1202,1203]; CONTROLLED_AGES=[75,150,300]; LONG_SEEDS=[1301,1302,1303]; LONG_AGES=[750,1500,3000]; FUTURE_SCHEDULE_SEEDS=[8101,8102,8103,8104,8105]
METRICS={'global_normalized_parameter_divergence':'max_normalized_parameter_divergence','clean_update_normalized_divergence':'max_clean_update_normalized_divergence','max_layer_relative_displacement':'max_layer_relative_displacement','function_space_kl':'max_calibration_kl','function_space_rms_logit':'max_calibration_rms_logit'}

def to_jsonable(x:Any)->Any:
    if isinstance(x,Path): return str(x)
    if torch.is_tensor(x): return x.detach().cpu().item() if x.numel()==1 else x.detach().cpu().tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,float): return x if math.isfinite(x) else None
    if isinstance(x,dict): return {str(k):to_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [to_jsonable(v) for v in x]
    return x

def write_json(p:Path,o:Any)->None: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(to_jsonable(o),indent=2,sort_keys=True),encoding='utf-8')
def write_md(p:Path,t:str)->None: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(t.strip()+'\n',encoding='utf-8')
def read_json(p:Path)->Any: return json.loads(p.read_text(encoding='utf-8'))
def write_csv(p:Path,rows:list[dict[str,Any]])->None:
    p.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def run_cmd(args:list[str])->str:
    try: return subprocess.check_output(args,cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as e: return f'ERROR: {e}'
class sched_step:
    def __init__(self,n:int): self.n=n; self.old=None
    def __enter__(self): self.old=base.SCHED_STEP; base.SCHED_STEP=self.n
    def __exit__(self,*a): base.SCHED_STEP=self.old

def public_group(g:dict[str,Any])->dict[str,Any]: return {k:v for k,v in g.items() if k!='params'}
def public_groups(sd:dict[str,Any])->list[dict[str,Any]]: return [public_group(g) for g in sd.get('param_groups',[])]
def groups_match(opt:torch.optim.Optimizer, ckpt:dict[str,Any])->bool: return [public_group(g) for g in opt.state_dict()['param_groups']]==public_groups(ckpt)
def opt_state_entries(opt:torch.optim.Optimizer)->int: return sum(1 for s in opt.state.values() if s)
def opt_state_keys(opt:torch.optim.Optimizer)->list[str]:
    ks=set()
    for s in opt.state.values(): ks.update(str(k) for k in s)
    return sorted(ks)
def load_optimizer_param_groups_empty_state(opt:torch.optim.Optimizer, ckpt:dict[str,Any])->None:
    opt.load_state_dict({'state':{},'param_groups':copy.deepcopy(ckpt['param_groups'])})
def fresh_from_payload_optimizer_state_reset_isolated(payload:dict[str,Any]):
    model,opt,sched=base.fresh_from_payload(payload,load_optimizer=False,load_scheduler=True)
    ckpt=payload['optimizer_state']; load_optimizer_param_groups_empty_state(opt,ckpt); base.optimizer_state_to_device(opt)
    audit={'checkpoint_param_groups':public_groups(ckpt),'candidate_param_groups':[public_group(g) for g in opt.state_dict()['param_groups']],'checkpoint_optimizer_lr':ckpt['param_groups'][0]['lr'],'candidate_optimizer_lr':opt.param_groups[0]['lr'],'lr_matches_checkpoint':opt.param_groups[0]['lr']==ckpt['param_groups'][0]['lr'],'param_groups_match_checkpoint':groups_match(opt,ckpt),'scheduler_state_matches_checkpoint':copy.deepcopy(sched.state_dict())==payload['scheduler_state'],'initial_optimizer_state_entries':opt_state_entries(opt),'initial_optimizer_state_keys':opt_state_keys(opt)}
    return model,opt,sched,audit

def p_primary(seed:int,age:int)->Path: return SRC/'gpu_scale'/'checkpoints'/f'seed_{seed}'/f'checkpoint_step_{age}.pt'
def p_control(cfg:str,seed:int,age:int)->Path: return V2/'controlled_scale'/'checkpoints'/cfg/f'seed_{seed}'/f'checkpoint_step_{age}.pt'
def p_long(seed:int,age:int)->Path: return V2/'long_training'/'checkpoints'/'candidate_B_19p48m'/f'seed_{seed}'/f'checkpoint_step_{age}.pt'

def audit_legacy_reset()->None:
    rows=[]
    specs=[]
    specs += [('primary_19p48m','candidate_B_19p48m',s,a,p_primary(s,a),100) for s in PRIMARY_SEEDS for a in PRIMARY_AGES]
    specs += [('controlled_scale',c['name'],s,a,p_control(c['name'],s,a),100) for c in v2.SCALE_CONFIGS for s in CONTROLLED_SEEDS for a in CONTROLLED_AGES]
    specs += [('long_training','candidate_B_19p48m',s,a,p_long(s,a),1000) for s in LONG_SEEDS for a in LONG_AGES]
    for camp,cfg,seed,age,path,ss in specs:
        with sched_step(ss):
            payload=base.load_checkpoint(path); scen=base.scenario_payload(payload,'optimizer_reset',None); _m,oopt,osched=base.fresh_from_payload(scen,load_optimizer=False,load_scheduler=True)
        ckpt=payload['optimizer_state']; rows.append({'campaign':camp,'model_size':cfg,'seed':seed,'checkpoint_age':age,'checkpoint_file':str(path.relative_to(ROOT)),'checkpoint_optimizer_lr':ckpt['param_groups'][0]['lr'],'old_fresh_adamw_lr':oopt.param_groups[0]['lr'],'old_lr_equals_checkpoint_lr':oopt.param_groups[0]['lr']==ckpt['param_groups'][0]['lr'],'checkpoint_param_group':public_group(ckpt['param_groups'][0]),'loaded_scheduler_state_old':osched.state_dict(),'checkpoint_scheduler_state':payload['scheduler_state']})
    write_json(OUT/'legacy_reset_audit.json',{'status':'COMPLETE','reviewer_concern_confirmed':any(not r['old_lr_equals_checkpoint_lr'] for r in rows),'rows':rows,'mismatches':[r for r in rows if not r['old_lr_equals_checkpoint_lr']]})

def same_model(a,b)->bool:
    with torch.no_grad(): return all(torch.equal(pa.detach().cpu(),pb.detach().cpu()) for pa,pb in zip(a.parameters(),b.parameters()))
def run_isolation_tests()->None:
    path=p_primary(1101,150); batches,_,dataset=base.load_batches(1101,base.TRAIN_STEPS+H+50)
    with sched_step(100):
        payload=base.load_checkpoint(path); cm,co,cs=base.fresh_from_payload(payload); xm,xo,xs,audit=fresh_from_payload_optimizer_state_reset_isolated(payload)
        init_empty=opt_state_entries(xo)==0 and opt_state_keys(xo)==[]; group_ok=groups_match(xo,payload['optimizer_state']); sched_ok=xs.state_dict()==payload['scheduler_state']; model_ok=same_model(cm,xm); clean_ok=groups_match(co,payload['optimizer_state'])
        x0,y0=batches[150]; cl=base.train_one(cm,co,cs,batches[150]); xl=base.train_one(xm,xo,xs,batches[150]); same_batch=torch.equal(x0,batches[150][0]) and torch.equal(y0,batches[150][1])
        result={'status':'PASS' if all([init_empty,group_ok,sched_ok,model_ok,clean_ok,same_batch,audit['lr_matches_checkpoint']]) else 'FAIL','checkpoint_file':str(path.relative_to(ROOT)),'dataset':dataset,'model_parameters_equal_checkpoint':model_ok,'candidate_lr_equals_checkpoint_lr':audit['lr_matches_checkpoint'],'candidate_param_groups_match_checkpoint':group_ok,'candidate_initial_state_empty_before_step':init_empty,'candidate_initial_state_keys_before_step':[],'scheduler_state_matches_checkpoint':sched_ok,'clean_reference_full_resume_param_groups_match':clean_ok,'same_training_batch_consumed':same_batch,'clean_loss_after_one_step':cl,'candidate_loss_after_one_step':xl,'candidate_state_keys_after_one_step':opt_state_keys(xo),'difference_source':'missing persistent AdamW per-parameter state only'}
    write_json(OUT/'optimizer_reset_isolation_tests.json',result)
    if result['status']!='PASS': raise RuntimeError('isolation tests failed')
def run_pair_isolated(payload:dict[str,Any],batches:list[Any],val_batches:list[Any],calibration_batch:Any,future_start:int|None=None,instrument:bool=False):
    cm,co,cs=base.fresh_from_payload(payload); xm,xo,xs,audit=fresh_from_payload_optimizer_state_reset_isolated(payload)
    start=int(payload['batch_pos']) if future_start is None else future_start; theta=v2.norm_model(cm); tau=v2.TAU_FRAC*(theta+EPS)
    cumc=cumx=cumd=0.0; steps=[]; mech=[]; divs=[]; ndivs=[]; vclean=[]; vcand=[]; gaps=[]; pgaps=[]; lrs=[]; xlrs=[]; upd_metric=[0.0]; max_layer=[]; mean_layer=[]; kls=[]; rms=[]
    d0=v2.l2_from_params(cm,xm); layers0=v2.layer_stats(cm,xm); f0=v2.function_metrics(cm,xm,calibration_batch)
    divs.append(d0); ndivs.append(d0/(theta+EPS)); max_layer.append(layers0['max_layer_relative_displacement']); mean_layer.append(layers0['mean_layer_relative_displacement']); kls.append(f0['calibration_kl']); rms.append(f0['calibration_rms_logit'])
    steps.append({'continuation_step':0,'parameter_divergence':d0,'normalized_parameter_divergence':d0/(theta+EPS),'validation_loss_gap':0.0,'perplexity_gap':0.0,'lr':co.param_groups[0]['lr'],'candidate_lr':xo.param_groups[0]['lr'],'source_task_loss':'','candidate_task_loss':'','clean_cumulative_update_distance':cumc,'candidate_cumulative_update_distance':cumx,'cumulative_update_discrepancy':cumd,**layers0,**f0})
    for k in range(1,H+1):
        batch=batches[(start+k-1)%len(batches)]; pc=v2.snapshot_named(cm); px=v2.snapshot_named(xm); fpc=v2.flat_gpu(cm); fpx=v2.flat_gpu(xm)
        cl=base.train_one(cm,co,cs,batch); xl=base.train_one(xm,xo,xs,batch); fc=v2.flat_gpu(cm); fx=v2.flat_gpu(xm)
        uc=fc-fpc; ux=fx-fpx; du=uc-ux; un=float(uc.norm().detach().cpu().item()); xn=float(ux.norm().detach().cpu().item()); dn=float(du.norm().detach().cpu().item()); cos=float((torch.dot(uc,ux)/(uc.norm()*ux.norm()+EPS)).detach().cpu().item()) if un>0 and xn>0 else math.nan
        cumc+=un; cumx+=xn; cumd+=dn; del fpc,fpx,fc,fx,uc,ux,du
        div=v2.l2_from_params(cm,xm); vlc=sum(v2.validation_loss(cm,b) for b in val_batches)/len(val_batches); vlx=sum(v2.validation_loss(xm,b) for b in val_batches)/len(val_batches); gap=abs(vlc-vlx); pg=abs(math.exp(min(vlc,50))-math.exp(min(vlx,50)))
        lay=v2.layer_stats(cm,xm,pc,px); fm=v2.function_metrics(cm,xm,calibration_batch); lr=co.param_groups[0]['lr']; xlr=xo.param_groups[0]['lr']
        divs.append(div); ndivs.append(div/(theta+EPS)); vclean.append(vlc); vcand.append(vlx); gaps.append(gap); pgaps.append(pg); lrs.append(lr); xlrs.append(xlr); upd_metric.append(div/(cumc+EPS)); max_layer.append(lay['max_layer_relative_displacement']); mean_layer.append(lay['mean_layer_relative_displacement']); kls.append(fm['calibration_kl']); rms.append(fm['calibration_rms_logit'])
        row={'continuation_step':k,'parameter_divergence':div,'normalized_parameter_divergence':div/(theta+EPS),'validation_loss_gap':gap,'perplexity_gap':pg,'lr':lr,'candidate_lr':xlr,'source_task_loss':vlc,'candidate_task_loss':vlx,'clean_cumulative_update_distance':cumc,'candidate_cumulative_update_distance':cumx,'cumulative_update_discrepancy':cumd,'clean_update_norm':un,'candidate_update_norm':xn,'delta_update_norm':dn,'update_cosine':cos,'clean_update_normalized_divergence':div/(cumc+EPS),**lay,**fm}
        steps.append(row)
        if instrument:
            mom=v2.optimizer_moment_stats(co,xo); cstep=v2.optimizer_step_value(co); xstep=v2.optimizer_step_value(xo); b1,b2=base.BETAS
            mech.append({**row,'theta_clean_norm':v2.norm_model(cm),'theta_candidate_norm':v2.norm_model(xm),'train_loss_clean':cl,'train_loss_candidate':xl,'clean_optimizer_step_counter':cstep,'candidate_optimizer_step_counter':xstep,'clean_bias_correction1':1.0-b1**cstep,'clean_bias_correction2':1.0-b2**cstep,'candidate_bias_correction1':1.0-b1**xstep,'candidate_bias_correction2':1.0-b2**xstep,'scheduler_last_epoch_clean':getattr(cs,'last_epoch',''),'scheduler_last_epoch_candidate':getattr(xs,'last_epoch',''),'scheduler_transition_indicator':int(k in {25,50,75,100}),'weight_decay':co.param_groups[0].get('weight_decay',''),**mom})
    tt=v2.first_crossing(divs,tau); rec=v2.recovery_step(gaps,0.05)
    summ={'theta_checkpoint_norm':theta,'tau_numeric':tau,'T_tau':tt,'censored_at_100':tt is None,'max_parameter_divergence':max(divs),'final_parameter_divergence':divs[-1],'max_normalized_parameter_divergence':max(ndivs),'final_normalized_parameter_divergence':ndivs[-1],'validation_loss_base':vclean,'validation_loss_rewrite':vcand,'validation_loss_gap':gaps,'perplexity_gap':pgaps,'lr_by_step':lrs,'candidate_lr_by_step':xlrs,'task_recovery_step':rec,'task_nonrecovered_h100':rec is None,'task_nonrecovered_h20':v2.recovery_step(gaps[:H20],0.05) is None,'max_validation_loss_gap':max(gaps) if gaps else 0.0,'final_validation_loss_gap':gaps[-1] if gaps else 0.0,'max_perplexity_gap':max(pgaps) if pgaps else 0.0,'final_perplexity_gap':pgaps[-1] if pgaps else 0.0,'max_clean_update_normalized_divergence':max(upd_metric),'final_clean_update_normalized_divergence':upd_metric[-1],'max_layer_relative_displacement':max(max_layer),'mean_layer_relative_displacement_max':max(mean_layer),'max_calibration_kl':max(kls),'final_calibration_kl':kls[-1],'max_calibration_rms_logit':max(rms),'final_calibration_rms_logit':rms[-1],'optimizer_reset_definition':'optimizer_state_reset_isolated','checkpoint_optimizer_lr':audit['checkpoint_optimizer_lr'],'isolated_candidate_initial_lr':audit['candidate_optimizer_lr'],'isolated_lr_matches_checkpoint':audit['lr_matches_checkpoint'],'isolated_param_groups_match_checkpoint':audit['param_groups_match_checkpoint'],'isolated_scheduler_state_matches_checkpoint':audit['scheduler_state_matches_checkpoint'],'isolated_initial_optimizer_state_entries':audit['initial_optimizer_state_entries'],'isolated_initial_optimizer_state_keys':audit['initial_optimizer_state_keys']}
    del cm,xm,co,xo,cs,xs; torch.cuda.empty_cache(); return summ,steps,mech

def hist(rows):
    c=Counter(str(r['T_tau']) for r in rows if r['T_tau'] is not None); return {str(i):c.get(str(i),0) for i in range(H+1)}
def group_summary(rows,key):
    out=[]
    for v in sorted({r[key] for r in rows},key=str):
        rs=[r for r in rows if r[key]==v]; tv=[r['T_tau'] for r in rs if r['T_tau'] is not None]
        out.append({key:v,'rows':len(rs),'crossed_h20':sum(r['T_tau'] is not None and int(r['T_tau'])<=H20 for r in rs),'crossed_h100':len(tv),'task_nonrecovered_h20':sum(bool(r['task_nonrecovered_h20']) for r in rs),'task_nonrecovered_h100':sum(bool(r['task_nonrecovered_h100']) for r in rs),'median_T_tau':float(np.median(tv)) if tv else ''})
    return out
def auc(points):
    pts=sorted(points); return sum((x1-x0)*(y0+y1)/2 for (x0,y0),(x1,y1) in zip(pts,pts[1:]))
def roc_pr(rows,field,label='task_nonrecovered_h100'):
    ths=[float('inf')]+sorted({float(r[field]) for r in rows},reverse=True)+[-1e-12]; roc=[]; pr=[]
    for th in ths:
        tp=fp=tn=fn=0
        for r in rows:
            pred=float(r[field])>=th; y=bool(r[label])
            if pred and y: tp+=1
            elif pred and not y: fp+=1
            elif not pred and not y: tn+=1
            else: fn+=1
        tpr=tp/(tp+fn) if tp+fn else 0.0; fpr=fp/(fp+tn) if fp+tn else 0.0; prec=tp/(tp+fp) if tp+fp else 1.0
        roc.append((fpr,tpr)); pr.append((tpr,prec))
    return {'AUROC':auc(roc),'AUPRC':auc(pr)}
def residual_spearman(rows,xfield,yfield,keys):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    xs=[]; ys=[]
    for rs in groups.values():
        mx=float(np.mean([float(r[xfield]) for r in rs])); my=float(np.mean([float(r[yfield]) for r in rs]))
        for r in rs: xs.append(float(r[xfield])-mx); ys.append(float(r[yfield])-my)
    return v2.spearman(xs,ys)
def seed_boot(rows,field,n=400):
    rng=random.Random(20260822); by={s:[r for r in rows if int(r['seed'])==s] for s in PRIMARY_SEEDS}; vals=[]
    for _ in range(n):
        sm=[]
        for s in [rng.choice(PRIMARY_SEEDS) for _ in PRIMARY_SEEDS]: sm+=by[s]
        val=v2.spearman([float(r[field]) for r in sm],[float(r['max_validation_loss_gap']) for r in sm])
        if val is not None: vals.append(val)
    return {'bootstrap_seed':20260822,'samples':len(vals),'mean':float(np.mean(vals)) if vals else None,'p05':float(np.percentile(vals,5)) if vals else None,'p95':float(np.percentile(vals,95)) if vals else None}
def metric_stats(rows):
    out={}
    for name,field in METRICS.items():
        out[name]={'field':field,'pooled_spearman':v2.spearman([float(r[field]) for r in rows],[float(r['max_validation_loss_gap']) for r in rows]),'roc_pr_h100':roc_pr(rows,field,'task_nonrecovered_h100'),'roc_pr_h20':roc_pr(rows,field,'task_nonrecovered_h20'),'per_seed_spearman':{str(s):v2.spearman([float(r[field]) for r in rows if int(r['seed'])==s],[float(r['max_validation_loss_gap']) for r in rows if int(r['seed'])==s]) for s in PRIMARY_SEEDS},'family_checkpoint_residualized_spearman':residual_spearman(rows,field,'max_validation_loss_gap',['rewrite_family','checkpoint_age']),'seed_cluster_bootstrap':seed_boot(rows,field),'leave_one_family_out':{},'within_family':{},'checkpoint_stratified':{}}
        for fam in base.REWRITE_FAMILIES:
            rs=[r for r in rows if r['rewrite_family']!=fam]; ws=[r for r in rows if r['rewrite_family']==fam]
            out[name]['leave_one_family_out'][fam]=v2.spearman([float(r[field]) for r in rs],[float(r['max_validation_loss_gap']) for r in rs])
            out[name]['within_family'][fam]=v2.spearman([float(r[field]) for r in ws],[float(r['max_validation_loss_gap']) for r in ws])
        for age in PRIMARY_AGES:
            rs=[r for r in rows if int(r['checkpoint_age'])==age]; out[name]['checkpoint_stratified'][str(age)]=v2.spearman([float(r[field]) for r in rs],[float(r['max_validation_loss_gap']) for r in rs])
    return out
def primary_aggregates(rows,out):
    old=read_json(V2/'h100_rows.json'); oldr={r['row_id']:r for r in old if r['rewrite_family']=='optimizer_reset'}; newr={r['row_id']:r for r in rows if r['rewrite_family']=='optimizer_reset'}
    sweep=[]
    for horizon in [20,100]:
        for eps in v2.TASK_THRESHOLDS:
            sweep.append({'horizon':horizon,'threshold':eps,'task_nonrecovered':sum(v2.recovery_step(r['validation_loss_gap'][:horizon],eps) is None for r in rows),'rows':len(rows)})
    summ={'status':'COMPLETE','rows':len(rows),'h20_crossings':sum(r['T_tau'] is not None and int(r['T_tau'])<=H20 for r in rows),'h100_crossings':sum(r['T_tau'] is not None for r in rows),'h20_task_nonrecovered':sum(bool(r['task_nonrecovered_h20']) for r in rows),'h100_task_nonrecovered':sum(bool(r['task_nonrecovered_h100']) for r in rows),'delayed_cases':sum(r['T_tau'] is not None and 10<int(r['T_tau'])<=20 for r in rows),'latency_histogram':hist(rows),'per_seed':group_summary(rows,'seed'),'per_family':group_summary(rows,'rewrite_family'),'per_checkpoint':group_summary(rows,'checkpoint_age'),'old_vs_corrected_optimizer_reset_T_tau':{rid:{'old':oldr[rid]['T_tau'],'corrected':newr[rid]['T_tau']} for rid in sorted(newr)},'metrics':metric_stats(rows),'task_threshold_sweep':sweep}
    write_json(out/'primary_h100_corrected_summary.json',summ); write_json(out/'primary_corrected_statistics.json',summ); write_json(out/'primary_corrected_latency_histogram.json',summ['latency_histogram']); write_csv(out/'primary_corrected_per_seed_summary.csv',summ['per_seed']); write_csv(out/'primary_corrected_per_family_summary.csv',summ['per_family']); write_csv(out/'primary_corrected_per_checkpoint_summary.csv',summ['per_checkpoint']); write_md(out/'primary_h100_corrected_summary.md',f"# Corrected Primary H100 Summary\n\n- Rows: {summ['rows']}\n- H20 crossings: {summ['h20_crossings']}\n- H100 crossings: {summ['h100_crossings']}\n- H20 task-nonrecovered: {summ['h20_task_nonrecovered']}\n- H100 task-nonrecovered: {summ['h100_task_nonrecovered']}\n- Delayed cases: {summ['delayed_cases']}")

def run_primary_corrected():
    out=OUT/'primary'; rows=[]; steps=[]
    for seed in PRIMARY_SEEDS:
        batches,_,dataset=base.load_batches(seed,base.TRAIN_STEPS+H+50); val=batches[301:304]; cal=batches[304]
        for age in PRIMARY_AGES:
            rid=f'gpu_seed{seed}_ckpt{age}_optimizer_reset'
            with sched_step(100): payload=base.load_checkpoint(p_primary(seed,age)); t0=time.perf_counter(); summ,st,_=run_pair_isolated(payload,batches,val,cal)
            summ.update({'row_id':rid,'experiment':'reviewer2026_v3_lr_isolation_primary','seed':seed,'checkpoint_age':age,'rewrite_family':'optimizer_reset','dataset':dataset,'checkpoint_file':str(p_primary(seed,age).relative_to(ROOT)),'runtime_seconds':time.perf_counter()-t0,'source':'rerun_lr_isolated_optimizer_reset'}); rows.append(summ)
            steps += [{'row_id':rid,'seed':seed,'checkpoint_age':age,'rewrite_family':'optimizer_reset',**s} for s in st]
            write_csv(out/'primary_optimizer_reset_corrected_rows.csv',rows); write_json(out/'primary_optimizer_reset_corrected_rows.json',rows); write_csv(out/'primary_optimizer_reset_corrected_steps.csv',steps)
    by={r['row_id']:r for r in rows}; corrected=[]
    for r in read_json(V2/'h100_rows.json'):
        if r['rewrite_family']=='optimizer_reset': corrected.append(copy.deepcopy(by[r['row_id']]))
        else:
            nr=copy.deepcopy(r); nr['source']='historical_unaffected_v2'; corrected.append(nr)
    write_csv(out/'primary_h100_corrected_99_rows.csv',corrected); write_json(out/'primary_h100_corrected_99_rows.json',corrected); primary_aggregates(corrected,out)

def delayed_keys():
    old=read_json(V2/'h100_rows.json')
    return sorted((int(r['seed']),int(r['checkpoint_age'])) for r in old if r['rewrite_family']=='optimizer_reset' and r['T_tau'] is not None and 10<int(r['T_tau'])<=20)

def run_mechanism_isolated():
    out=OUT/'mechanism'; rows=[]; cases=[]; old={r['row_id']:r for r in read_json(V2/'delayed_mechanism_cases.json')}
    for seed,age in delayed_keys():
        batches,_,dataset=base.load_batches(seed,base.TRAIN_STEPS+H+50); val=batches[301:304]; cal=batches[304]
        with sched_step(100): payload=base.load_checkpoint(p_primary(seed,age)); summ,_,mech=run_pair_isolated(payload,batches,val,cal,instrument=True)
        rid=f'gpu_seed{seed}_ckpt{age}_optimizer_reset'; cases.append({'row_id':rid,'seed':seed,'checkpoint_age':age,'rewrite_family':'optimizer_reset','source':'rerun_lr_isolated_optimizer_reset','old_T_tau':old[rid]['T_tau'],**summ}); rows += [{'row_id':rid,'seed':seed,'checkpoint_age':age,'rewrite_family':'optimizer_reset',**m} for m in mech]
        write_csv(out/'mechanism_isolated_steps.csv',rows); write_json(out/'mechanism_isolated_cases.json',cases)
    lines=['# Isolated Optimizer-Reset Mechanism','','| row | old T_tau | corrected T_tau | max gap | max delta update |','|---|---:|---:|---:|---:|']
    for c in cases:
        rs=[r for r in rows if r['row_id']==c['row_id']]; lines.append(f"| {c['row_id']} | {c['old_T_tau']} | {c['T_tau']} | {c['max_validation_loss_gap']:.6g} | {max(float(r['delta_update_norm']) for r in rs):.6g} |")
    write_md(out/'mechanism_isolated_summary.md','\n'.join(lines)); write_json(out/'mechanism_isolated_summary.json',{'status':'COMPLETE','cases':len(cases),'step_rows':len(rows),'old_vs_corrected_T_tau':{c['row_id']:{'old':c['old_T_tau'],'corrected':c['T_tau']} for c in cases}})

def run_multi_future_isolated():
    out=OUT/'multi_future'; rows=[]
    for seed,age in delayed_keys():
        with sched_step(100): payload=base.load_checkpoint(p_primary(seed,age))
        for fseed in FUTURE_SCHEDULE_SEEDS:
            batches,_,_=base.load_batches(fseed,H+60); val=batches[1:4]; cal=batches[4]
            with sched_step(100): summ,_,_=run_pair_isolated(payload,batches,val,cal,future_start=5)
            rows.append({'row_id':f'gpu_seed{seed}_ckpt{age}_optimizer_reset','source_seed':seed,'checkpoint_age':age,'future_schedule_seed':fseed,'T_tau':summ['T_tau'],'censored':summ['T_tau'] is None,'task_recovery_step':summ['task_recovery_step'],'task_nonrecovered_h20':summ['task_nonrecovered_h20'],'task_nonrecovered_h100':summ['task_nonrecovered_h100'],'max_validation_loss_gap':summ['max_validation_loss_gap'],'final_validation_loss_gap':summ['final_validation_loss_gap'],'max_function_kl':summ['max_calibration_kl'],'final_function_kl':summ['final_calibration_kl'],'source':'rerun_lr_isolated_optimizer_reset','isolated_lr_matches_checkpoint':summ['isolated_lr_matches_checkpoint'],'isolated_initial_optimizer_state_entries':summ['isolated_initial_optimizer_state_entries']}); write_csv(out/'multi_future_isolated_rows.csv',rows)
    sums=[]
    for key in sorted({(r['source_seed'],r['checkpoint_age']) for r in rows}):
        rs=[r for r in rows if (r['source_seed'],r['checkpoint_age'])==key]; ts=[r['T_tau'] for r in rs]; tv=[t for t in ts if t is not None]
        sums.append({'source_seed':key[0],'checkpoint_age':key[1],'future_schedules':len(rs),'P_T_le_10':sum(t is not None and t<=10 for t in ts)/len(rs),'P_T_le_20':sum(t is not None and t<=20 for t in ts)/len(rs),'P_T_le_50':sum(t is not None and t<=50 for t in ts)/len(rs),'P_T_le_100':sum(t is not None and t<=100 for t in ts)/len(rs),'median_T_tau':float(np.median(tv)) if tv else None,'task_nonrecovered_h20':sum(bool(r['task_nonrecovered_h20']) for r in rs),'task_nonrecovered_h100':sum(bool(r['task_nonrecovered_h100']) for r in rs),'max_validation_loss_gap_mean':float(np.mean([r['max_validation_loss_gap'] for r in rs])),'max_function_kl_mean':float(np.mean([r['max_function_kl'] for r in rs]))})
    write_json(out/'multi_future_isolated_rows.json',rows); write_json(out/'multi_future_isolated_summary.json',{'status':'COMPLETE','rows':len(rows),'case_summaries':sums}); write_md(out/'multi_future_isolated_summary.md','# Isolated Multi-Future Summary\n\n'+'\n'.join(f"- seed {r['source_seed']} age {r['checkpoint_age']}: P(T<=20)={r['P_T_le_20']:.2f}, P(T<=100)={r['P_T_le_100']:.2f}" for r in sums))
def campaign_aggs(rows,out,kind,groups):
    summ={'status':'COMPLETE','kind':kind,'rows':len(rows),'h100_crossings':sum(r['T_tau'] is not None for r in rows),'h20_crossings':sum(r['T_tau'] is not None and int(r['T_tau'])<=H20 for r in rows),'task_nonrecovered_h100':sum(bool(r['task_nonrecovered_h100']) for r in rows),'task_nonrecovered_h20':sum(bool(r['task_nonrecovered_h20']) for r in rows),'latency_histogram':hist(rows),'metric_summary':heavy.metric_summary(rows)}
    for g in groups:
        summ[f'per_{g}']=group_summary(rows,g); write_csv(out/f'corrected_per_{g}_summary.csv',summ[f'per_{g}'])
    write_json(out/'corrected_latency_histogram.json',summ['latency_histogram']); write_json(out/'corrected_metric_summary.json',summ['metric_summary']); write_json(out/'corrected_summary.json',summ)

def run_controlled_corrected():
    out=OUT/'controlled_scale'; resets=[]
    for cfg in v2.SCALE_CONFIGS:
        for seed in CONTROLLED_SEEDS:
            batches,_,dataset=base.load_batches(seed,300+H+50); val=batches[301:304]; cal=batches[304]
            for age in CONTROLLED_AGES:
                with sched_step(100): payload=base.load_checkpoint(p_control(cfg['name'],seed,age)); summ,_,_=run_pair_isolated(payload,batches,val,cal)
                rid=f"controlled_scale_{cfg['name']}_seed{seed}_ckpt{age}_optimizer_reset"; summ.update({'row_id':rid,'experiment':'reviewer2026_v3_lr_isolation_controlled_scale','model_config':cfg['name'],'seed':seed,'checkpoint_age':age,'training_steps':300,'rewrite_family':'optimizer_reset','dataset':dataset,'checkpoint_file':str(p_control(cfg['name'],seed,age).relative_to(ROOT)),'source':'rerun_lr_isolated_optimizer_reset'}); resets.append(summ); write_csv(out/'optimizer_reset_corrected_rows.csv',resets); write_json(out/'optimizer_reset_corrected_rows.json',resets)
    by={(r['model_config'],int(r['seed']),int(r['checkpoint_age']),r['rewrite_family']):r for r in resets}; corrected=[]
    for r in read_json(V2/'controlled_scale'/'rows.json'):
        key=(r['model_config'],int(r['seed']),int(r['checkpoint_age']),r['rewrite_family'])
        if r['rewrite_family']=='optimizer_reset': corrected.append(copy.deepcopy(by[key]))
        else:
            nr=copy.deepcopy(r); nr['source']='historical_unaffected_v2'; corrected.append(nr)
    write_csv(out/'corrected_rows_297.csv',corrected); write_json(out/'corrected_rows_297.json',corrected); campaign_aggs(corrected,out,'controlled_scale',['model_config','seed','rewrite_family'])

def run_long_corrected():
    out=OUT/'long_training'; resets=[]
    for seed in LONG_SEEDS:
        batches,_,dataset=base.load_batches(seed,3000+H+50); val=batches[3001:3004]; cal=batches[3004]
        for age in LONG_AGES:
            with sched_step(1000): payload=base.load_checkpoint(p_long(seed,age)); summ,_,_=run_pair_isolated(payload,batches,val,cal)
            rid=f'long_training_candidate_B_19p48m_seed{seed}_ckpt{age}_optimizer_reset'; summ.update({'row_id':rid,'experiment':'reviewer2026_v3_lr_isolation_long_training','model_config':'candidate_B_19p48m','seed':seed,'checkpoint_age':age,'training_steps':3000,'rewrite_family':'optimizer_reset','dataset':dataset,'checkpoint_file':str(p_long(seed,age).relative_to(ROOT)),'source':'rerun_lr_isolated_optimizer_reset'}); resets.append(summ); write_csv(out/'optimizer_reset_corrected_rows.csv',resets); write_json(out/'optimizer_reset_corrected_rows.json',resets)
    by={(int(r['seed']),int(r['checkpoint_age']),r['rewrite_family']):r for r in resets}; corrected=[]
    for r in read_json(V2/'long_training'/'rows.json'):
        key=(int(r['seed']),int(r['checkpoint_age']),r['rewrite_family'])
        if r['rewrite_family']=='optimizer_reset': corrected.append(copy.deepcopy(by[key]))
        else:
            nr=copy.deepcopy(r); nr['source']='historical_unaffected_v2'; corrected.append(nr)
    write_csv(out/'corrected_rows_99.csv',corrected); write_json(out/'corrected_rows_99.json',corrected); campaign_aggs(corrected,out,'long_training',['checkpoint_age','seed','rewrite_family'])

def bstats(rows,field,th):
    tp=fp=tn=fn=0
    for r in rows:
        pred=float(r[field])>=th; y=bool(r['task_nonrecovered_h100'])
        if pred and y: tp+=1
        elif pred and not y: fp+=1
        elif not pred and not y: tn+=1
        else: fn+=1
    sens=tp/(tp+fn) if tp+fn else 0.0; spec=tn/(tn+fp) if tn+fp else 0.0; prec=tp/(tp+fp) if tp+fp else 0.0; rec=sens; f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'sensitivity':sens,'specificity':spec,'balanced_accuracy':(sens+spec)/2,'precision':prec,'recall':rec,'F1':f1}
def choose_th(rows,field):
    best=None
    for th in [float('inf')]+sorted({float(r[field]) for r in rows},reverse=True)+[-1e-12]:
        st=bstats(rows,field,th); cand=(st['balanced_accuracy'],st['sensitivity']+st['specificity'])
        if best is None or cand>best[0]: best=(cand,th,st)
    return best[1],best[2]
def leave_one_seed_out():
    rows=read_json(OUT/'primary'/'primary_h100_corrected_99_rows.json'); out=[]; nested={'status':'COMPLETE','threshold_selection_rule':'maximum balanced accuracy / Youden J on two training seeds only','metrics':{}}
    for name,field in METRICS.items():
        nested['metrics'][name]={}
        for held in PRIMARY_SEEDS:
            train=[r for r in rows if int(r['seed'])!=held]; test=[r for r in rows if int(r['seed'])==held]; th,tr=choose_th(train,field); te=bstats(test,field,th); desc=roc_pr(test,field,'task_nonrecovered_h100'); rec={'metric':name,'field':field,'heldout_seed':held,'threshold':th,**{f'train_{k}':v for k,v in tr.items()},**te,'heldout_AUROC':desc['AUROC'],'heldout_AUPRC':desc['AUPRC']}; out.append(rec); nested['metrics'][name][str(held)]=rec
    write_csv(OUT/'leave_one_seed_out_discrimination.csv',out); write_json(OUT/'leave_one_seed_out_discrimination.json',nested)
    lines=['# Leave-One-Training-Seed-Out Robustness','','Thresholds are selected on the other two training seeds using maximum balanced accuracy / Youden J, then applied to the held-out training seed. This is not an independent external test set.','','| metric | held-out seed | threshold | balanced accuracy | precision | recall | F1 | AUROC | AUPRC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in out: lines.append(f"| {r['metric']} | {r['heldout_seed']} | {r['threshold']:.6g} | {r['balanced_accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['F1']:.3f} | {r['heldout_AUROC']:.3f} | {r['heldout_AUPRC']:.3f} |")
    write_md(OUT/'leave_one_seed_out_discrimination.md','\n'.join(lines))

def calibration_validation_disjointness():
    camps=[{'campaign':'primary_v2_and_v3','training_steps':300,'validation_indices':[301,302,303],'calibration_index':304,'seeds':PRIMARY_SEEDS},{'campaign':'controlled_scale','training_steps':300,'validation_indices':[301,302,303],'calibration_index':304,'seeds':CONTROLLED_SEEDS},{'campaign':'long_training','training_steps':3000,'validation_indices':[3001,3002,3003],'calibration_index':3004,'seeds':LONG_SEEDS},{'campaign':'multi_future','training_steps':None,'validation_indices':[1,2,3],'calibration_index':4,'seeds':FUTURE_SCHEDULE_SEEDS}]
    for c in camps: c['disjoint_by_index']=c['calibration_index'] not in set(c['validation_indices']); c['status']='PASS' if c['disjoint_by_index'] else 'FAIL'
    st='PASS' if all(c['status']=='PASS' for c in camps) else 'FAIL'; write_json(OUT/'calibration_validation_disjointness.json',{'status':st,'campaigns':camps})
    if st!='PASS': raise RuntimeError('calibration/validation overlap detected')

def provenance():
    write_json(OUT/'PROVENANCE.json',{'starting_commit':SOURCE_STARTING_COMMIT,'branch':run_cmd(['git','branch','--show-current']),'current_commit_at_start':run_cmd(['git','rev-parse','HEAD']),'dirty_status_at_start':run_cmd(['git','status','--short']),'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
def manifest():
    files=[]
    for p in OUT.rglob('*'):
        if p.is_file() and p.name!='MANIFEST.json': files.append({'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size})
    write_json(OUT/'MANIFEST.json',{'status':'COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),'files':files})
def run_all():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    OUT.mkdir(parents=True,exist_ok=True); provenance(); audit_legacy_reset(); run_isolation_tests(); run_primary_corrected(); run_mechanism_isolated(); run_multi_future_isolated(); run_controlled_corrected(); run_long_corrected(); leave_one_seed_out(); calibration_validation_disjointness(); manifest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',default='all'); steps=[s.strip() for s in ap.parse_args().steps.split(',') if s.strip()]; OUT.mkdir(parents=True,exist_ok=True)
    funcs={'provenance':provenance,'audit':audit_legacy_reset,'tests':run_isolation_tests,'primary':run_primary_corrected,'mechanism':run_mechanism_isolated,'multi':run_multi_future_isolated,'controlled':run_controlled_corrected,'long':run_long_corrected,'loso':leave_one_seed_out,'disjoint':calibration_validation_disjointness,'manifest':manifest}
    if 'all' in steps: run_all()
    else:
        for s in steps: funcs[s]()
        manifest()
    print(f'reviewer2026_v3_lr_isolation steps complete: {steps}')
if __name__=='__main__': main()