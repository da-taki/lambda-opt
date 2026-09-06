"""Freeze, execute and summarize the prospectively specified CNN study."""
from run_experiment import *

def freeze():
    pilot=json.loads((OUT/'pilot.json').read_text())
    assert pilot['acceptable'] and not pilot['rewrite_comparisons_run']
    p={'experiment':'cross_architecture_cnn_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
       'architecture':str(CNN()),'parameter_count':140714,'dtype':'float32',
       'dataset':'official CIFAR-10 python version; 50000 train / 10000 test',
       'preprocessing':{'scale':'uint8 / 255 (ToTensor equivalent)','mean':[.4914,.4822,.4465],'std':[.2470,.2435,.2616],'augmentation':None},
       'batch_construction':{'batch_size':128,'order':'CPU torch.randperm per epoch','generator_seed':'seed + 1000000','drop_last':True,'workers':0,'pairing':'same precomputed indices and same prepared batch object for all branches'},
       'optimizer':{'name':'AdamW','lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'amsgrad':False,'maximize':False,'foreach':False,'fused':False,'gradient_clipping':None},
       'scheduler':None,'loss':'CrossEntropyLoss mean','source_max_steps':3000,'max_batch_step':3100,
       'checkpoint_ages':[500,1500,3000],'seeds':[4101,4102,4103,4104,4105],'continuation_horizon':100,
       'functional_evaluation_steps':[0,1,25,50,75,100],
       'functional_evaluation':'entire official 10000-example test split; batch size 256; loss and accuracy',
       'source_evaluation_steps':[0,500,1500,3000],
       'rewrites':{'clean':'identity','reset_m':'clear exp_avg only; preserve all else','reset_v':'clear exp_avg_sq only; preserve all else','full_per_parameter_reset':'empty per-parameter state restarting m/v/step; preserve parameter groups and scheduler'},
       'metrics':{'S1':'norm(candidate_update-clean_update)/(norm(clean_update)+1e-12)',
         'absolute':['clean update L2','candidate update L2','update difference L2','cosine similarity'],
         'cosine':'dot(u,v)/(norm(u)*norm(v)+1e-12); null for zero update',
         'trajectory':'L2(candidate-clean parameters) at steps 0..100',
         'tau':'0.10 * norm(checkpoint parameters)','crossing':'first distance >= tau; null if censored at H=100'},
       'hypotheses':{'H1':'reset-v produces larger seed-level median S1 than reset-m','H2':'reset-v produces larger seed-level median S1 than full reset','H3':'reset-v causes more rapid or more frequent trajectory divergence than reset-m','H4':'reset-v > full reset > reset-m ordering survives architecture/task change'},
       'analysis':{'unit':'independent seed; checkpoint age is a repeated measure',
         'S1':'within-seed median over 3 ages, then median/min/max/IQR across 5 seeds; strict paired direction counts',
         'H3':'per seed compare crossing count and median first crossing with noncrossings coded H+1=101; favorable if more crossings OR smaller median; adverse if fewer OR larger median',
         'verdict':'SUPPORTED if H1/H2/full>m hold in all 5 seeds and H3 favorable in >=3 with no adverse seeds; NOT SUPPORTED if H1 or H2 holds in <=2 seeds; otherwise MIXED RESULT','p_values':False},
       'determinism':{'deterministic_algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True,'tf32':False,'cublas_workspace_config':':4096:8','cpu_threads':4,'dropout':False,'BatchNorm':False},
       'pilot':{'seed':4099,'steps':500,'sha256':sha(OUT/'pilot.json'),'decision':'original architecture and hyperparameters learn; no changes; keep 5 seeds and requested ages','estimated_primary_seconds':pilot['estimated_primary_5_seed_seconds']},
       'optional_epsilon':{'condition':'primary completes and estimated extra runtime <120 seconds','seeds':[4101,4102,4103],'ages':[500,1500,3000],'epsilons':[1e-8,1e-6,1e-4],'horizon':1,'families':['clean','reset_v'],'same_epsilon_both_branches':True,'analysis':'seed medians across ages; epsilon replay-only; supporting evidence'},
       'failure_policy':'record every failure; no selective rows; post-collection bug invalidates affected cohort; rerun whole affected cohort in new directory',
       'provenance':{'git_commit':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'working_tree':'dirty; preexisting untracked work preserved',
         'code_sha256':{name:sha(OUT/name) for name in ['run_experiment.py','study.py','test_experiment.py']},
         'legacy_utility_sha256':sha(ROOT/'scripts/run_epsilon_sensitivity.py')}}
    h=hashlib.sha256(canonical(p)).hexdigest()
    write_json('protocol.json',p)
    with (OUT/'protocol.sha256').open('x') as f: f.write(h+'\n')
    with (OUT/'protocol.md').open('x',encoding='utf-8') as f:
        f.write('# Frozen prospective CNN replication protocol\n\nCanonical JSON SHA256: '+h+'\n\nFrozen after source-only pilot and before rewrite comparisons. Hash uses UTF-8 JSON with sorted keys and compact separators; all fields included.\n\n~~~json\n'+json.dumps(p,indent=2)+'\n~~~\n')
    print('FROZEN '+h,flush=True)

def read_protocol():
    p=json.loads((OUT/'protocol.json').read_text())
    h=hashlib.sha256(canonical(p)).hexdigest()
    assert h==(OUT/'protocol.sha256').read_text().strip()
    for name,value in p['provenance']['code_sha256'].items(): assert sha(OUT/name)==value, name
    assert sha(ROOT/'scripts/run_epsilon_sensitivity.py')==p['provenance']['legacy_utility_sha256']
    return p,h

def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def primary():
    p,h=read_protocol()
    assert json.loads((OUT/'test_results.json').read_text())['status']=='PASS'
    write_json('PRIMARY_STARTED.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':h})
    start=time.perf_counter()
    train,test,root=data(); write_json('environment.json',environment(root))
    checkpoints=OUT/'checkpoints'; checkpoints.mkdir(exist_ok=False)
    manifest=[]; source=[]; first=[]; trajectory=[]; functional=[]; validation=[]
    for seed in p['seeds']:
        ids=indices(seed); model,opt=build(seed)
        torch.save(ids,checkpoints/f'batches_{seed}.pt')
        batch_hash=sha(checkpoints/f'batches_{seed}.pt')
        ev=evaluate(model,test)
        source.append({'seed':seed,'step':0,'train_loss':None,'test_loss':ev['loss'],'test_accuracy':ev['accuracy'],'protocol_sha256':h})
        for k in range(1,p['source_max_steps']+1):
            loss=step(model,opt,prepare(train,ids[k-1]))
            ev=evaluate(model,test) if k in p['source_evaluation_steps'] else None
            source.append({'seed':seed,'step':k,'train_loss':loss,'test_loss':ev['loss'] if ev else None,'test_accuracy':ev['accuracy'] if ev else None,'protocol_sha256':h})
            if k in p['checkpoint_ages']:
                path=checkpoints/f'seed_{seed}_age_{k}.pt'
                snap=payload(model,opt,seed,k); snap['protocol_sha256']=h
                torch.save(snap,path)
                manifest.append({'seed':seed,'checkpoint_age':k,'source_checkpoint':str(path.relative_to(OUT)),'source_checkpoint_sha256':sha(path),'protocol_sha256':h,'batch_file_sha256':batch_hash})
                print(f'source seed={seed} age={k} test_accuracy={ev["accuracy"]:.4f}',flush=True)
        del model,opt
    write_csv('source_training.csv',source); write_json('checkpoint_manifest.json',manifest)
    for item in manifest:
        seed,age=item['seed'],item['checkpoint_age']
        snap=torch.load(OUT/item['source_checkpoint'],map_location=DEVICE,weights_only=True)
        ids=torch.load(checkpoints/f'batches_{seed}.pt',weights_only=True)
        branches={f:clone(snap,f) for f in FAMILIES}
        origin=flat_params(branches['clean'][0]); theta=float(origin.norm()); tau=.1*theta
        for family,(m,o) in branches.items():
            assert equal(m.state_dict(),snap['model_state'])
            assert equal(o.state_dict()['param_groups'],snap['optimizer_state']['param_groups'])
        duplicate=clone(snap)
        crossings={f:None for f in FAMILIES}
        meta={f:{**item,'rewrite_family':f,'row_id':f'{seed}_{age}_{f}'} for f in FAMILIES}
        baseline=evaluate(branches['clean'][0],test)
        for f in FAMILIES:
            trajectory.append({**meta[f],'continuation_step':0,'parameter_distance':0.,'normalized_parameter_distance':0.,'theta_checkpoint_norm':theta,'tau':tau,'batch_sha256':''})
            functional.append({**meta[f],'continuation_step':0,**baseline})
        local_first={}
        for k in range(1,p['continuation_horizon']+1):
            batch_ids=ids[age+k-1]; batch=prepare(train,batch_ids)
            bh=hashlib.sha256(batch_ids.numpy().tobytes()).hexdigest()
            losses={f:step(m,o,batch) for f,(m,o) in branches.items()}
            if k<=2:
                step(*duplicate,batch)
                assert equal(duplicate[0].state_dict(),branches['clean'][0].state_dict())
                assert equal(duplicate[1].state_dict(),branches['clean'][1].state_dict())
            clean=flat_params(branches['clean'][0])
            for f,(m,o) in branches.items():
                candidate=flat_params(m); distance=param_l2(branches['clean'][0],m)
                assert torch.isfinite(candidate).all()
                if k==1:
                    metrics=first_metrics(clean-origin,candidate-origin)
                    assert math.isclose(metrics['first_update_delta_l2'],distance,rel_tol=2e-5,abs_tol=1e-7)
                    local_first[f]={**meta[f],**metrics,'theta_checkpoint_norm':theta,'tau':tau,'batch_sha256':bh,'initial_parameters_identical':True,'clean_train_loss':losses['clean'],'candidate_train_loss':losses[f]}
                    assert losses[f]==losses['clean']
                if crossings[f] is None and distance>=tau: crossings[f]=k
                trajectory.append({**meta[f],'continuation_step':k,'parameter_distance':distance,'normalized_parameter_distance':distance/(theta+1e-12),'theta_checkpoint_norm':theta,'tau':tau,'batch_sha256':bh})
                if k in p['functional_evaluation_steps']: functional.append({**meta[f],'continuation_step':k,**evaluate(m,test)})
        for f in FAMILIES: first.append({**local_first[f],'trajectory_crossed':crossings[f] is not None,'T_tau':crossings[f]})
        validation.append({**item,'clone_equality':True,'first_two_clean_steps_exact':True,'first_step_loss_equality':True,'first_step_update_distance_consistency':True,'same_batch_object_all_branches':True})
        print(f'replay seed={seed} age={age} complete ({len(validation)}/15)',flush=True)
        del branches,duplicate,snap
    write_csv('first_step_rows.csv',first); write_csv('trajectory_rows.csv',trajectory); write_csv('functional_rows.csv',functional)
    write_json('checkpoint_validation.json',validation)
    write_json('primary_runtime.json',{'seconds':time.perf_counter()-start,'optimizer_steps':21030,'status':'COMPLETE','deviations':[]})
    analyze(first,trajectory,functional,p,h)

def analyze(first,traj,func,p,h):
    summaries=[]
    for seed in p['seeds']:
        for f in FAMILIES:
            rows=[r for r in first if r['seed']==seed and r['rewrite_family']==f]
            summaries.append({'seed':seed,'rewrite_family':f,'checkpoint_rows':len(rows),
                'median_S1':float(np.median([r['S1'] for r in rows])),
                'median_clean_update_l2':float(np.median([r['clean_first_update_l2'] for r in rows])),
                'median_candidate_update_l2':float(np.median([r['candidate_first_update_l2'] for r in rows])),
                'median_delta_update_l2':float(np.median([r['first_update_delta_l2'] for r in rows])),
                'crossings':sum(r['trajectory_crossed'] for r in rows),
                'median_capped_T_tau':float(np.median([r['T_tau'] if r['T_tau'] is not None else 101 for r in rows])),'protocol_sha256':h})
    write_csv('seed_summary.csv',summaries)
    lookup={(r['seed'],r['rewrite_family']):r for r in summaries}
    counts={}
    for a,b in [('reset_v','reset_m'),('reset_v','full_per_parameter_reset'),('full_per_parameter_reset','reset_m')]:
        counts[a+'_gt_'+b]=sum(lookup[s,a]['median_S1']>lookup[s,b]['median_S1'] for s in p['seeds'])
    favorable=adverse=0
    for s in p['seeds']:
        a,b=lookup[s,'reset_v'],lookup[s,'reset_m']
        favorable+=a['crossings']>b['crossings'] or a['median_capped_T_tau']<b['median_capped_T_tau']
        adverse+=a['crossings']<b['crossings'] or a['median_capped_T_tau']>b['median_capped_T_tau']
    if all(x==5 for x in counts.values()) and favorable>=3 and adverse==0: verdict='REPLICATION SUPPORTED'
    elif counts['reset_v_gt_reset_m']<=2 or counts['reset_v_gt_full_per_parameter_reset']<=2: verdict='REPLICATION NOT SUPPORTED'
    else: verdict='MIXED RESULT'
    agg={'protocol_sha256':h,'independent_seeds':5,'paired_checkpoints':15,'paired_seed_direction_counts':counts,'H3_favorable_seeds':favorable,'H3_adverse_seeds':adverse,'verdict':verdict,'families':{}}
    for f in FAMILIES:
        vals=[r['median_S1'] for r in summaries if r['rewrite_family']==f]
        rows=[r for r in first if r['rewrite_family']==f]
        terminal=[r for r in func if r['rewrite_family']==f and r['continuation_step']==100]
        acc=[float(np.median([r['accuracy'] for r in terminal if r['seed']==s])) for s in p['seeds']]
        loss=[float(np.median([r['loss'] for r in terminal if r['seed']==s])) for s in p['seeds']]
        agg['families'][f]={'median_of_seed_medians_S1':float(np.median(vals)),'individual_seed_medians_S1':vals,'min':min(vals),'max':max(vals),'iqr':float(np.percentile(vals,75)-np.percentile(vals,25)),'checkpoint_crossings':sum(r['trajectory_crossed'] for r in rows),'checkpoint_rows':len(rows),'T_tau_crossed':[r['T_tau'] for r in rows if r['trajectory_crossed']],'H100_seed_median_accuracy':acc,'H100_seed_median_loss':loss,'H100_median_of_seed_medians_accuracy':float(np.median(acc)),'H100_median_of_seed_medians_loss':float(np.median(loss))}
    write_json('aggregate_summary.json',agg)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for s in p['seeds']: ax.plot(range(3),[lookup[s,f]['median_S1'] for f in FAMILIES[1:]],'o-',label=str(s))
    ax.set_xticks(range(3),['Reset m','Reset v','Full reset']); ax.set_yscale('log')
    ax.set_ylabel('Within-seed median S1 (3 checkpoint ages)'); ax.legend(title='Independent seed')
    fig.tight_layout(); fig.savefig(OUT/'seed_median_S1.png',dpi=180); fig.savefig(OUT/'seed_median_S1.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    colors={'reset_m':'#26828e','reset_v':'#c64040','full_per_parameter_reset':'#7562ab'}
    for ax,age in zip(axes,p['checkpoint_ages']):
        for f in FAMILIES[1:]:
            for j,s in enumerate(p['seeds']):
                rr=[r for r in traj if r['seed']==s and r['checkpoint_age']==age and r['rewrite_family']==f]
                ax.plot([r['continuation_step'] for r in rr],[r['normalized_parameter_distance'] for r in rr],color=colors[f],alpha=.55,label=f if j==0 else None)
        ax.axhline(.1,color='black',ls='--',lw=1,label='threshold 0.10'); ax.set_title(f'Checkpoint {age}')
        ax.set_xlabel('Matched continuation step'); ax.set_yscale('symlog',linthresh=.001)
    axes[0].set_ylabel('Parameter distance / checkpoint norm'); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'trajectory_distances.png',dpi=180); fig.savefig(OUT/'trajectory_distances.pdf'); plt.close(fig)
    print(json.dumps(agg,indent=2),flush=True)

def epsilon_check():
    p,h=read_protocol(); assert (OUT/'aggregate_summary.json').exists()
    spec=p['optional_epsilon']; pilot=json.loads((OUT/'pilot.json').read_text())
    estimate=54*pilot['training_seconds']/500
    if estimate>=120:
        write_json('epsilon_skipped.json',{'reason':'estimated extra runtime >=120 seconds','estimate':estimate}); return
    start=time.perf_counter(); train,_,_=data(); rows=[]
    write_json('EPSILON_STARTED.json',{'protocol_sha256':h})
    manifest=json.loads((OUT/'checkpoint_manifest.json').read_text())
    with (OUT/'first_step_rows.csv').open() as f: primary_rows=list(csv.DictReader(f))
    for item in manifest:
        if item['seed'] not in spec['seeds']: continue
        seed,age=item['seed'],item['checkpoint_age']; snap=torch.load(OUT/item['source_checkpoint'],map_location=DEVICE,weights_only=True)
        batch=prepare(train,indices(seed)[age])
        for eps in spec['epsilons']:
            cm,co=clone(snap,epsilon=eps); vm,vo=clone(snap,'reset_v',epsilon=eps)
            origin=flat_params(cm); step(cm,co,batch); step(vm,vo,batch)
            metrics=first_metrics(flat_params(cm)-origin,flat_params(vm)-origin)
            if eps==1e-8:
                prior=next(r for r in primary_rows if int(r['seed'])==seed and int(r['checkpoint_age'])==age and r['rewrite_family']=='reset_v')
                assert metrics['S1']==float(prior['S1'])
            rows.append({**item,'rewrite_family':'reset_v','resume_epsilon':eps,'clean_epsilon':eps,'candidate_epsilon':eps,**metrics})
    write_csv('epsilon_sensitivity_rows.csv',rows)
    summary=[]
    for eps in spec['epsilons']:
        vals=[float(np.median([r['S1'] for r in rows if r['seed']==s and r['resume_epsilon']==eps])) for s in spec['seeds']]
        summary.append({'epsilon':eps,'seed_medians':vals,'median_of_seed_medians':float(np.median(vals))})
    write_json('epsilon_sensitivity_summary.json',{'protocol_sha256':h,'seeds':spec['seeds'],'summary':summary,'runtime_seconds':time.perf_counter()-start,'baseline_exact_regression':True,'supporting_only':True})
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for i,s in enumerate(spec['seeds']): ax.loglog(spec['epsilons'],[r['seed_medians'][i] for r in summary],'o-',label=str(s))
    ax.set_xlabel('Matched CLEAN / RESET-V replay epsilon'); ax.set_ylabel('Within-seed median reset-v S1'); ax.legend(title='Seed')
    fig.tight_layout(); fig.savefig(OUT/'epsilon_sensitivity.png',dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['freeze','primary','epsilon']); args=parser.parse_args()
    try: {'freeze':freeze,'primary':primary,'epsilon':epsilon_check}[args.phase]()
    except Exception:
        write_json('FAILURE_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json',{'phase':args.phase,'traceback':traceback.format_exc()})
        raise

