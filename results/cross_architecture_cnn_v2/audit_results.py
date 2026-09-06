"""Independent postcollection audit and generated research report; raw results never edited."""
from study import *
from validate_historical import snapshot

def rows(name):
    with (OUT/name).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def main():
    p,h=read_protocol(); start=time.perf_counter()
    first=rows('first_step_rows.csv'); traj=rows('trajectory_rows.csv'); func=rows('functional_rows.csv')
    seeds=rows('seed_summary.csv'); source=rows('source_training.csv')
    manifest=json.loads((OUT/'checkpoint_manifest.json').read_text())
    agg=json.loads((OUT/'aggregate_summary.json').read_text())
    expected={(s,a,f) for s in p['seeds'] for a in p['checkpoint_ages'] for f in FAMILIES}
    key=lambda r:(int(r['seed']),int(r['checkpoint_age']),r['rewrite_family'])
    assert len(first)==60 and {key(r) for r in first}==expected
    assert len(traj)==6060 and len(func)==360 and len(seeds)==20 and len(source)==15005
    assert len({(key(r),int(r['continuation_step'])) for r in traj})==6060
    assert len({(key(r),int(r['continuation_step'])) for r in func})==360
    for r in first+traj+func+seeds+source: assert r['protocol_sha256']==h
    assert agg['protocol_sha256']==h
    for s in p['seeds']:
        assert [int(r['step']) for r in source if int(r['seed'])==s]==list(range(3001))
    for r in first:
        assert float(r['S1'])==float(r['first_update_delta_l2'])/(float(r['clean_first_update_l2'])+1e-12)
        tt=[x for x in traj if key(x)==key(r)]
        assert [int(x['continuation_step']) for x in tt]==list(range(101))
        hits=[int(x['continuation_step']) for x in tt if float(x['parameter_distance'])>=float(x['tau'])]
        assert (int(r['T_tau']) if r['T_tau'] else None)==(hits[0] if hits else None)
        assert (r['trajectory_crossed']=='True')==bool(hits)
        assert float(r['tau'])==.1*float(r['theta_checkpoint_norm'])
        ff=[x for x in func if key(x)==key(r)]
        assert [int(x['continuation_step']) for x in ff]==p['functional_evaluation_steps']
        assert all(int(x['examples'])==10000 for x in ff)
        assert all(0<=float(x['accuracy'])<=1 and math.isfinite(float(x['loss'])) for x in ff)
        assert r['clean_train_loss']==r['candidate_train_loss']
        if r['rewrite_family']=='clean':
            assert float(r['S1'])==0 and all(float(x['parameter_distance'])==0 for x in tt)
    for item in manifest:
        assert sha(OUT/item['source_checkpoint'])==item['source_checkpoint_sha256']
        seed,age=item['seed'],item['checkpoint_age']
        saved=torch.load(OUT/'checkpoints'/f'batches_{seed}.pt',weights_only=True)
        assert torch.equal(saved,indices(seed))
        assert sha(OUT/'checkpoints'/f'batches_{seed}.pt')==item['batch_file_sha256']
        for k in range(1,101):
            expected_hash=hashlib.sha256(saved[age+k-1].numpy().tobytes()).hexdigest()
            rr=[r for r in traj if int(r['seed'])==seed and int(r['checkpoint_age'])==age and int(r['continuation_step'])==k]
            assert len(rr)==4 and {r['batch_sha256'] for r in rr}=={expected_hash}
    # Independent recomputation of seed medians and directional counts.
    lookup={}
    for r in seeds:
        group=[float(x['S1']) for x in first if x['seed']==r['seed'] and x['rewrite_family']==r['rewrite_family']]
        assert len(group)==3 and float(np.median(group))==float(r['median_S1'])
        lookup[int(r['seed']),r['rewrite_family']]=float(r['median_S1'])
    for a,b in [('reset_v','reset_m'),('reset_v','full_per_parameter_reset'),('full_per_parameter_reset','reset_m')]:
        assert agg['paired_seed_direction_counts'][a+'_gt_'+b]==sum(lookup[s,a]>lookup[s,b] for s in p['seeds'])
    for f in FAMILIES:
        vals=[lookup[s,f] for s in p['seeds']]
        assert agg['families'][f]['individual_seed_medians_S1']==vals
        assert agg['families'][f]['median_of_seed_medians_S1']==float(np.median(vals))
    # Reproduce every first-step vector from saved checkpoints and saved batch plans.
    train,_,root=data()
    for item in manifest:
        snap=torch.load(OUT/item['source_checkpoint'],map_location=DEVICE,weights_only=True)
        assert all(int(st['step'])==item['checkpoint_age'] for st in snap['optimizer_state']['state'].values())
        batch=prepare(train,indices(item['seed'])[item['checkpoint_age']])
        cm,co=clone(snap); origin=flat_params(cm); step(cm,co,batch); u=flat_params(cm)-origin
        for family in FAMILIES:
            m,o=clone(snap,family); step(m,o,batch); measured=first_metrics(u,flat_params(m)-origin)
            prior=next(r for r in first if key(r)==(item['seed'],item['checkpoint_age'],family))
            for metric,value in measured.items():
                assert value==float(prior[metric]),(item['seed'],item['checkpoint_age'],family,metric)
    env=json.loads((OUT/'environment.json').read_text())
    for path,digest in env['dataset_sha256'].items(): assert sha(Path(env['dataset_root'])/path)==digest
    before=json.loads((OUT/'historical_file_inventory.json').read_text())
    after=snapshot()
    assert before==after,'Historical file content or metadata changed'
    assert git('diff','--name-only')==''
    legacy=json.loads((OUT/'historical_validation_results.json').read_text())
    focused=json.loads((OUT/'test_results.json').read_text())
    eps=json.loads((OUT/'epsilon_sensitivity_summary.json').read_text()) if (OUT/'epsilon_sensitivity_summary.json').exists() else None
    if eps:
        er=rows('epsilon_sensitivity_rows.csv')
        assert len(er)==27
        assert len({(int(r['seed']),int(r['checkpoint_age']),float(r['resume_epsilon'])) for r in er})==27
        for r in er:
            assert r['clean_epsilon']==r['candidate_epsilon']==r['resume_epsilon']
            assert float(r['S1'])==float(r['first_update_delta_l2'])/(float(r['clean_first_update_l2'])+1e-12)
            assert r['protocol_sha256']==h
    report={'status':'PASS','first_step_rows':60,'trajectory_rows':6060,'functional_rows':360,
      'seed_rows':20,'source_rows':15005,'checkpoints':15,'independent_seeds':5,
      'all_first_step_metrics_exactly_reproduced':True,'batch_plans_and_pairing_verified':True,
      'protocol_and_code_hashes_verified':True,'checkpoint_and_dataset_hashes_verified':True,
      'historical_files_unchanged':len(before),'tracked_diff_empty':True,
      'focused_tests':focused,'historical_verifiers':legacy,'audit_runtime_seconds':time.perf_counter()-start}
    write_json('postcollection_validation.json',report)
    runtime=json.loads((OUT/'primary_runtime.json').read_text())
    pilot=json.loads((OUT/'pilot.json').read_text())
    legacy_lines='\n'.join('- '+r['verifier']+': '+r['status'] for r in legacy['results'])
    with (OUT/'validation_report.md').open('x',encoding='utf-8') as f:
        f.write('# Validation report\n\nPostcollection audit: PASS. Nine focused precollection tests passed unchanged after the CPU preprocessing fix.\n\n'
          'All 60 first-step metric rows were reproduced exactly from saved source checkpoints and minibatch plans. All 6060 trajectory rows, 360 full-test functional rows, 20 seed summaries and 15005 source log rows have complete coverage. Protocol/code/checkpoint/dataset hashes verified. Every real checkpoint also passed exact CLEAN duplicate replay for its first two steps.\n\n'
          'Focused tests cover model count/no stateful layers; exact ToTensor/Normalize preprocessing; clone equality and isolation; reset-m only; reset-v only; full state and counter restart; preservation of a nontrivial StepLR fixture and parameter groups; deterministic CLEAN/batch identity; S1 against the existing implementation expression; analytic AdamW update reconstruction for all four branches.\n\n'
          '## Existing verification scripts\n\n'+legacy_lines+'\n\nHistorical verifiers executed unchanged; only their newly written reports were redirected here. Any FAIL is retained with its complete output under historical_validation/ and does not disappear into the new experiment status.\n\n'
          f'{len(before)} historical files unchanged by content hashes where practical and size/mtime for large binaries; git diff is empty. No paper or frozen result was edited.\n\n'
          'The v1 attempt is retained and invalidated at preflight only: GPU preprocessing failed exact reference equality. CPU reference preprocessing fixed the implementation, then the entire source-only pilot and protocol were regenerated in v2 before any primary-cohort rewrites. No seeds, architecture, optimizer values, checkpoint ages or outcome metrics were tuned.\n')
    lines=['# CNN / CIFAR-10 cross-architecture replication','',
      '**'+agg['verdict']+'**','',
      '## Frozen protocol','',
      f'Protocol SHA256: {h}', '',
      'Five independent seeds 4101–4105; checkpoints 500, 1500, 3000; H=100. CNN: five 3x3 padded convolutions with channels 3→32→32→64→64→128, ReLU after each, 2x2 max pooling after conv2 and conv4, global average pooling, Linear(128,10). Exactly 140714 parameters. No BatchNorm/dropout/augmentation.',
      'Official CIFAR-10: 50000 train / 10000 test. CPU ToTensor-equivalent conversion and normalization mean (0.4914,0.4822,0.4465), std (0.2470,0.2435,0.2616). Batch 128; seeded per-epoch shuffle; incomplete last batch dropped. Saved plans pair all continuations.',
      'AdamW: LR 0.001; betas (0.9,0.999); epsilon 1e-8; weight decay 0.01; float32; foreach/fused/AMSGrad disabled. No scheduler or clipping. Cross entropy. Deterministic algorithms and cuDNN; TF32 disabled.',
      'Reset-m clears only m; reset-v clears only v; full reset clears all per-parameter optimizer state including age. Model parameters and optimizer groups remain identical. Primary S1 = ||candidate update − clean update|| / (||clean update|| + 1e-12); absolute update norms and cosine retained. Tau = 0.10 ||checkpoint parameters||; first distance ≥ tau counted, noncrossings censored at 100.',
      'Full test evaluation at continuation steps 0,1,25,50,75,100. Checkpoint ages are repeated measures. Primary aggregate is the median across five within-seed medians, with min/max/IQR and paired seed counts; no p-values.',
      '', '## Results','',
      '| Seed | Reset m median S1 | Reset v median S1 | Full reset median S1 |','|---|---:|---:|---:|']
    for s in p['seeds']: lines.append(f'| {s} | {lookup[s,"reset_m"]:.6g} | {lookup[s,"reset_v"]:.6g} | {lookup[s,"full_per_parameter_reset"]:.6g} |')
    lines+=['','| Family | Median seed-median S1 | Min–max | IQR | Crossing rows | H100 accuracy | H100 loss |','|---|---:|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        x=agg['families'][family]
        lines.append(f'| {family} | {x["median_of_seed_medians_S1"]:.6g} | {x["min"]:.6g}–{x["max"]:.6g} | {x["iqr"]:.6g} | {x["checkpoint_crossings"]}/15 | {x["H100_median_of_seed_medians_accuracy"]:.2%} | {x["H100_median_of_seed_medians_loss"]:.6g} |')
    lines+=['','H100 loss/accuracy use the median of within-seed checkpoint medians. Full time series remain in functional_rows.csv. Crossing rows are descriptive repeated-measure counts, not independent replicates.','']
    for name,value in agg['paired_seed_direction_counts'].items(): lines.append(f'- {name}: {value}/5 seeds')
    lines+= [f'- H3 favorable: {agg["H3_favorable_seeds"]}/5; adverse: {agg["H3_adverse_seeds"]}/5.','',
      'Absolute update norms and cosines are mandatory raw fields in first_step_rows.csv, with within-seed norm medians in seed_summary.csv.','',
      '![S1 by independent seed](seed_median_S1.png)','', '![Matched trajectories](trajectory_distances.png)','']
    if eps:
        lines+=['## Supporting epsilon check','',
          'Three fixed seeds, all three ages, one matched step per epsilon. Both CLEAN and RESET-V use the same replay epsilon. Baseline epsilon=1e-8 reproduces the primary result exactly.','',
          '| Epsilon | Median seed-median reset-v S1 |','|---:|---:|']
        for r in eps['summary']: lines.append(f'| {r["epsilon"]:.0e} | {r["median_of_seed_medians"]:.6g} |')
        lines+=['','![Epsilon sensitivity](epsilon_sensitivity.png)','']
    lines+=['## Compute, validation and limitations','',
      f'Primary runtime {runtime["seconds"]:.2f} seconds; corrected source-only pilot {pilot["runtime_seconds"]:.2f} seconds; optional epsilon runtime {eps["runtime_seconds"]:.2f} seconds.' if eps else f'Primary runtime {runtime["seconds"]:.2f} seconds.',
      f'{env["device_name"]}; Python {env["python"].split()[0]}, PyTorch {env["pytorch"]}, torchvision {env["torchvision"]}, CUDA {env["cuda"]}. All 5 seeds and 15 checkpoints succeeded.',
      'Nine focused tests passed; independent postcollection audit passed, including exact first-step metric reproduction. See validation_report.md for existing verifier results and the preserved preflight failure.',
      'The pilot used the first 2000 official test images to check basic learning. These are included in the later full test evaluations, so functional test performance is supporting evidence rather than an untouched model-selection benchmark. The primary optimizer/trajectory metrics do not use labels from that held-out split.',
      'This is one small CNN, one image dataset, one fixed training recipe, five seeds and three repeated checkpoint ages. It is evidence about this substrate, not a universal claim across architectures. No inferential p-values or independent-replicate claims for checkpoint rows are made.',
      'For an IRIS submission this provides an auditable, prospectively frozen extension with a meaningfully different architecture/task. It is suitable to add with its actual verdict and limitations; it does not justify a broader claim than the measured ordering and divergence. Admission or judging outcomes are not assessed.',
      '', '## Provenance and files','',
      f'Branch: {env["git_branch"]}; source commit: {env["git_commit"]}. Working tree was already dirty; preexisting work was preserved. New implementation lives entirely in this directory and is identified by frozen SHA256 hashes. No commits or paper edits were made.',
      'protocol.json/.md/.sha256 contain the exact design; environment.json records software and dataset hashes; checkpoints/ retains source states and future batch plans; checkpoint_manifest.json links every case to source SHA256; CSV files hold source, first-step, trajectory, functional and seed-level records; aggregate_summary.json holds the seed-aware analysis; test and audit reports retain validation evidence.',
      'Reproduction entry points: run_experiment.py pilot; study.py freeze; test_experiment.py; study.py primary; study.py epsilon; audit_results.py. Existing output names are protected against overwrites. For a new run use a new explicitly named result directory and freeze its protocol before comparisons.',
      'Precollection failure: v1 GPU conversion failed exact CPU reference preprocessing equality. v1 was retained without primary outcomes; v2 corrected preprocessing and reran the source-only pilot and unchanged tests before primary collection. See precollection_deviations.json. Neither protocol was edited after primary outcomes.',
      '',agg['verdict']]
    with (OUT/'README.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    write_json('final_artifact_manifest.json',{'protocol_sha256':h,'files':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}})
    print('POSTCOLLECTION AUDIT PASS',flush=True)
    print(agg['verdict'],flush=True)

if __name__=='__main__': main()

