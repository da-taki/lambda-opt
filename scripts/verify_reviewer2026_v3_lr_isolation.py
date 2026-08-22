from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'reviewer2026_v3_lr_isolation'

def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def req(errors,rel):
    if not (OUT/rel).exists(): errors.append(f'missing {rel}')
def check_reset_rows(errors, rows, label, expected):
    if len(rows)!=expected: errors.append(f'{label} row count {len(rows)} != {expected}')
    for r in rows:
        if r.get('source')!='rerun_lr_isolated_optimizer_reset': errors.append(f'{label} non-rerun source {r.get("row_id")}')
        if not bool(r.get('isolated_lr_matches_checkpoint')): errors.append(f'{label} LR mismatch {r.get("row_id")}')
        if not bool(r.get('isolated_param_groups_match_checkpoint')): errors.append(f'{label} param group mismatch {r.get("row_id")}')
        if not bool(r.get('isolated_scheduler_state_matches_checkpoint')): errors.append(f'{label} scheduler mismatch {r.get("row_id")}')
        if int(r.get('isolated_initial_optimizer_state_entries',-1)) != 0: errors.append(f'{label} optimizer state not empty {r.get("row_id")}')
        if r.get('isolated_initial_optimizer_state_keys') not in ([], ''): errors.append(f'{label} optimizer state keys survived {r.get("row_id")}')
def main():
    errors=[]
    required=['legacy_reset_audit.json','optimizer_reset_isolation_tests.json','primary/primary_optimizer_reset_corrected_rows.json','primary/primary_optimizer_reset_corrected_steps.csv','primary/primary_h100_corrected_99_rows.json','primary/primary_h100_corrected_summary.json','primary/primary_corrected_statistics.json','mechanism/mechanism_isolated_cases.json','mechanism/mechanism_isolated_steps.csv','mechanism/mechanism_isolated_summary.md','multi_future/multi_future_isolated_rows.json','multi_future/multi_future_isolated_summary.json','multi_future/multi_future_isolated_summary.md','controlled_scale/optimizer_reset_corrected_rows.json','controlled_scale/corrected_rows_297.json','controlled_scale/corrected_summary.json','controlled_scale/corrected_metric_summary.json','long_training/optimizer_reset_corrected_rows.json','long_training/corrected_rows_99.json','long_training/corrected_summary.json','long_training/corrected_metric_summary.json','leave_one_seed_out_discrimination.json','leave_one_seed_out_discrimination.csv','leave_one_seed_out_discrimination.md','calibration_validation_disjointness.json','MANIFEST.json']
    for r in required: req(errors,r)
    if errors: return report(errors)
    audit=read_json(OUT/'legacy_reset_audit.json')
    if audit.get('status')!='COMPLETE' or not audit.get('reviewer_concern_confirmed'): errors.append('legacy LR confound audit incomplete or not confirmed')
    if len(audit.get('rows',[]))!=45: errors.append('legacy audit row count mismatch')
    tests=read_json(OUT/'optimizer_reset_isolation_tests.json')
    if tests.get('status')!='PASS': errors.append('isolation unit tests did not PASS')
    primary_reset=read_json(OUT/'primary/primary_optimizer_reset_corrected_rows.json'); check_reset_rows(errors,primary_reset,'primary reset',9)
    primary=read_json(OUT/'primary/primary_h100_corrected_99_rows.json')
    if len(primary)!=99: errors.append('corrected primary dataset is not 99 rows')
    if sum(r['rewrite_family']=='optimizer_reset' for r in primary)!=9: errors.append('corrected primary reset rows != 9')
    if any(r['rewrite_family']=='optimizer_reset' and r.get('source')!='rerun_lr_isolated_optimizer_reset' for r in primary): errors.append('corrected primary silently reused old optimizer_reset row')
    if len(read_csv(OUT/'primary/primary_optimizer_reset_corrected_steps.csv'))!=9*101: errors.append('primary corrected step count mismatch')
    mech=read_json(OUT/'mechanism/mechanism_isolated_cases.json')
    if len(mech)!=9: errors.append('mechanism cases != 9')
    if len(read_csv(OUT/'mechanism/mechanism_isolated_steps.csv'))!=9*100: errors.append('mechanism step rows != 900')
    mf=read_json(OUT/'multi_future/multi_future_isolated_rows.json')
    if len(mf)!=45: errors.append('multi-future rows != 45')
    ctrl_reset=read_json(OUT/'controlled_scale/optimizer_reset_corrected_rows.json'); check_reset_rows(errors,ctrl_reset,'controlled reset',27)
    ctrl=read_json(OUT/'controlled_scale/corrected_rows_297.json')
    if len(ctrl)!=297: errors.append('controlled corrected dataset != 297')
    if any(r['rewrite_family']=='optimizer_reset' and r.get('source')!='rerun_lr_isolated_optimizer_reset' for r in ctrl): errors.append('controlled silently reused old optimizer_reset row')
    long_reset=read_json(OUT/'long_training/optimizer_reset_corrected_rows.json'); check_reset_rows(errors,long_reset,'long reset',9)
    long=read_json(OUT/'long_training/corrected_rows_99.json')
    if len(long)!=99: errors.append('long corrected dataset != 99')
    if any(r['rewrite_family']=='optimizer_reset' and r.get('source')!='rerun_lr_isolated_optimizer_reset' for r in long): errors.append('long silently reused old optimizer_reset row')
    loso=read_json(OUT/'leave_one_seed_out_discrimination.json')
    if loso.get('status')!='COMPLETE' or len(read_csv(OUT/'leave_one_seed_out_discrimination.csv'))!=15: errors.append('leave-one-seed-out robustness incomplete')
    dis=read_json(OUT/'calibration_validation_disjointness.json')
    if dis.get('status')!='PASS': errors.append('calibration/validation disjointness did not PASS')
    for rel in ['primary/primary_corrected_per_seed_summary.csv','primary/primary_corrected_per_family_summary.csv','primary/primary_corrected_per_checkpoint_summary.csv','controlled_scale/corrected_per_model_config_summary.csv','controlled_scale/corrected_per_seed_summary.csv','controlled_scale/corrected_per_rewrite_family_summary.csv','long_training/corrected_per_checkpoint_age_summary.csv','long_training/corrected_per_seed_summary.csv','long_training/corrected_per_rewrite_family_summary.csv']:
        req(errors,rel)
    manifest=read_json(OUT/'MANIFEST.json')
    for item in manifest.get('files',[]):
        p=ROOT/item['path']
        if not p.exists(): errors.append(f'manifest path missing {item["path"]}')
        elif sha256(p)!=item['sha256']: errors.append(f'manifest hash mismatch {item["path"]}')
    report(errors)
def report(errors):
    status='PASS' if not errors else 'FAIL'
    (OUT/'verification_report_v3_lr_isolation.json').write_text(json.dumps({'status':status,'errors':errors},indent=2,sort_keys=True),encoding='utf-8')
    if errors:
        print('\n'.join(errors)); raise SystemExit(1)
    print('reviewer2026_v3_lr_isolation verification passed')
if __name__=='__main__': main()