"""Prospective CNN replication. Outputs are exclusive-create; no historical edits."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse, copy, csv, hashlib, json, math, platform, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_epsilon_sensitivity import rewrite_payload, flat_params, param_l2

FAMILIES = ['clean', 'reset_m', 'reset_v', 'full_per_parameter_reset']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)

def write_csv(name, rows):
    with (OUT / name).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def setup(seed):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(128,10))

def build(seed):
    setup(seed)
    model = CNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999),
        eps=1e-8, weight_decay=0.01, foreach=False, fused=False)
    return model, opt

def data():
    # torchvision checks official file MD5s, and we additionally record SHA256s.
    root = ROOT / 'experiments' / 'data'
    try:
        train = torchvision.datasets.CIFAR10(root, train=True, download=False)
        test = torchvision.datasets.CIFAR10(root, train=False, download=False)
    except RuntimeError:
        root = OUT / 'data'
        train = torchvision.datasets.CIFAR10(root, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root, train=False, download=True)
    # Keep uint8 dataset on CPU; convert only selected batches, identically to ToTensor.
    def tensors(ds):
        return torch.from_numpy(ds.data.copy()).permute(0,3,1,2).contiguous(), torch.tensor(ds.targets)
    return tensors(train), tensors(test), root

def prepare(ds, ids):
    x = ds[0][ids].to(DEVICE).float().div_(255.)
    mean = x.new_tensor([0.4914,0.4822,0.4465])[None,:,None,None]
    std = x.new_tensor([0.2470,0.2435,0.2616])[None,:,None,None]
    return (x-mean)/std, ds[1][ids].to(DEVICE)

def indices(seed, steps=3100):
    g = torch.Generator().manual_seed(seed + 1_000_000)
    batches = []
    while len(batches) < steps:
        order = torch.randperm(50000, generator=g)
        batches.extend(order[i:i+128] for i in range(0, 50000-127, 128))
    return torch.stack(batches[:steps])

def step(model, opt, batch):
    model.train(); opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(batch[0]), batch[1])
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
    loss.backward(); opt.step()
    return float(loss.detach())

@torch.no_grad()
def evaluate(model, ds, n=10000):
    model.eval(); loss = correct = 0
    for i in range(0,n,256):
        x,y = prepare(ds, slice(i,min(i+256,n)))
        logits = model(x)
        loss += float(F.cross_entropy(logits,y,reduction='sum'))
        correct += int((logits.argmax(1)==y).sum())
    loss /= n
    if not math.isfinite(loss): raise FloatingPointError('nonfinite held-out loss')
    return {'loss':loss, 'accuracy':correct/n, 'examples':n}

def payload(model,opt,seed,age):
    return copy.deepcopy({'model_state':model.state_dict(), 'optimizer_state':opt.state_dict(),
        'scheduler_state':None, 'seed':seed, 'checkpoint_age':age, 'batch_pos':age,
        'torch_rng_state':torch.get_rng_state(),
        'cuda_rng_state':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else []})

def clone(p, family='clean', epsilon=None):
    p = copy.deepcopy(p) if family=='clean' else rewrite_payload(p,family,None)
    model,opt = build(p['seed'])
    model.load_state_dict(p['model_state']); opt.load_state_dict(p['optimizer_state'])
    if epsilon is not None:
        for group in opt.param_groups: group['eps'] = epsilon
    torch.set_rng_state(p['torch_rng_state'].cpu())
    if DEVICE.type=='cuda': torch.cuda.set_rng_state_all([r.cpu() for r in p['cuda_rng_state']])
    return model,opt

def first_metrics(u,v):
    a,b,d = float(u.norm()),float(v.norm()),float((v-u).norm())
    return {'S1':d/(a+1e-12), 'clean_first_update_l2':a, 'candidate_first_update_l2':b,
        'first_update_delta_l2':d,
        'first_update_cosine_similarity':float(torch.dot(u,v)/(u.norm()*v.norm()+1e-12)) if a*b>0 else None}

def environment(root):
    return {'utc':datetime.now(timezone.utc).isoformat(), 'git_commit':git('rev-parse','HEAD'),
        'git_branch':git('branch','--show-current'), 'git_status':git('status','--short'),
        'python':sys.version, 'pytorch':torch.__version__, 'torchvision':torchvision.__version__,
        'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(), 'device':str(DEVICE),
        'device_name':torch.cuda.get_device_name() if DEVICE.type=='cuda' else platform.processor(),
        'platform':platform.platform(), 'cpu_threads':4,
        'dataset':'CIFAR-10 python version; official 50000 train / 10000 test',
        'dataset_source':'https://www.cs.toronto.edu/~kriz/cifar.html',
        'dataset_root':str(root),
        'dataset_sha256':{str(p.relative_to(root)):sha(p) for p in sorted((root/'cifar-10-batches-py').iterdir()) if p.is_file()},
        'implementation_sha256':sha(__file__),
        'reused_utility_sha256':sha(ROOT/'scripts/run_epsilon_sensitivity.py')}

def pilot():
    if (OUT/'pilot.json').exists(): raise FileExistsError('pilot already exists')
    start = time.perf_counter()
    train,test,root = data()
    model,opt = build(4099)
    ids = indices(4099,500)
    rows=[]
    baseline=evaluate(model,test,2000)
    t=time.perf_counter()
    for k in range(500):
        loss=step(model,opt,prepare(train,ids[k]))
        rows.append({'step':k+1,'train_loss':loss})
        if (k+1)%100==0: print(f'pilot step {k+1}/500 loss={loss:.4f}',flush=True)
    train_seconds=time.perf_counter()-t
    t=time.perf_counter(); final=evaluate(model,test,2000); eval_seconds=time.perf_counter()-t
    # All 5 seeds: 15000 source + 6000 replay + 300 optional clean equality steps.
    estimate = train_seconds/500*21300 + eval_seconds*5*15*4*5
    report={'seed':4099,'steps':500,'baseline':baseline,'final':final,
        'first_50_mean_loss':float(np.mean([r['train_loss'] for r in rows[:50]])),
        'last_50_mean_loss':float(np.mean([r['train_loss'] for r in rows[-50:]])),
        'training_seconds':train_seconds,'evaluation_2000_seconds':eval_seconds,
        'estimated_primary_5_seed_seconds':estimate,'parameter_count':sum(p.numel() for p in model.parameters()),
        'runtime_seconds':time.perf_counter()-start, 'rewrite_comparisons_run':False,
        'acceptable':final['accuracy']>0.20 and final['loss']<baseline['loss']}
    write_csv('pilot_source_training.csv',rows); write_json('pilot.json',report)
    write_json('environment_pilot.json',environment(root))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['pilot'])
    args=parser.parse_args()
    try: pilot()
    except Exception:
        write_json('FAILURE_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json',
            {'phase':args.phase,'traceback':traceback.format_exc()})
        raise
