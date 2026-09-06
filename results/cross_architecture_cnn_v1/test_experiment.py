"""Focused precollection validation; synthetic fixtures only."""
import unittest, ast, inspect
from study import *

class Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup(999)
        cls.model,cls.opt=build(999)
        cls.batch=(torch.randn(8,3,32,32,device=DEVICE),torch.arange(8,device=DEVICE)%10)
        for _ in range(3): step(cls.model,cls.opt,cls.batch)
        cls.p=payload(cls.model,cls.opt,999,3)

    def test_architecture_and_preprocessing(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),140714)
        self.assertEqual(len(list(self.model.buffers())),0)
        self.assertFalse(any(isinstance(m,(nn.Dropout,nn.modules.batchnorm._BatchNorm)) for m in self.model.modules()))
        from torchvision.transforms import ToTensor, Normalize
        from PIL import Image
        arr=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
        expected=Normalize([.4914,.4822,.4465],[.2470,.2435,.2616])(ToTensor()(Image.fromarray(arr))).to(DEVICE)
        ds=(torch.from_numpy(arr.copy()).permute(2,0,1)[None],torch.tensor([2]))
        actual,_=prepare(ds,[0])
        self.assertTrue(torch.equal(actual[0],expected))

    def test_checkpoint_clone_equality_and_independence(self):
        a,ao=clone(self.p); b,bo=clone(self.p)
        self.assertTrue(equal(a.state_dict(),b.state_dict()))
        self.assertTrue(equal(ao.state_dict(),bo.state_dict()))
        before=copy.deepcopy(self.p)
        step(a,ao,self.batch)
        self.assertTrue(equal(before,self.p))
        self.assertTrue(equal(b.state_dict(),self.p['model_state']))

    def check_rewrite(self,family,key):
        original=copy.deepcopy(self.p); candidate=rewrite_payload(original,family,None)
        self.assertTrue(equal(original,self.p))
        self.assertTrue(equal(candidate['model_state'],original['model_state']))
        self.assertTrue(equal(candidate['optimizer_state']['param_groups'],original['optimizer_state']['param_groups']))
        self.assertTrue(equal(candidate['scheduler_state'],original['scheduler_state']))
        for i,state in original['optimizer_state']['state'].items():
            changed=candidate['optimizer_state']['state'][i]
            self.assertEqual(set(state),set(changed))
            self.assertGreater(int(torch.count_nonzero(state[key])),0)
            for name,value in state.items():
                self.assertTrue(equal(changed[name],torch.zeros_like(value) if name==key else value))
        m,o=clone(self.p,family)
        self.assertTrue(equal(m.state_dict(),original['model_state']))
        self.assertTrue(equal(o.state_dict(),candidate['optimizer_state']))

    def test_reset_m_only(self): self.check_rewrite('reset_m','exp_avg')
    def test_reset_v_only(self): self.check_rewrite('reset_v','exp_avg_sq')

    def test_full_reset_and_step_restart(self):
        transformed=rewrite_payload(self.p,'full_per_parameter_reset',None)
        self.assertEqual(transformed['optimizer_state']['state'],{})
        self.assertTrue(equal(transformed['optimizer_state']['param_groups'],self.p['optimizer_state']['param_groups']))
        m,o=clone(self.p,'full_per_parameter_reset')
        self.assertTrue(equal(m.state_dict(),self.p['model_state']))
        self.assertEqual(len(o.state),0)
        step(m,o,self.batch)
        self.assertTrue(all(float(s['step'])==1 for s in o.state.values()))
        m2,o2=clone(self.p,'reset_v'); step(m2,o2,self.batch)
        self.assertTrue(all(float(s['step'])==4 for s in o2.state.values()))

    def test_nontrivial_scheduler_state_preserved(self):
        m,o=clone(self.p); scheduler=torch.optim.lr_scheduler.StepLR(o,step_size=2,gamma=.7)
        for _ in range(3): step(m,o,self.batch); scheduler.step()
        p=payload(m,o,999,6); p['scheduler_state']=scheduler.state_dict()
        for family in FAMILIES[1:]:
            rewritten=rewrite_payload(p,family,None)
            self.assertTrue(equal(p['scheduler_state'],rewritten['scheduler_state']))
            self.assertTrue(equal(p['optimizer_state']['param_groups'],rewritten['optimizer_state']['param_groups']))
            self.assertTrue(equal(p['model_state'],rewritten['model_state']))

    def test_clean_determinism_and_same_minibatches(self):
        plans=indices(999,5); self.assertTrue(torch.equal(plans,indices(999,5)))
        self.assertFalse(torch.equal(plans,indices(1000,5)))
        branches=[clone(self.p) for _ in range(2)]
        dataset=(torch.arange(50000,dtype=torch.uint8)[:,None,None,None].expand(-1,3,32,32),torch.arange(50000)%10)
        seen=[[],[]]
        for ids in plans[:3]:
            batch=prepare(dataset,ids)
            for j,(m,o) in enumerate(branches):
                seen[j].append(hashlib.sha256(batch[0].cpu().numpy().tobytes()+batch[1].cpu().numpy().tobytes()).hexdigest())
                step(m,o,batch)
            self.assertTrue(equal(branches[0][0].state_dict(),branches[1][0].state_dict()))
            self.assertTrue(equal(branches[0][1].state_dict(),branches[1][1].state_dict()))
        self.assertEqual(seen[0],seen[1])

    def test_S1_matches_legacy_actual_expression(self):
        import run_epsilon_sensitivity as legacy
        tree=ast.parse(inspect.getsource(legacy.run_pair))
        expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='s1' for t in n.targets))
        for u,v in [(torch.tensor([1.,2.,3.]),torch.tensor([-2.,4.,1.])),(torch.zeros(3),torch.ones(3)),(torch.ones(3),torch.ones(3))]:
            metrics=first_metrics(u,v)
            expected=eval(compile(ast.Expression(expr),'<legacy S1>','eval'),{'delta_norm':float((v-u).norm()),'clean_norm':float(u.norm()),'METRIC_EPS':legacy.METRIC_EPS})
            self.assertEqual(metrics['S1'],expected)

    def test_first_step_analytic_adamw_reconstruction(self):
        for family in FAMILIES:
            m,o=clone(self.p,family)
            before=[p.detach().clone() for p in m.parameters()]
            o.zero_grad(set_to_none=True); F.cross_entropy(m(self.batch[0]),self.batch[1]).backward()
            predicted=[]
            group=o.param_groups[0]; b1,b2=group['betas']
            for param in m.parameters():
                state=o.state.get(param,{})
                age=float(state['step'])+1 if state else 1
                mom=state['exp_avg'].clone() if state else torch.zeros_like(param)
                var=state['exp_avg_sq'].clone() if state else torch.zeros_like(param)
                mom=mom*b1+param.grad*(1-b1)
                var=var*b2+param.grad.square()*(1-b2)
                predicted.append(param.detach()*(1-group['lr']*group['weight_decay'])-group['lr']/(1-b1**age)*mom/((var/(1-b2**age)).sqrt()+group['eps']))
            o.step()
            actual=torch.cat([(param.detach()-old).flatten() for param,old in zip(m.parameters(),before)])
            reconstructed=torch.cat([(pred-old).flatten() for pred,old in zip(predicted,before)])
            relative=float((actual-reconstructed).norm()/(actual.norm()+1e-12))
            self.assertLess(relative,2e-5,(family,relative))

if __name__=='__main__':
    read_protocol()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Validation))
    write_json('test_results.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'failures':[str(x) for x in result.failures],'errors':[str(x) for x in result.errors],'utc':datetime.now(timezone.utc).isoformat(),'fixtures':'synthetic data, seed 999; no primary cohort rewrite outcomes'})
    sys.exit(0 if result.wasSuccessful() else 1)

