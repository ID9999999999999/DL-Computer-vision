"""Offline, non-performance checks. Uses original manifest and shipped existing baseline checkpoint."""
import hashlib,json,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from compare_poly1 import ROOT,MANIFEST,metrics,poly1_cross_entropy
class PipelineChecks(unittest.TestCase):
    def test_manifest(self):
        a=json.loads(MANIFEST.read_text());self.assertEqual(len(a),99)
        self.assertEqual(len(set(x['path'] for x in a)),99)
        self.assertTrue({x['group'] for x in a if x['split']=='train'}.isdisjoint({x['group'] for x in a if x['split']=='validation'}))
        self.assertEqual(sum(x['split']=='train' for x in a),70)
        self.assertEqual(sum(x['split']=='validation' for x in a),29)
    def test_poly1_epsilon_zero_equals_ce(self):
        x=torch.tensor([[3.,-1.],[-2.,4.]],requires_grad=True);y=torch.tensor([0,1])
        self.assertTrue(torch.allclose(poly1_cross_entropy(x,y,0),F.cross_entropy(x,y)))
    def test_poly1_gradient(self):
        x=torch.tensor([[.5,-.25],[1.,2.]],requires_grad=True);y=torch.tensor([0,1])
        l=poly1_cross_entropy(x,y,1);l.backward();self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.abs().max().item(),0.)
    def test_confusion(self):
        m=metrics([0,0,1,1],[0,1,1,1]);self.assertEqual(m['confusion_matrix'],[[1,1],[0,2]])
        self.assertAlmostEqual(m['recall_defective'],1.0)
    def test_evaluate_loss_is_finite(self):
        from compare_poly1 import evaluate
        class ConstantHead(torch.nn.Module):
            def forward(self,x):
                return torch.stack([x[:,0,0,0],-x[:,0,0,0]],dim=1)
        rows=[{'object':'demo_normal','group':'n','label':0},{'object':'demo_defective','group':'d','label':1}]
        images=[torch.ones(3,2,2),-torch.ones(3,2,2)]
        for mode in ('ce','poly1'):
            stats,records=evaluate(ConstantHead(),images,rows,[0,1],2,mode,1.0,'cpu')
            self.assertEqual(stats['confusion_matrix'],[[1,0],[0,1]])
            self.assertTrue(float('-inf')<stats['validation_loss']<float('inf'))
            self.assertEqual(len(records),2)
    @unittest.skipUnless((ROOT/'resnet18_supervised/best_model.pt').exists(), 'Historical 44.8MB checkpoint not included in Git; optional test')
    def test_existing_checkpoint(self):
        p=ROOT/'resnet18_supervised/best_model.pt'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'1d8c22121278890a09393b65ed8d69a1bc0b621f2851d37821e98f7f65253185')
        model=resnet18(weights=None);model.fc=torch.nn.Linear(512,2)
        state=torch.load(p,map_location='cpu',weights_only=True);model.load_state_dict(state)
        self.assertEqual(model.fc.out_features,2)
if __name__=='__main__':unittest.main(verbosity=2)
