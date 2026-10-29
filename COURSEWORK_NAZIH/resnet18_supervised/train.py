import json,hashlib,random,time,csv,platform
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models import resnet18,ResNet18_Weights
R=Path(__file__).resolve().parent;B=R.parent
cfg=json.loads((R/'config.json').read_text());rows=json.loads((R/'manifest.json').read_text())
torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.use_deterministic_algorithms(True)
assert {r['group'] for r in rows if r['split']=='train'}.isdisjoint({r['group'] for r in rows if r['split']=='validation'})
tensors=[]
for r in rows:
 p=B/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 with Image.open(p) as im:a=torch.from_numpy(np.array(im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC))).permute(2,0,1).float()/255
 tensors.append((a-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
wp=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth';assert hashlib.sha256(wp.read_bytes()).hexdigest()=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
model.fc=torch.nn.Linear(model.fc.in_features,2)
opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);lossfn=torch.nn.CrossEntropyLoss()
initial={k:model.state_dict()[k].clone() for k in ['conv1.weight','fc.weight']}
train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation'];history=[];best=-1;bestpred=None;start=time.perf_counter()
def evaluate():
 model.eval();cm=np.zeros((2,2),dtype=int);preds=[];loss=0
 with torch.inference_mode():
  for start in range(0,len(val),cfg['batch_size']):
   ix=val[start:start+cfg['batch_size']];x=torch.stack([tensors[i] for i in ix]);y=torch.tensor([rows[i]['label'] for i in ix]);z=model(x);loss+=float(lossfn(z,y))*len(ix);prob=z.softmax(1)
   for i,zz,pr in zip(ix,z,prob):
    pred=int(zz.argmax());cm[rows[i]['label'],pred]+=1;preds.append(dict(object=rows[i]['object'],label=rows[i]['label'],prediction=pred,prob_defective=float(pr[1])))
 f1=[]
 for c in range(2):
  tp=cm[c,c];fp=cm[:,c].sum()-tp;fn=cm[c,:].sum()-tp;f1.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
 return {'loss':loss/len(val),'macro_f1':sum(f1)/2,'accuracy':float(np.trace(cm)/cm.sum()),'confusion_matrix':cm.tolist()},preds
for epoch in range(1,cfg['epochs']+1):
 model.train();order=torch.randperm(len(train)).tolist();total=0
 for startbatch in range(0,len(order),cfg['batch_size']):
  ix=[train[j] for j in order[startbatch:startbatch+cfg['batch_size']]];x=torch.stack([tensors[i] for i in ix]);y=torch.tensor([rows[i]['label'] for i in ix]);opt.zero_grad(set_to_none=True);z=model(x);loss=lossfn(z,y);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();total+=float(loss.detach())*len(ix)
 metrics,preds=evaluate();rec={'epoch':epoch,'train_loss':total/len(train),'validation':metrics};history.append(rec);print(json.dumps(rec),flush=True)
 if metrics['macro_f1']>best:
  best=metrics['macro_f1'];bestpred=preds;torch.save(model.state_dict(),R/'best_model.pt');best_epoch=epoch
 (R/'history.json').write_text(json.dumps(history,indent=2))
updates={k:float((model.state_dict()[k]-v).abs().max()) for k,v in initial.items()};assert all(v>0 for v in updates.values())
state=torch.load(R/'best_model.pt',map_location='cpu',weights_only=True);model.load_state_dict(state);m,p=evaluate();assert p==bestpred
fresh=resnet18(weights=None);fresh.fc=torch.nn.Linear(512,2);fresh.load_state_dict(state);fresh.eval()
with torch.inference_mode():reload_error=float((fresh(tensors[val[0]][None])-model(tensors[val[0]][None])).abs().max())
assert reload_error<1e-6
with (R/'validation_predictions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(p[0]));w.writeheader();w.writerows(p)
result={'status':'Completed supervised engineering pilot; NOT independent test evidence','best_epoch':best_epoch,'validation':m,'parameter_count':sum(p.numel() for p in model.parameters()),'parameter_update_max_abs':updates,'checkpoint_reload_logit_max_abs_error':reload_error,'runtime_seconds':time.perf_counter()-start,'environment':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__},'checkpoint_sha256':hashlib.sha256((R/'best_model.pt').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((R/'config.json').read_bytes()).hexdigest()}
(R/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
