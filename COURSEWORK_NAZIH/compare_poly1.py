"""Controlled *development-only* comparison, ResNet18 CE vs Poly-1 CE.
Uses the exact historical supervised pilot manifest, resolution and hyperparameters.
This experiment reuses previously accessed images; validation is NOT a new blind test.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,random,time
from pathlib import Path
import numpy as np
import torch,torchvision
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision.models import resnet18,ResNet18_Weights
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'resnet18_supervised/manifest.json'
CONFIG=ROOT/'resnet18_supervised/config.json'
WEIGHTS_SHA='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'

def metrics(truth,pred):
    cm=np.zeros((2,2),dtype=int)
    for y,p in zip(truth,pred):cm[int(y),int(p)]+=1
    out={'confusion_matrix':cm.tolist(),'n':int(cm.sum()),'accuracy':float(np.trace(cm)/cm.sum())}
    ps=[];rs=[];fs=[]
    for c in (0,1):
        tp=int(cm[c,c]);fp=int(cm[:,c].sum()-tp);fn=int(cm[c,:].sum()-tp)
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        ps.append(precision);rs.append(recall);fs.append(f1)
    out.update({'precision_normal':ps[0],'precision_defective':ps[1],
                'recall_normal':rs[0],'recall_defective':rs[1],
                'f1_normal':fs[0],'f1_defective':fs[1],
                'macro_f1':float(np.mean(fs))})
    return out

def poly1_cross_entropy(logits,labels,epsilon=1.0):
    # Eq. in PolyLoss (Leng et al., ICLR 2022): CE + epsilon * (1 - p_true).
    ce=F.cross_entropy(logits,labels,reduction='none')
    true_probability=F.softmax(logits,dim=1).gather(1,labels[:,None]).squeeze(1)
    return (ce + epsilon*(1.-true_probability)).mean()

def load_rows_and_images(cfg):
    rows=json.loads(MANIFEST.read_text())
    assert len(rows)==99 and {r['split'] for r in rows}=={'train','validation'}
    assert {r['label'] for r in rows}=={0,1}
    assert {r['group'] for r in rows if r['split']=='train'}.isdisjoint({r['group'] for r in rows if r['split']=='validation'})
    assert len({r['path'] for r in rows})==len(rows)
    assert {r['path'] for r in rows if r['split']=='train'}.isdisjoint({r['path'] for r in rows if r['split']=='validation'})
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    val=[i for i,r in enumerate(rows) if r['split']=='validation']
    assert len(train)==70 and len(val)==29
    assert {y:sum(rows[i]['label']==y for i in train) for y in (0,1)}=={0:35,1:35}
    assert {y:sum(rows[i]['label']==y for i in val) for y in (0,1)}=={0:15,1:14}
    missing=[r['path'] for r in rows if not (ROOT/r['path']).exists()]
    if missing:raise FileNotFoundError(f'Missing {len(missing)} original images. Run python download_missing.py. Example: {missing[:3]}')
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    images=[]
    for r in rows:
        path=ROOT/r['path'];b=path.read_bytes()
        if hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError(f'Image hash mismatch: {r["path"]}')
        with Image.open(path) as im:
            arr=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)).copy()
        image=torch.from_numpy(arr).permute(2,0,1).float()/255
        images.append((image-mean)/std)
    return rows,train,val,images

def set_seed(seed):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def model_initial_state(seed):
    set_seed(seed)
    # Keep torchvision construction/order same as historical CE pilot.
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    wp=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    if not wp.exists() or hashlib.sha256(wp.read_bytes()).hexdigest()!=WEIGHTS_SHA:
        raise RuntimeError('Verified ImageNet ResNet18 weights unavailable. Connect to network for one-time download; never substitute random weights.')
    model.fc=nn.Linear(model.fc.in_features,2)
    sha=hashlib.sha256()
    for k,v in model.state_dict().items():sha.update(k.encode());sha.update(v.detach().cpu().numpy().tobytes())
    return model,sha.hexdigest()

def evaluate(model,images,rows,val,batch_size,loss_name,epsilon,device):
    model.eval();truth=[];pred=[];records=[];losssum=0.
    with torch.inference_mode():
        for start in range(0,len(val),batch_size):
            ids=val[start:start+batch_size]
            x=torch.stack([images[i] for i in ids]).to(device)
            y=torch.tensor([rows[i]['label'] for i in ids],device=device)
            z=model(x)
            loss=poly1_cross_entropy(z,y,epsilon) if loss_name=='poly1' else F.cross_entropy(z,y)
            losssum+=loss.item()*len(ids)
            prob=z.softmax(1)
            for k,i in enumerate(ids):
                t=int(y[k]);p=int(z[k].argmax());truth.append(t);pred.append(p)
                records.append({'object':rows[i]['object'],'group':rows[i]['group'],'true_label':t,
                                'predicted_label':p,'prob_defective':float(prob[k,1])})
    stats=metrics(truth,pred);stats['validation_loss']=losssum/len(val)
    return stats,records

def train_one(loss_name,rows,train,val,images,cfg,args,output_dir,device):
    model,initial_hash=model_initial_state(cfg['seed']);model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    batch_size=cfg['batch_size'];history=[];best=-1.;best_epoch=None;best_rows=None;best_stats=None
    selected_conv_init=model.conv1.weight.detach().clone();selected_fc_init=model.fc.weight.detach().clone()
    beginning=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        model.train();order=torch.randperm(len(train)).tolist();train_loss=0.
        for start in range(0,len(order),batch_size):
            ids=[train[j] for j in order[start:start+batch_size]]
            x=torch.stack([images[i] for i in ids]).to(device)
            y=torch.tensor([rows[i]['label'] for i in ids],device=device)
            opt.zero_grad(set_to_none=True);z=model(x)
            loss=poly1_cross_entropy(z,y,args.epsilon) if loss_name=='poly1' else F.cross_entropy(z,y)
            assert torch.isfinite(loss),'Non-finite loss'
            loss.backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            opt.step();train_loss+=loss.item()*len(ids)
        stats,per_image=evaluate(model,images,rows,val,batch_size,loss_name,args.epsilon,device)
        history.append({'epoch':epoch,'training_loss':train_loss/len(train),'validation':stats})
        print(loss_name,'epoch',epoch,'val macro-F1',round(stats['macro_f1'],5),flush=True)
        if stats['macro_f1']>best:
            best=stats['macro_f1'];best_epoch=epoch;best_rows=per_image;best_stats=stats
            torch.save(model.state_dict(),output_dir/f'{loss_name}_best.pt')
    updates={'conv1_max_delta':float((model.conv1.weight-selected_conv_init).abs().max()),
             'fc_max_delta':float((model.fc.weight-selected_fc_init).abs().max())}
    assert updates['conv1_max_delta']>0 and updates['fc_max_delta']>0
    with (output_dir/f'{loss_name}_validation_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(best_rows[0]));writer.writeheader();writer.writerows(best_rows)
    result={'method':loss_name,'initial_state_sha256':initial_hash,'best_epoch':best_epoch,'best_validation':best_stats,
            'training_history':history,'parameter_updates':updates,'train_seconds':time.perf_counter()-beginning}
    (output_dir/f'{loss_name}_result.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['both','ce','poly1'],default='both')
    ap.add_argument('--epsilon',type=float,default=1.)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--out',default='comparison_results')
    args=ap.parse_args()
    assert args.epsilon>=-1 and np.isfinite(args.epsilon)
    cfg=json.loads(CONFIG.read_text())
    assert cfg['epochs']==3 and cfg['batch_size']==4 and cfg['input_size']==448
    assert cfg['seed']==20261028 and cfg['learning_rate']==.0001 and cfg['weight_decay']==.0001
    torch.set_num_threads(cfg['threads'])
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    print('Development validation ONLY. Source data previously accessed; not independent confirmation.',flush=True)
    rows,train,val,images=load_rows_and_images(cfg)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    names=['ce','poly1'] if args.mode=='both' else [args.mode]
    results={}
    for name in names:results[name]=train_one(name,rows,train,val,images,cfg,args,out,device)
    if len(results)==2:
        assert results['ce']['initial_state_sha256']==results['poly1']['initial_state_sha256'], 'Initial weights diverged!'
        ce=results['ce']['best_validation'];po=results['poly1']['best_validation']
        diff={'delta_macro_f1':po['macro_f1']-ce['macro_f1'],
              'delta_defective_recall':po['recall_defective']-ce['recall_defective'],
              'same_initialization':True,
              'claim':'Development comparison only; no independent final test or confirmed specimen identity'}
        (out/'comparison.json').write_text(json.dumps(diff,indent=2))
        print('comparison',diff)
    (out/'run_metadata.json').write_text(json.dumps({'validation_only':True,'old_test_reused':True,
        'verified_physical_independence':False,'seed':cfg['seed'],'epsilon':args.epsilon,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__},
        'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
