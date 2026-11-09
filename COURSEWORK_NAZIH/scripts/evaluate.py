"""Exploratory evaluation AFTER all 18 new checkpoints are frozen. Never tunes model or thresholds."""
from pathlib import Path
import argparse,hashlib,io,json,sys,time,zipfile
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,precision_recall_fscore_support
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config,read_json,sha_file,verify_archive,digest_state
MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]

def frozen_checkpoint_inventory():
    c=config();count=0
    for size in c['new_training_sizes']:
        for seed in c['seeds']:
            for loss in c['arms']:
                info=ROOT/f'results/size{size}_seed{seed}_{loss}.json'
                pt=ROOT/f'results/size{size}_seed{seed}_{loss}.pt'
                if not info.exists() or not pt.exists():raise RuntimeError(f'NOT FROZEN: {info} or {pt} missing. Complete all training before opening test images.')
                assert read_json(info)['checkpoint_sha256']==sha_file(pt),'Modified checkpoint found'
                count+=1
    assert count==18
    return count

def scoring(y,p,score):
    cm=[[sum(a==i and b==j for a,b in zip(y,p)) for j in (0,1)] for i in (0,1)]
    f1=[2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in (0,1)]
    return dict(macro_f1=sum(f1)/2,accuracy=sum(cm[i][i] for i in (0,1))/len(y),
         defective_recall=cm[1][1]/sum(cm[1]),normal_fpr=cm[0][1]/sum(cm[0]),
         balanced_accuracy=float(balanced_accuracy_score(y,p)),
         auroc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
         confusion_matrix=cm)

def evaluate(archive,size,seed,arm):
    c=config();assert size in c['new_training_sizes'] and seed in c['seeds'] and arm in c['arms']
    frozen_checkpoint_inventory()
    out=ROOT/f'results/eval_size{size}_seed{seed}_{arm}.json'
    if out.exists():raise FileExistsError('Refusing to overwrite prior test output')
    pt=ROOT/f'results/size{size}_seed{seed}_{arm}.pt';test=read_json(ROOT/'input/test_manifest.json')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    m=resnet18(weights=None);m.fc=torch.nn.Linear(512,2)
    m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True));m.eval()
    before=digest_state(m.state_dict());pred=[];start=time.monotonic()
    with zipfile.ZipFile(archive) as z,torch.inference_mode():
        for i in range(0,1004,4):
            rows=test[i:i+4];batch=[]
            for row in rows:
                data=z.read(row['path']);assert hashlib.sha256(data).hexdigest()==row['sha256']
                with Image.open(io.BytesIO(data)) as im:
                    arr=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC),dtype=np.uint8).copy()
                x=torch.from_numpy(arr).permute(2,0,1).float().div_(255)
                batch.append((x-MEAN)/STD)
            logits=m(torch.stack(batch))
            for row,zlog in zip(rows,logits):
                pred.append({'path':row['path'],'sha256':row['sha256'],'label':row['label'],
                             'prediction':int(zlog.argmax()),'defect_score':float(zlog[1]-zlog[0])})
    assert digest_state(m.state_dict())==before
    y=[x['label'] for x in pred];p=[x['prediction'] for x in pred];score=[x['defect_score'] for x in pred]
    data={'size':size,'seed':seed,'loss':arm,'metrics':scoring(y,p,score),'predictions':pred,
        'checkpoint_sha256':sha_file(pt),'elapsed_seconds':time.monotonic()-start,
        'status':'exploratory: 1004 previously used images, not independent final confirmation'}
    out.write_text(json.dumps(data,indent=2)+'\n');print('EVALUATED:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True)
    p.add_argument('--size',required=True,type=int);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--loss',required=True,choices=['ce','poly1']);a=p.parse_args()
    with verify_archive(a.archive) as z:pass
    evaluate(a.archive,a.size,a.seed,a.loss)