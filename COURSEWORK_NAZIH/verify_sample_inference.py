"""Existing checkpoint sanity on four previously known normal samples. Not performance estimation."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'resnet18_supervised/manifest.json').read_text())
subset=[r for r in rows if (ROOT/r['path']).exists()]
assert len(subset)>=4
model=resnet18(weights=None);model.fc=torch.nn.Linear(512,2)
model.load_state_dict(torch.load(ROOT/'resnet18_supervised/best_model.pt',map_location='cpu',weights_only=True));model.eval()
mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
res=[]
with torch.inference_mode():
    for r in subset:
        p=ROOT/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        with Image.open(p) as im:a=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)).copy()
        x=(torch.from_numpy(a).permute(2,0,1).float()/255-mean)/std
        logits=model(x[None]);res.append({'object':r['object'],'true_label':r['label'],'prediction':int(logits.argmax(1)[0]),'prob_defective':float(logits.softmax(1)[0,1])})
output={'purpose':'checkpoint reload/inference sanity only; not independent accuracy or F1','n':len(res),'rows':res}
(ROOT/'sample_inference_result.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
