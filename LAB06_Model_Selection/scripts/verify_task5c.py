"""Verify distributed Lab 06 Task 5C artifact, with optional full image inference."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'results/results.json').exists() else HERE.parent
sys.path.insert(0,str(HERE))
from task5c_glare_train import load_checkpoint,model_from_ckpt,probs,primary_metrics,prepare,EXPECTED_SHA,SEED

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',type=Path,default=None,help='Optional original 399MB dataset zip for full image hash and inference QA')
    opt=parser.parse_args()
    res=ROOT/'results';summary=json.loads((res/'results.json').read_text())
    allrows=json.loads((ROOT/'metadata/manifest.json').read_text()) if (ROOT/'metadata/manifest.json').exists() else json.loads((res/'verified_manifest.json').read_text())
    rows=sorted(allrows,key=lambda x:(['train','val','test'].index(x['split']),x['image_id']))
    assert len(rows)==1800
    parts={s:[x for x in rows if x['split']==s] for s in ['train','val','test']}
    assert {k:len(v) for k,v in parts.items()}=={'train':1260,'val':270,'test':270}
    source_sets={s:{x['source_id'] for x in v} for s,v in parts.items()}
    assert {s:len(v) for s,v in source_sets.items()}=={'train':210,'val':45,'test':45}
    assert not(source_sets['train']&source_sets['val'] or source_sets['train']&source_sets['test'] or source_sets['val']&source_sets['test'])
    original=ROOT/'weights/yolov8n_original_best.pt'
    if not original.exists():original=ROOT/'inputs/yolov8n_original_best.pt'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==EXPECTED_SHA
    ckpt=load_checkpoint(original)
    base=(ckpt['ema'] if ckpt.get('ema') is not None else ckpt['model']).float().state_dict()
    results={}
    for arm in ['control','glare']:
        with torch.serialization.safe_globals([]):
            model_record=torch.load(res/arm/'best_state_dict.pt',weights_only=True,map_location='cpu')
        assert model_record['original_checkpoint_sha256']==EXPECTED_SHA
        changed=sum(torch.count_nonzero(v.float()-base[k].float()).item() for k,v in model_record['state_dict'].items())
        assert changed>0,arm
        with (res/arm/'test_predictions.csv').open(newline='') as f:p=list(csv.DictReader(f))
        assert len(p)==270
        expected=summary['arms'][arm]['test']
        counter=Counter((x['true_label'],x['pred_label']) for x in p)
        cm=[[counter['good','good'],counter['good','bad']],[counter['bad','good'],counter['bad','bad']]]
        assert cm==expected['confusion_matrix'],(arm,cm,expected['confusion_matrix'])
        for item,r in zip(p,parts['test']):
            assert item['image_id']==r['image_id'] and item['true_label']==r['quality_label']
            assert (float(item['prob_bad'])>=0.5)==(item['pred_label']=='bad')
        results[arm]={'checkpoint_changes':int(changed),'test_predictions':len(p),'cm_matches_report':True}
    print('Artifact audit passed: manifest split, class mapping, original SHA, new trained weights, and 270 predictions per arm.')
    if opt.zip is not None:
        (ROOT/'temp_verified_cache').mkdir(parents=True,exist_ok=True)
        d,arrays=prepare(opt.zip,ROOT/'metadata/manifest.json' if (ROOT/'metadata/manifest.json').exists() else res/'verified_manifest.json',ROOT/'temp_verified_cache')
        ids=np.array([i for i,x in enumerate(d) if x['split']=='test'],dtype=np.int64)
        labels=torch.tensor([0 if x['quality_label']=='bad' else 1 for x in d],dtype=torch.long)
        torch.set_num_threads(4)
        for arm in ['control','glare']:
            with torch.serialization.safe_globals([]):saved=torch.load(res/arm/'best_state_dict.pt',weights_only=True,map_location='cpu')
            m=model_from_ckpt(ckpt)
            m.load_state_dict(saved['state_dict'],strict=True)
            scores,_=probs(m,arrays,labels,ids)
            with (res/arm/'test_predictions.csv').open(newline='') as f:recs=list(csv.DictReader(f))
            difference=max(abs(float(x['prob_bad'])-float(p)) for x,p in zip(recs,scores))
            assert difference<1e-6,(arm,difference)
            results[arm]['model_reproduced_from_images']=True
            results[arm]['max_probability_delta']=difference
        print('Full image-level verification passed: 1,800 SHA-256 checks and exact checkpoint inference on 270 test images for both models.')
    print(json.dumps({'status':'PASS','full_image_verification':bool(opt.zip),'arms':results},indent=2))

if __name__=='__main__':main()
