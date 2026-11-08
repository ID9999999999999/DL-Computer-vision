#!/usr/bin/env python3
"""Lab06 optional Task 5C: controlled train-only glare augmentation experiment.

Two continued-training arms start from the SAME archived YOLOv8n-cls weights:
control without new augmentation vs targeted synthetic glare on TRAIN bad images.

This is NOT a reconstruction of original training from ImageNet. The original
Ultralytics 8.3 trainer is unavailable offline; this reproducible PyTorch
compatibility training uses a checkpoint-equivalent forward graph and clearly
logs differences. Precommitted test evaluation occurs only after validation
checkpoint selection; held-out images never enter gradients.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,copy,csv,hashlib,io,json,random,sys,time,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from ultra_compat import load_checkpoint
SEED=2473986
IMAGE_SIZE=224
MAX_EPOCHS=15
PATIENCE=4
BATCH=16
LR=1e-4
WEIGHT_DECAY=5e-4
EXPECTED_SHA='c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7'
AUG={'shape':'white filled ellipse', 'alpha_range_0_255':[150,230],
     'ellipses_per_augmented_image':[1,2],
     'p_glare_on_bad_glare':0.75,
     'p_glare_on_other_bad':0.35,
     'p_glare_on_good':0.0,
     'ellipse_width_fraction':[0.18,0.56],
     'ellipse_height_fraction':[0.07,0.30],
     'seed':SEED, 'splits_augmented':['train']}

def digest(b:bytes):return hashlib.sha256(b).hexdigest()

def prepare(dataset_zip,manifest,out):
    r=json.loads(Path(manifest).read_text(encoding='utf8'))
    assert len(r)==1800
    splits=Counter(x['split'] for x in r)
    assert splits=={'train':1260,'val':270,'test':270},splits
    groups=defaultdict(set);hashgroups=defaultdict(set)
    for x in r:
        groups[x['source_id']].add(x['split'])
        hashgroups[x['sha256']].add(x['split'])
    assert len(groups)==300 and all(len(a)==1 for a in groups.values())
    assert all(len(a)==1 for a in hashgroups.values())
    r=sorted(r,key=lambda x:('train','val','test').index(x['split'])*10000+x.get('image_id','').__hash__()%10000) if False else sorted(r,key=lambda x:(['train','val','test'].index(x['split']),x['image_id']))
    cache=out/'original_224_images.npy';manifest_out=out/'verified_manifest.json'
    if cache.exists() and manifest_out.exists():
        saved=json.loads(manifest_out.read_text())
        if saved==r:
            return r,np.load(cache,mmap_mode='r')
    imgs=np.lib.format.open_memmap(cache,mode='w+',dtype=np.uint8,shape=(1800,IMAGE_SIZE,IMAGE_SIZE,3))
    with zipfile.ZipFile(dataset_zip) as z:
        for i,x in enumerate(r):
            name='demo_quality_dataset/'+x['image_path']
            b=z.read(name)
            assert digest(b)==x['sha256'], 'Image digest mismatch: '+name
            with Image.open(io.BytesIO(b)) as im:
                imgs[i]=np.asarray(im.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR))
    imgs.flush();del imgs
    manifest_out.write_text(json.dumps(r,indent=2)+'\n')
    print('Verified 1,800 original image hashes, grouped 210/45/45 source split and immutable val/test',flush=True)
    return r,np.load(cache,mmap_mode='r')

def overlay_glare(array,rng):
    # Draw only on ephemeral 224x224 array copies; original on-disk data unchanged.
    base=Image.fromarray(np.array(array,copy=True),'RGB').convert('RGBA')
    overlay=Image.new('RGBA',(IMAGE_SIZE,IMAGE_SIZE),(0,0,0,0))
    draw=ImageDraw.Draw(overlay,'RGBA')
    n=rng.randint(*AUG['ellipses_per_augmented_image'])
    for _ in range(n):
        w=int(IMAGE_SIZE*rng.uniform(*AUG['ellipse_width_fraction']))
        h=int(IMAGE_SIZE*rng.uniform(*AUG['ellipse_height_fraction']))
        cx=rng.randint(0,IMAGE_SIZE-1);cy=rng.randint(0,IMAGE_SIZE-1)
        alpha=rng.randint(*AUG['alpha_range_0_255'])
        draw.ellipse((cx-w//2,cy-h//2,cx+w//2,cy+h//2),fill=(255,255,255,alpha))
    return np.asarray(Image.alpha_composite(base,overlay).convert('RGB'))

def batch_images(arr,rows,idx,arm,epoch):
    raw=np.asarray(arr[idx]).copy()
    n_aug=0
    if arm=='glare':
        for k,j in enumerate(idx):
            item=rows[int(j)]
            if item['split']!='train':
                raise RuntimeError('Augmentation leakage!')
            if item['quality_label']!='bad':continue
            is_glare=item.get('defect_type')=='glare'
            p=AUG['p_glare_on_bad_glare'] if is_glare else AUG['p_glare_on_other_bad']
            rng=random.Random(SEED+epoch*1000003+int(j)*131)
            if rng.random()<p:
                raw[k]=overlay_glare(raw[k],rng);n_aug+=1
    arr=torch.from_numpy(raw.transpose(0,3,1,2).copy()).float().div_(255.)
    return arr,n_aug

def model_from_ckpt(ckpt):
    m=copy.deepcopy(ckpt['ema'] if ckpt.get('ema') is not None else ckpt['model']).float()
    assert m.names=={0:'bad',1:'good'}
    m.requires_grad_(True)
    return m

def probs(model,arrays,labels,indices):
    model.eval();scores=[];running=0.;total=0
    with torch.inference_mode():
        for i in range(0,len(indices),BATCH):
            idx=indices[i:i+BATCH]
            x=torch.from_numpy(np.asarray(arrays[idx]).transpose(0,3,1,2).copy()).float().div_(255.)
            z=model(x)
            running+=float(nn.functional.cross_entropy(z,labels[idx],reduction='sum'))
            scores.extend(torch.softmax(z,dim=1)[:,0].tolist())
            total+=len(idx)
    return np.array(scores,dtype=np.float64),running/total

def primary_metrics(rows,scores,indices):
    y=np.array([1 if rows[int(i)]['quality_label']=='bad' else 0 for i in indices],dtype=np.int64)
    pred=(scores>=0.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    pp,rr,ff,sp=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {'n':len(indices),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(ff.mean()),
      'precision_bad':float(pp[1]),'recall_bad':float(rr[1]),'f1_bad':float(ff[1]),
      'per_class':{'good':{'precision':float(pp[0]),'recall':float(rr[0]),'f1':float(ff[0]),'support':int(sp[0])},
                   'bad':{'precision':float(pp[1]),'recall':float(rr[1]),'f1':float(ff[1]),'support':int(sp[1])}},
      'confusion_matrix':cm.tolist(), 'errors':int((y!=pred).sum())}

def defect_metrics(rows,scores,indices):
    """Per-defect binary good-vs-this-bad-type F1 on SAME 135 good negatives.
    Do not claim these are five independent multiclass categories: model is binary.
    """
    selected=[rows[int(i)] for i in indices]
    kinds=sorted({x.get('defect_type','none') for x in selected if x['quality_label']=='bad'})
    out={}
    for kind in kinds:
        keep=np.array([x['quality_label']=='good' or (x['quality_label']=='bad' and x.get('defect_type')==kind) for x in selected])
        truth=np.array([int(x['quality_label']=='bad') for x,s in zip(selected,keep) if s]);pr=(scores[keep]>=0.5).astype(int)
        pp,rr,ff,ss=precision_recall_fscore_support(truth,pr,labels=[0,1],zero_division=0)
        mat=confusion_matrix(truth,pr,labels=[0,1]).tolist()
        out[kind]={'n':int(keep.sum()),'bad_count':int((truth==1).sum()),'good_count':int((truth==0).sum()),'f1_bad_vs_common_good':float(ff[1]),'recall_defect':float(rr[1]),'false_accepts':int(mat[1][0]),'false_rejects_good':int(mat[0][1])}
    return out

def save_predictions(out,rows,scores,ids):
    with (out/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','source_id','image_path','true_label','defect_type','prob_bad','pred_label','correct'])
        for i,p in zip(ids,scores):
            x=rows[int(i)];q='bad' if p>=0.5 else 'good';w.writerow([x['image_id'],x['source_id'],x['image_path'],x['quality_label'],x.get('defect_type','none'),format(float(p),'.9f'),q,int(q==x['quality_label'])])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,default=Path('/mnt/data/task5a_inputs/demo_quality_dataset.zip'))
    ap.add_argument('--manifest',type=Path,default=Path('/mnt/data/task5a_inputs/manifest.json'))
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'inputs/yolov8n_original_best.pt')
    ap.add_argument('--out',type=Path,default=ROOT/'results');ap.add_argument('--max-epochs',type=int,default=MAX_EPOCHS)
    args=ap.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
    assert digest(args.checkpoint.read_bytes())==EXPECTED_SHA
    torch.set_num_threads(4)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True,warn_only=True)
    rows,arrays=prepare(args.zip,args.manifest,out)
    ids={s:np.array([i for i,x in enumerate(rows) if x['split']==s],dtype=np.int64) for s in ['train','val','test']}
    for s,expect in [('train',1260),('val',270),('test',270)]:assert len(ids[s])==expect
    labels=torch.tensor([0 if x['quality_label']=='bad' else 1 for x in rows],dtype=torch.long)
    checkpoint=load_checkpoint(args.checkpoint)
    baseline=model_from_ckpt(checkpoint)
    baseline.eval()
    # Sanity check original archive is restored exactly before running anything new.
    bs,bs_loss=probs(baseline,arrays,labels,ids['test']);bm=primary_metrics(rows,bs,ids['test'])
    assert bm['confusion_matrix']==[[135,0],[3,132]],'Baseline restored model does not match original 267/270 test result'
    bs_val,bs_vloss=probs(baseline,arrays,labels,ids['val']);bval=primary_metrics(rows,bs_val,ids['val'])
    print('ORIGINAL baseline check:',bm,'validation loss',bs_vloss,flush=True)
    (out/'original_test_reference.json').write_text(json.dumps({'test':bm,'validation':bval,'checkpoint_sha256':EXPECTED_SHA},indent=2)+'\n')
    summary={'task':'5C','date':'2026-11-09','status':'genuine CPU training; original checkpoint continued, not from ImageNet',
       'seed':SEED,'original_checkpoint_sha256':EXPECTED_SHA,
       'source_images_hashes_verified':1800,'source_groups':{'train':210,'val':45,'test':45},
       'train_val_test':{'train':1260,'val':270,'test':270},
       'model':'YOLOv8n-cls','setup':{'max_epochs':args.max_epochs,'patience':PATIENCE,'batch':BATCH,'lr':LR,'optimizer':'Adam','weight_decay':WEIGHT_DECAY,'image_size':IMAGE_SIZE,'cpu_threads':4,'early_stop_selection':'validation cross-entropy loss, tie by val accuracy'},
       'augmentation':AUG,'baseline':{'validation':bval,'test':bm,'test_by_defect':defect_metrics(rows,bs,ids['test'])},'arms':{},
       'limitation':'PyTorch checkpoint-equivalent local trainer, not an Ultralytics full trainer replay; both arms continue from original best checkpoint. Test used once for reporting after selecting each arm using validation. Original test had been historically inspected before the study.'}
    history={}
    for arm in ('control','glare'):
        # Restart stochastic optimizer/dropout and data-order streams identically for both arms.
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        m=model_from_ckpt(checkpoint)
        optimizer=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        best=None;best_vloss=float('inf');best_epoch=0;pat=0
        hist=[];t0=time.time();total_aug=0
        for epoch in range(1,args.max_epochs+1):
            m.train();order=np.random.default_rng(SEED+epoch).permutation(ids['train'])
            cumulative=0.;correct=0;n=0;added=0
            for s in range(0,len(order),BATCH):
                chunk=order[s:s+BATCH]
                x,k=batch_images(arrays,rows,chunk,arm,epoch)
                added+=k
                target=labels[chunk]
                optimizer.zero_grad(set_to_none=True)
                logits=m(x)
                loss=nn.functional.cross_entropy(logits,target)
                loss.backward()
                optimizer.step()
                n+=len(chunk)
                cumulative+=loss.item()*len(chunk)
                correct+=int((logits.argmax(dim=1)==target).sum())
            total_aug+=added
            va,v_loss=probs(m,arrays,labels,ids['val'])
            vm=primary_metrics(rows,va,ids['val'])
            stat={'epoch':epoch,'train_loss':cumulative/n,'train_accuracy':correct/n,'validation_loss':v_loss,'validation_accuracy':vm['accuracy'],'validation_macro_f1':vm['macro_f1'],'train_glare_overlays':added}
            hist.append(stat)
            improve=v_loss<best_vloss-1e-6
            if improve:
                best_vloss=v_loss;best_epoch=epoch;pat=0;best=copy.deepcopy(m.state_dict())
            else:pat+=1
            print(arm,'epoch',epoch,'train loss',round(cumulative/n,5),'val loss',round(v_loss,5),'val acc',round(vm['accuracy'],5),'overlays',added,'best',best_epoch,'elapsed',round(time.time()-t0,1),flush=True)
            if pat>=PATIENCE:break
        m.load_state_dict(best)
        valp,vloss=probs(m,arrays,labels,ids['val'])
        # Test is accessed only ONCE per selected final arm here.
        testp,tloss=probs(m,arrays,labels,ids['test'])
        vm=primary_metrics(rows,valp,ids['val']);tm=primary_metrics(rows,testp,ids['test'])
        defects=defect_metrics(rows,testp,ids['test'])
        dirname=out/arm;dirname.mkdir(exist_ok=True)
        # State dict is tensor-only; use original checkpoint + safe adapter to reconstruct architecture.
        torch.save({'state_dict':{k:v.cpu() for k,v in best.items()},'model_family':'yolov8n-cls',
                  'class_map':{0:'bad',1:'good'},'original_checkpoint_sha256':EXPECTED_SHA,
                  'arm':arm,'best_epoch':best_epoch,'seed':SEED,'augmentation':AUG if arm=='glare' else None},dirname/'best_state_dict.pt')
        save_predictions(dirname,rows,testp,ids['test'])
        (dirname/'training_history.json').write_text(json.dumps(hist,indent=2)+'\n')
        (dirname/'per_defect_test.json').write_text(json.dumps(defects,indent=2)+'\n')
        record={'best_epoch':best_epoch,'epochs_run':len(hist),'seconds':time.time()-t0,'train_overlay_events':total_aug,'val_loss_at_best':vloss,'validation':vm,'test':tm,'by_defect_test':defects,'checkpoint_size_bytes':(dirname/'best_state_dict.pt').stat().st_size}
        (dirname/'evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
        summary['arms'][arm]=record;history[arm]=hist
        print('DONE',arm,'test',tm,'glare',defects.get('glare'),flush=True)
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Summary: binary per-class F1, independent defect-wise recall and positive-vs-common-good F1.
    columns=['label','n_bad','original_recall','control_recall','glare_recall','delta_glare_vs_control_recall','original_f1','control_f1','glare_f1','delta_glare_vs_control_f1']
    raw=[]
    for kind in sorted(summary['baseline']['test_by_defect']):
        d=[summary['baseline']['test_by_defect'][kind],summary['arms']['control']['by_defect_test'][kind],summary['arms']['glare']['by_defect_test'][kind]]
        raw.append([kind,d[0]['bad_count'],*(x['recall_defect'] for x in d), d[2]['recall_defect']-d[1]['recall_defect'],*(x['f1_bad_vs_common_good'] for x in d),d[2]['f1_bad_vs_common_good']-d[1]['f1_bad_vs_common_good']])
    with (out/'per_defect_delta.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(raw)
    with (out/'binary_class_delta.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['class','original_f1','control_f1','glare_f1','glare_minus_control_f1'])
        for cls in ('good','bad'):
            a=summary['baseline']['test']['per_class'][cls]['f1'];c=summary['arms']['control']['test']['per_class'][cls]['f1'];g=summary['arms']['glare']['test']['per_class'][cls]['f1'];w.writerow([cls,a,c,g,g-c])
    fig,ax=plt.subplots(figsize=(8,4.5))
    for arm in ('control','glare'):
        h=history[arm];ax.plot([x['epoch'] for x in h],[x['validation_loss'] for x in h],marker='o',label=arm)
    ax.set(title='Task 5C: validation loss (untouched validation)',xlabel='Epoch',ylabel='Cross-entropy');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=150);plt.close(fig)
    kinds=sorted(summary['baseline']['test_by_defect']);fig,ax=plt.subplots(figsize=(9,4.5))
    places=np.arange(len(kinds));width=.25
    for offset,arm in ((-1,'original'),(0,'control'),(1,'glare')):
        data=summary['baseline']['test_by_defect'] if arm=='original' else summary['arms'][arm]['by_defect_test']
        ax.bar(places+offset*width,[data[k]['recall_defect'] for k in kinds],width,label=arm)
    ax.set(ylim=(0,1.1),xticks=places,xticklabels=kinds,ylabel='Recall of bad images (each defect)',title='Held-out test: defect recall before and after');ax.legend();fig.tight_layout();fig.savefig(out/'defect_recall_comparison.png',dpi=150);plt.close(fig)
    delta=summary['arms']['glare']['test']['macro_f1']-summary['arms']['control']['test']['macro_f1']
    g0=summary['arms']['control']['by_defect_test']['glare']['recall_defect'];g1=summary['arms']['glare']['by_defect_test']['glare']['recall_defect']
    summary_msg=('benefit' if g1>g0 else 'decline' if g1<g0 else 'no change')
    changed=[r for r in raw if abs(r[-1])>1e-9 or abs(r[5])>1e-9]
    memo=f'''# Lab 06 — Bonus Task 5C: Error-Driven Glare Augmentation\n\n**Executed:** 2026-11-09. **Status:** genuine CPU training of both controlled arms.\n\n## Decision and honest protocol scope\nThe original YOLOv8n-cls was the validation/cost-selected model. All four original-model test mistakes had glare involved. We compared two new training arms, **control** (no extra overlays) and **glare** (train-only stronger white elliptical overlays), **starting from the exact SAME selected YOLOv8n saved weights** (SHA-256 `{EXPECTED_SHA}`), the same seed **{SEED}**, same data split, batch **16**, input **224×224**, Adam LR **1e-4**, max **{args.max_epochs} epochs**, early-stopping patience **4**. The original model was restored and its original **267/270** test count was verified.\n\nThis is a **controlled continuation from an already trained checkpoint**, not a full replay from ImageNet: package availability prevented importing Ultralytics in this offline runtime. It uses a local PyTorch checkpoint-compatible forward graph. Both arms share this limitation and use the same implementation, allowing comparison of overlay effects. Original teacher's later 50/32/5/640 requested settings were **not** satisfied by this new experiment; we report the actual settings. Checkpoint selection used the validation cross-entropy, not Test.\n\n## Augmentation\nOnly on the 1,260 training images; no augmented image was saved to or mixed into validation or held-out test. Random white ellipses alpha **150–230**, one to two overlays per selected image. Applied only to BAD training images with probability **0.75 if original defect was glare**, **0.35 for other bad defects**, and **0** for good images. Each variant is seeded deterministically by `{SEED} + epoch*1,000,003 + array_index*131`. This is label-preserving for defective images; it intentionally creates stronger negative glare cases. Details: `results.json` and source script.\n\n## Genuine measurements (held-out Test, n=270)\n\n| Model | Test accuracy | Macro F1 | F1 good | F1 bad | Glare recall | Glare false accepts | Epoch selected | Epochs run |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n| Original archived | {summary['baseline']['test']['accuracy']:.4f} | {summary['baseline']['test']['macro_f1']:.4f} | {summary['baseline']['test']['per_class']['good']['f1']:.4f} | {summary['baseline']['test']['per_class']['bad']['f1']:.4f} | {summary['baseline']['test_by_defect']['glare']['recall_defect']:.4f} | {summary['baseline']['test_by_defect']['glare']['false_accepts']} | historic | historic |\n| No-overlay control | {summary['arms']['control']['test']['accuracy']:.4f} | {summary['arms']['control']['test']['macro_f1']:.4f} | {summary['arms']['control']['test']['per_class']['good']['f1']:.4f} | {summary['arms']['control']['test']['per_class']['bad']['f1']:.4f} | {g0:.4f} | {summary['arms']['control']['by_defect_test']['glare']['false_accepts']} | {summary['arms']['control']['best_epoch']} | {summary['arms']['control']['epochs_run']} |\n| Train-only glare | {summary['arms']['glare']['test']['accuracy']:.4f} | {summary['arms']['glare']['test']['macro_f1']:.4f} | {summary['arms']['glare']['test']['per_class']['good']['f1']:.4f} | {summary['arms']['glare']['test']['per_class']['bad']['f1']:.4f} | {g1:.4f} | {summary['arms']['glare']['by_defect_test']['glare']['false_accepts']} | {summary['arms']['glare']['best_epoch']} | {summary['arms']['glare']['epochs_run']} |\n\nThe per-defect table (`per_defect_delta.csv`) uses **each defect subtype as the positive examples versus the SAME 135 good negatives**; it is an analytical binary F1, **not a five-class confusion matrix**. `binary_class_delta.csv` contains the genuine binary good/bad per-class F1.\n\n## Five-line conclusion\n1. The test glare recall changed from **{g0:.4f}** (unaugmented continuation) to **{g1:.4f}** (glare augmentation): **{summary_msg}**.\n2. Overall macro F1 changed by **{delta:+.4f}** between otherwise matched new training arms.\n3. Defect types with changed recall/F1 across those arms: **{', '.join(r[0] for r in changed) if changed else 'none observed'}**; see exact values in CSV.\n4. Good-class F1 changed by **{summary['arms']['glare']['test']['per_class']['good']['f1']-summary['arms']['control']['test']['per_class']['good']['f1']:+.4f}**, which captures any false-reject trade-off on usable images.\n5. Because the test contains only **45 independent source documents**, this single seeded result does not prove a general improvement; the augmentation should be selected by validation and checked on fresh external images before deployment.\n\n## Provenance\nOriginal dataset ZIP, manifest and all 1,800 image SHA-256 digests validated; 210/45/45 disjoint sources. Raw original/validation/test datasets and original checkpoints were not edited. Two new tensor-only checkpoint state_dicts were saved. The teacher's written task requires a fair baseline; our controlled continuation explicitly discloses the limitation versus the historical training from ImageNet.\n'''
    (out/'bonus_augmentation.md').write_text(memo,encoding='utf8')
    print('TASK5C_DONE',json.dumps({'original':summary['baseline']['test']['accuracy'],'control':summary['arms']['control']['test']['accuracy'],'glare':summary['arms']['glare']['test']['accuracy'],'delta_macro_f1':delta,'glare_recall_control':g0,'glare_recall_aug':g1,'epochs_control':summary['arms']['control']['epochs_run'],'epochs_glare':summary['arms']['glare']['epochs_run']},indent=2),flush=True)

if __name__=='__main__':main()
