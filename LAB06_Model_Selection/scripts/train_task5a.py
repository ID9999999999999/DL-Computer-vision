#!/usr/bin/env python3
"""Task 5A: exact frozen-checkpoint -> top-30 MobileNetV2 partial fine-tuning.

Reuses the archived 70/15/15 source-grouped manifest and original Keras checkpoint.
The checkpoint embeds MobileNetV2 pixel rescaling; images are supplied in 0..255.
Runs under Keras 3 torch backend when TensorFlow is unavailable, without changing
source weights or experimental training hyperparameters.
"""
from __future__ import annotations
import os
os.environ.setdefault('KERAS_BACKEND','torch')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,hashlib,json,sys,time,platform,copy
from pathlib import Path
import numpy as np
from PIL import Image
import keras
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=2473986
EPOCHS=50
PATIENCE=5
BATCH=32
LEARNING_RATE=1e-5
UNFREEZE_LAST=30
IMAGE_SIZE=224
RESULT_REL=Path('results/mobilenetv2_finetune_top30')
EXPECTED_BASELINE_SHA256='014226cd942aee422a04b19835eba9bf89c198939b996a88cf14b955eeed854e'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check_manifest(rows,dataset_root):
    assert len(rows)==1800, 'Expected original 1800-row manifest'
    required={('train','good'):630,('train','bad'):630,('val','good'):135,('val','bad'):135,('test','good'):135,('test','bad'):135}
    for (split,cls),n in required.items():
        assert sum(r['split']==split and r['quality_label']==cls for r in rows)==n,(split,cls)
    source_groups={s:{r['source_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert all(not(source_groups[a]&source_groups[b]) for a,b in [('train','val'),('train','test'),('val','test')]),'Source leakage'
    assert set(source_groups['train']|source_groups['val']|source_groups['test'])=={r['source_id'] for r in rows}
    assert len(source_groups['train'])==210 and len(source_groups['val'])==45 and len(source_groups['test'])==45
    for r in rows:
        p=dataset_root/r['image_path']
        assert p.is_file(),f'Missing image: {p}'
        assert sha256(p)==r['sha256'],f'Image changed since original run: {p}'
    print('Verified exact source-grouped 70/15/15 split, 300 disjoint sources and 1800 image hashes.',flush=True)
    return {s:sorted((r for r in rows if r['split']==s),key=lambda r:r['image_id']) for s in ('train','val','test')}

def load_images(split_rows,root):
    x=np.empty((len(split_rows),IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.uint8)
    y=np.empty(len(split_rows),dtype=np.float32)
    for i,r in enumerate(split_rows):
        with Image.open(root/r['image_path']) as im:
            x[i]=np.asarray(im.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR))
        y[i]=float(r['quality_label']=='bad')
    return x,y

class ArrayDataset(keras.utils.PyDataset):
    def __init__(self,imgs,labels,batch_size=BATCH,shuffle=False,seed=SEED):
        super().__init__(workers=1,use_multiprocessing=False,max_queue_size=2)
        self.x=imgs;self.y=labels;self.batch_size=batch_size;self.shuffle=shuffle;self.rng=np.random.default_rng(seed)
        self.idx=np.arange(len(imgs))
        if shuffle:self.rng.shuffle(self.idx)
    def __len__(self):return (len(self.x)+self.batch_size-1)//self.batch_size
    def __getitem__(self,i):
        ids=self.idx[i*self.batch_size:(i+1)*self.batch_size]
        return self.x[ids].astype('float32'),self.y[ids].reshape(-1,1)
    def on_epoch_end(self):
        if self.shuffle:self.rng.shuffle(self.idx)

def compute_metrics(y,p):
    pred=(np.asarray(p).ravel()>=0.5).astype(int)
    actual=np.asarray(y).astype(int).ravel()
    cm=confusion_matrix(actual,pred,labels=[0,1])
    pp,rr,ff,su=precision_recall_fscore_support(actual,pred,labels=[0,1],zero_division=0)
    return {'accuracy':float(accuracy_score(actual,pred)),
        'precision_macro':float(np.mean(pp)),'recall_macro':float(np.mean(rr)),'f1_macro':float(np.mean(ff)),
        'precision_bad':float(pp[1]),'recall_bad':float(rr[1]),'f1_bad':float(ff[1]),
        'per_class':{'good':{'precision':float(pp[0]),'recall':float(rr[0]),'f1':float(ff[0]),'support':int(su[0])},
                     'bad':{'precision':float(pp[1]),'recall':float(rr[1]),'f1':float(ff[1]),'support':int(su[1])}},
        'confusion_matrix':cm.tolist(),'test_samples':int(len(y)),'error_count':int(np.sum(actual!=pred))}

def plot_curves(history,out):
    h=history.history
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for col,title,keys in [(0,'Cross-entropy loss',('loss','val_loss')),(1,'Accuracy',('accuracy','val_accuracy'))]:
        for key in keys:
            if key in h:axs[col].plot(np.arange(1,len(h[key])+1),h[key],label=key)
        axs[col].set(title=title,xlabel='Epoch');axs[col].grid(alpha=.2);axs[col].legend()
    fig.suptitle('Task 5A — MobileNetV2 partial fine-tuning (top 30 layers)')
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)

def plot_cm(cm,out):
    fig,ax=plt.subplots(figsize=(5,4.5))
    ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['good','bad'],yticklabels=['good','bad'],xlabel='Predicted',ylabel='True',title='Fine-tuned model: held-out test')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center',color='white' if cm[i][j]>max(1,np.max(cm)/2) else 'black',fontsize=18)
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)

def benchmark(model,images,threads=4):
    import torch
    torch.set_num_threads(threads)
    pix=images[:min(32,len(images))]
    def call(k):
        x=Image.fromarray(pix[k%len(pix)]).resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR)
        x=np.asarray(x,dtype=np.float32)[None,...]
        z=model(x,training=False)
        return float(np.asarray(keras.ops.convert_to_numpy(z)).reshape(-1)[0])
    for k in range(20):call(k)
    samples=[]
    for k in range(100):
        t=time.perf_counter_ns();call(k);samples.append((time.perf_counter_ns()-t)/1e6)
    return {'warmups':20,'timed_predictions':100,'batch_size':1,'cpu_threads':threads,
      'mean_ms':float(np.mean(samples)),'median_ms':float(np.median(samples)),
      'p95_ms':float(np.percentile(samples,95)),'samples_ms':samples,
      'scope':'PIL whole-image resize, RGB float conversion, forward, host scalar output; excludes file/decode and weight loading',
      'backend':keras.backend.backend()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True)
    args=ap.parse_args();base=args.baseline;out=args.output_root/RESULT_REL;out.mkdir(parents=True,exist_ok=True)
    keras.utils.set_random_seed(SEED)
    import torch
    torch.set_num_threads(4)
    try:torch.use_deterministic_algorithms(True,warn_only=True)
    except Exception:pass
    assert sha256(base)==EXPECTED_BASELINE_SHA256,'Checkpoint SHA mismatch'
    rows=json.load(open(args.manifest,encoding='utf-8'));parts=check_manifest(rows,args.dataset_root)
    splits={}
    for split in ('train','val','test'):
        splits[split]=load_images(parts[split],args.dataset_root)
        print('loaded',split,len(splits[split][1]),flush=True)
    saved=keras.models.load_model(base,compile=False)
    backbone=saved.get_layer('mobilenetv2_1.00_224')
    assert len(backbone.layers)==154
    # Preserve original head weights. Only top 30 MobileNetV2 backbone layer objects may unfreeze.
    backbone.trainable=True
    for lay in backbone.layers:lay.trainable=False
    target=backbone.layers[-UNFREEZE_LAST:]
    for lay in target:lay.trainable=True
    # Freeze BN in those last 30 to preserve pretrained statistics and avoid drift on a small dataset.
    bn_frozen=[]
    for lay in target:
        if isinstance(lay,keras.layers.BatchNormalization):
            lay.trainable=False;bn_frozen.append(lay.name)
    assert all(not x.trainable for x in backbone.layers[:-UNFREEZE_LAST])
    assert saved.get_layer('dense').trainable and saved.get_layer('p_bad').trainable
    trainable_params=int(sum(np.prod(v.shape) for v in saved.trainable_weights));total_params=saved.count_params()
    print('TRAINABLE',trainable_params,'TOTAL',total_params,'BN kept frozen',bn_frozen,'backend',keras.backend.backend(),flush=True)
    selected_names=[x.name for x in target]
    saved.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss=keras.losses.BinaryCrossentropy(),metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
    # The 4 GiB environment cannot handle a physical Keras batch of 32.
    # Implement genuine Adam fine-tuning with four 8-image microbatches per
    # effective 32-image update; BN moving statistics remain frozen.
    chk=out/'best.keras'
    print('Training starting (memory-safe PyTorch autograd): effective batch=32 = 4x8 accumulation, max epochs=50, patience=5, lr=1e-5',flush=True)
    ts=time.perf_counter()
    train_x,train_y=splits['train']; val_x,val_y=splits['val']
    permutation_rng=np.random.default_rng(SEED)
    train_history={'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]}
    best_val=float('inf');best_epoch=0;bad_epochs=0
    optimizer=torch.optim.Adam((p for p in saved.parameters() if p.requires_grad),lr=LEARNING_RATE)
    microbatch=8
    for epoch in range(EPOCHS):
        ids=permutation_rng.permutation(len(train_y))
        train_loss_sum=0.;train_correct=0
        for start in range(0,len(ids),BATCH):
            chunk=ids[start:start+BATCH];optimizer.zero_grad(set_to_none=True)
            for j in range(0,len(chunk),microbatch):
                sample_ids=chunk[j:j+microbatch]
                xx=torch.from_numpy(train_x[sample_ids].astype('float32'))
                yy=torch.from_numpy(train_y[sample_ids].reshape(-1,1).astype('float32'))
                prediction=saved(xx,training=True)
                loss=torch.nn.functional.binary_cross_entropy(prediction,yy)
                (loss*(len(sample_ids)/len(chunk))).backward()
                train_loss_sum+=float(loss.detach())*len(sample_ids)
                train_correct+=int(((prediction.detach()>=0.5).int()==yy.int()).sum())
                del xx,yy,prediction,loss
            optimizer.step()
        tl=train_loss_sum/len(train_y);ta=train_correct/len(train_y)
        vl_sum=0.;vcorr=0
        with torch.no_grad():
            for j in range(0,len(val_y),microbatch):
                xx=torch.from_numpy(val_x[j:j+microbatch].astype('float32'))
                yy=torch.from_numpy(val_y[j:j+microbatch].reshape(-1,1).astype('float32'))
                prediction=saved(xx,training=False)
                loss=torch.nn.functional.binary_cross_entropy(prediction,yy)
                vl_sum+=float(loss.detach())*len(yy)
                vcorr+=int(((prediction.detach()>=0.5).int()==yy.int()).sum())
                del xx,yy,prediction,loss
        vl=vl_sum/len(val_y);va=vcorr/len(val_y)
        for k,v in [('loss',tl),('accuracy',ta),('val_loss',vl),('val_accuracy',va)]:train_history[k].append(v)
        if vl<best_val:
            best_val=vl;best_epoch=epoch+1;bad_epochs=0
            saved.save(chk,overwrite=True)
        else:bad_epochs+=1
        print(f'EPOCH {epoch+1:02d} loss={tl:.5f} accuracy={ta:.5f} val_loss={vl:.5f} val_accuracy={va:.5f} best={best_epoch} bad_epochs={bad_epochs}',flush=True)
        (out/'interim_history.json').write_text(json.dumps(train_history,indent=2))
        if bad_epochs>=PATIENCE:
            print('EARLY STOP at epoch',epoch+1,'best validation loss epoch',best_epoch,flush=True)
            break
    class History:
        history=train_history
    hist=History()
    duration=time.perf_counter()-ts
    print('Training finished in',duration,'sec, epochs',len(hist.history['loss']),flush=True)
    plot_curves(hist,out)
    with open(out/'history.json','w') as f:json.dump({k:[float(x) for x in v] for k,v in hist.history.items()},f,indent=2)
    # Load best weights from validation loss, then evaluate the test exactly once.
    best=keras.models.load_model(chk,compile=False)
    xs,ys=splits['test']
    with torch.no_grad():
        prob=np.concatenate([best(torch.from_numpy(xs[i:i+8].astype('float32')),training=False).detach().cpu().numpy().reshape(-1) for i in range(0,len(xs),8)])
    met=compute_metrics(ys,prob)
    plot_cm(np.asarray(met['confusion_matrix']),out)
    errdir=out/'errors';errdir.mkdir(exist_ok=True)
    errs=[]
    for i,r in enumerate(parts['test']):
        if int(prob[i]>=0.5)!=int(ys[i]):
            dst=errdir/Path(r['image_path']).name
            with Image.open(args.dataset_root/r['image_path']) as im:im.save(dst)
            errs.append({'image_id':r['image_id'],'image':str(dst.relative_to(args.output_root)),'source_id':r['source_id'],'true':int(ys[i]),'pred':int(prob[i]>=0.5),'p_bad':float(prob[i]),'defect_type':r['defect_type']})
    with open(out/'errors.json','w')as f:json.dump(errs,f,indent=2)
    perf=benchmark(best,splits['val'][0]);baseline_model=keras.models.load_model(base,compile=False)
    perf_before=benchmark(baseline_model,splits['val'][0]);
    # Historical baseline test metrics come from the original frozen-run evaluation; no test-driven selection.
    historical={'accuracy':0.9962962962962963,'precision_macro':0.9963235294117647,'recall_macro':0.9962962962962962,'f1_macro':0.9962962454903359,
                'confusion_matrix':[[135,0],[1,134]],'test_samples':270}
    size=(chk.stat().st_size/(1024*1024));old_size=base.stat().st_size/(1024*1024)
    result={'experiment':'Lab06 Task5A frozen-vs-fine-tune','week':9,'model':'mobilenetv2','regime':'finetune_top30','input_size':[224,224],'seed':SEED,'device':f'CPU (keras-{keras.backend.backend()})','epochs_planned':EPOCHS,'epochs_run':len(hist.history['loss']),'patience':PATIENCE,'batch_size':BATCH,'physical_microbatch_size':8,'gradient_accumulation_steps':4,'learning_rate':LEARNING_RATE,'unfrozen_backbone_layer_count':UNFREEZE_LAST,'selected_layers':selected_names,'batch_norm_layers_kept_frozen':bn_frozen,
        'train_time_s':duration,'latency_ms_per_image':perf['mean_ms'],'model_size_mb':size*1048576/1000000,'model_size_mib':size,'params':total_params,'trainable_params':trainable_params,'test_samples':270,'metrics':met,'confusion_matrix':met['confusion_matrix'],'history':{k:[float(v) for v in vals] for k,vals in hist.history.items()},
        'benchmark':perf,'baseline_benchmark_remeasured_on_same_CPU_backend':perf_before,
        'baseline_archived_test':historical,'baseline_archived_model_size_mib':old_size,
        'original_checkpoint_sha256':sha256(base),'finetune_checkpoint_sha256':sha256(chk),
        'manifest_sha256':sha256(args.manifest),'test_threshold':0.5,'selected_by':'validation_loss',
        'source_id_disjoint_split':True,
        'protocol_note':'Same 70/15/15 original source-grouped split and seed 2473986. Train on train only, select on val only, evaluate once on archived test.',
        'backend_note':'Keras 3 torch backend loads the original Keras checkpoint and performs genuine Adam autograd updates (TensorFlow unavailable here). To fit the 4 GiB memory limit, four physical microbatches of 8 accumulate before each optimizer step (effective batch 32); BatchNorm stays frozen.',
        'honesty_note':'This archived test was evaluated during prior Tasks 1-4, so a new independent final test would be preferable for claims of unbiased generalization.'}
    with open(out/'results.json','w')as f:json.dump(result,f,indent=2)
    with open(out/'predictions_test.json','w')as f:json.dump([{'image_id':r['image_id'],'y_true':int(ys[i]),'p_bad':float(prob[i]),'y_pred':int(prob[i]>=0.5)} for i,r in enumerate(parts['test'])],f,indent=2)
    baseline=historical
    delta={key:met[key]-baseline[key] for key in ['accuracy','precision_macro','recall_macro','f1_macro']}
    val_gap=hist.history['accuracy'][-1]-hist.history['val_accuracy'][-1]
    report=args.output_root/'reports/bonus_finetune_delta.md';report.parent.mkdir(exist_ok=True,parents=True)
    report.write_text(f'''# Task 5A — Frozen versus Top-30 Fine-Tuning Delta\n\n**Date:** 8 November 2026. **Source split:** exact archived 70/15/15 source-grouped manifest (1260/270/270); seed **{SEED}**. \n\n## What changed\n\nThe pretrained MobileNetV2 **original frozen checkpoint** (SHA256 `{sha256(base)}`) was reloaded. We unfroze the final 30 named backbone layers, retained batch-normalization layers frozen for stable statistics, and kept the custom binary head trainable. Adam learning rate **1e-5**, max **50 epochs**, early-stopping **patience 5**, **effective batch size 32** (four microbatches of 8 because of the runtime memory cap). The teacher calls 1e-5 “10 times smaller”; relative to the original head-learning rate 1e-3, it is mathematically **100 times smaller**. Our experiment follows the explicit numeric value 1e-5.\n\n## Frozen vs fine-tuned (same 270-image test set)\n\n| Metric | Original frozen (archived) | Fine-tuned top-30 | Delta |\n|---|---:|---:|---:|\n| Accuracy | {baseline['accuracy']:.4f} | {met['accuracy']:.4f} | {delta['accuracy']:+.4f} |\n| Precision macro | {baseline['precision_macro']:.4f} | {met['precision_macro']:.4f} | {delta['precision_macro']:+.4f} |\n| Recall macro | {baseline['recall_macro']:.4f} | {met['recall_macro']:.4f} | {delta['recall_macro']:+.4f} |\n| F1 macro | {baseline['f1_macro']:.4f} | {met['f1_macro']:.4f} | {delta['f1_macro']:+.4f} |\n| Incorrect test images | 1 | {met['error_count']} | {met['error_count']-1:+d} |\n\n**Training:** {len(hist.history['loss'])} epochs; {duration:.1f} s; best epoch {int(np.argmin(hist.history['val_loss'])+1)}. **Train–validation accuracy gap** at last epoch: {val_gap:+.4f}. **Baseline CPU latency remeasured:** {perf_before['mean_ms']:.2f} ms/image vs fine-tuned {perf['mean_ms']:.2f} ms/image (same warm-up, 100 timed runs, Keras torch CPU, 4 threads); they are runtime- and load-dependent. **Checkpoint sizes:** old {old_size:.2f} MiB; new {size:.2f} MiB (serialization/optimizer state may differ, even if architecture parameter count does not).\n\n## Five-line delta memo\n\n1. Fine-tuning changed held-out accuracy by **{delta['accuracy']:+.4f}** and macro F1 by **{delta['f1_macro']:+.4f}**, compared with the saved frozen baseline.\n2. The modified classifier produced **{met['error_count']}** test mistakes versus **1** for the baseline, so this is the measured change in false accepts/rejects.\n3. The final train–validation accuracy gap is **{val_gap:+.4f}**. The validation-loss trajectory and gap must both be inspected before declaring overfitting.\n4. End-to-end CPU latency was remeasured under the same backend: frozen **{perf_before['mean_ms']:.2f} ms**, fine-tuned **{perf['mean_ms']:.2f} ms**; checkpoint size changed from **{old_size:.2f}** to **{size:.2f} MiB**, possibly due to serialization metadata.\n5. **Deployment verdict:** {'Prefer keeping the frozen baseline; the new score does not beat its archived F1.' if delta['f1_macro']<=0 else 'Fine-tuned model improves the small synthetic test score, but further independent real-world validation is required before deployment.'}\n\n## Limitations\n\nOnly 300 independent source documents underlie 1,800 synthetic variants. The same held-out test set was analyzed in earlier tasks; final unbiased evaluation on fresh unseen source documents is preferable. The original baseline was trained on TensorFlow/T4; fine-tuning here used Keras with the compatible PyTorch CPU backend, with preserved source weights. The new training uses physical microbatches of 8 with gradient accumulation to match an effective 32-image update; this differs in execution memory layout from one physical 32-image batch. The original baseline test metrics are archived, while CPU latencies are rebenchmarked on the same host/backend for comparability. Training stopped according to validation loss and **test data was not used for hyperparameter selection**.\n''',encoding='utf-8')
    print('FINAL RESULT',json.dumps({'accuracy':met['accuracy'],'macro_f1':met['f1_macro'],'test_errors':len(errs),'epochs':len(hist.history['loss']),'checkpoint':str(chk),'report':str(report)}),flush=True)

if __name__=='__main__':main()
