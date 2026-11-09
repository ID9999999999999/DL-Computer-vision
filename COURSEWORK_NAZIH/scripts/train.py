"""New sample-size arms only. The historical 70-image, 12-epoch runs are never retrained.
Every arm uses exactly 70 training-image exposures per epoch (216 optimizer updates).
CPU and package versions must match the archived duration experiment; otherwise stop.
"""
from pathlib import Path
import argparse,hashlib,json,os,random,sys,time,zipfile
from functools import lru_cache
import numpy as np
from PIL import Image
import torch,torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config,read_json,sha_file,digest_state,verify_archive,verify_poly1,verify_source70,nested_size_schedule,old_initial_reference,poly1

IM_MEAN=torch.tensor([.485,.456,.406])[:,None,None]
IM_STD=torch.tensor([.229,.224,.225])[:,None,None]

def make_loader(archive,pool):
    name_to_row={r['zip_path']:r for r in pool}
    z=zipfile.ZipFile(archive,'r')
    @lru_cache(maxsize=100)
    def load(name):
        row=name_to_row[name]
        if row['from_original70']:
            data=(ROOT/'input/train'/Path(name).name).read_bytes()
        else:data=z.read(name)
        assert hashlib.sha256(data).hexdigest()==row['sha256'],f'Image hash mismatch: {name}'
        import io
        with Image.open(io.BytesIO(data)) as im:
            arr=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC),dtype=np.uint8).copy()
        a=torch.from_numpy(arr).permute(2,0,1).float().div_(255)
        return (a-IM_MEAN)/IM_STD
    return load,z

def test_environment(strict=True):
    cfg=config();want=cfg['strict_reference_environment']
    got={'torch':torch.__version__,'torchvision':torchvision.__version__,
         'numpy':np.__version__,'pillow':Image.__version__}
    if strict and got!=want:raise RuntimeError('ENVIRONMENT DRIFT. Archived 70-image experiment used '+str(want)+'; current '+str(got)+'. Stop to avoid a confounded sample-size comparison.')
    assert not torch.cuda.is_initialized(), 'GPU is not part of the historical reference'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    verify_poly1()
    return got

def run(size,seed,arm,archive):
    cfg=config();assert size in cfg['new_training_sizes'] and seed in cfg['seeds'] and arm in cfg['arms']
    test_environment()
    reference=old_initial_reference(seed)
    mlist=read_json(ROOT/'results/size_manifests.json');pool=mlist[str(size)]
    assert len(pool)==size
    schedules=nested_size_schedule(mlist,size,seed)
    load,z=make_loader(archive,pool)
    checkpoint=ROOT/f'results/size{size}_seed{seed}_{arm}.pt'
    result_path=ROOT/f'results/size{size}_seed{seed}_{arm}.json'
    if checkpoint.exists() or result_path.exists():raise FileExistsError('Never overwrite a completed or partial run: '+str(result_path))
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    source_pretrained=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    assert sha_file(source_pretrained)==cfg['pretrained_sha256'],'Pretrained ImageNet weights changed'
    m.fc=nn.Linear(512,2)
    initial_hash=digest_state(m.state_dict())
    orders=[torch.randperm(70).tolist() for _ in range(cfg['rounds'])]
    order_hash=hashlib.sha256(json.dumps(orders).encode()).hexdigest()
    if initial_hash!=reference['paired_hashes']['initial_state'] or order_hash!=reference['all_orders_hash']:
        raise RuntimeError('Reference initial state / 12-epoch batch order diverged. Do not compare new results with old 70.')
    optimizer=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[];start=time.monotonic()
    for ep,canonical_order in enumerate(orders):
        mapped=schedules[ep]
        assert len(mapped)==70 and len(set(mapped))==70
        m.train();objective=0
        for offset in range(0,70,4):
            pool_ids=[mapped[j] for j in canonical_order[offset:offset+4]]
            batch=torch.stack([load(pool[i]['zip_path']) for i in pool_ids])
            labels=torch.tensor([pool[i]['label'] for i in pool_ids],dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits=m(batch)
            loss=F.cross_entropy(logits,labels) if arm=='ce' else poly1(logits,labels,epsilon=1)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            optimizer.step()
            objective+=float(loss.detach())*len(pool_ids)
        history.append({'epoch':ep+1,'train_objective':objective/70})
        print(f'new size={size} seed={seed} {arm} epoch={ep+1} train_loss={objective/70:.5f}',flush=True)
    # BN recalibration uses the exact SAME 70 original training examples in the SAME manifest order as the old reference.
    # This avoids introducing a second intervention (calibration set composition/size).
    m.eval();before=digest_state(dict(m.named_parameters()))
    for mod in m.modules():
        if isinstance(mod,torch.nn.BatchNorm2d):mod.reset_running_stats();mod.momentum=None;mod.train()
    canonical=verify_source70()
    with torch.no_grad():
        for i in range(0,70,4):
            batch=torch.stack([load(r['zip_path']) for r in canonical[i:i+4]])
            m(batch)
    assert before==digest_state(dict(m.named_parameters()))
    m.eval()
    torch.save(m.state_dict(),checkpoint)
    result={'size':size,'seed':seed,'loss':arm,'epochs':12,'reference70_unchanged':True,
            'initial_state_sha256':initial_hash,'original_epoch_orders_sha256':order_hash,
            'source_archive_sha256':cfg['target_archive_sha256'],
            'training_manifest_sha256':sha_file(ROOT/'results/size_manifests.json'),
            'train_history':history,'checkpoint_sha256':sha_file(checkpoint),
            'calibration_images':'original70 in original order, independent of size',
            'training_unique_images':size,'total_optimizer_steps':216,
            'reference_equivalent_rounds':12,'effective_full_data_passes':840/size,
            'runtime_seconds':time.monotonic()-start,
            'status':'trained; external evaluation prohibited until all 18 models frozen'}
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    z.close();print('COMPLETE:',result_path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',required=True,type=int);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--loss',required=True,choices=['ce','poly1']);p.add_argument('--archive',required=True)
    a=p.parse_args()
    with verify_archive(a.archive) as z:pass
    run(a.size,a.seed,a.loss,a.archive)