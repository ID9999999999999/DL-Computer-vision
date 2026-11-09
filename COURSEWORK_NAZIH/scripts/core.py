"""Shared, read-only guards for Nazih's training-set-size study."""
import hashlib, io, json, random, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))
def digest_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_file(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(chunk),b''):h.update(part)
    return h.hexdigest()
def digest_state(state):
    h=hashlib.sha256()
    for name,tensor in state.items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def poly1(logits,labels,epsilon=1.0):
    import torch.nn.functional as F
    pt=logits.softmax(1).gather(1,labels[:,None]).squeeze(1)
    return (F.cross_entropy(logits,labels,reduction='none')+epsilon*(1-pt)).mean()

def verify_poly1():
    import torch
    import torch.nn.functional as F
    logits=torch.tensor([[1.2,-.7],[-1.4,.8],[.3,.1]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1,1]);assert torch.equal(poly1(logits,y,0),F.cross_entropy(logits,y))
    assert torch.autograd.gradcheck(lambda x:poly1(x,y,1),(logits,))
    p=logits.softmax(1);pt=p.gather(1,y[:,None]);expected=(1+pt)*(p-F.one_hot(y,2))/len(y)
    assert torch.allclose(torch.autograd.grad(poly1(logits,y),logits)[0],expected)
    return True

def config():return read_json(ROOT/'protocol.json')

def verify_source70():
    rows=read_json(ROOT/'input/target_manifest.json')
    assert len(rows)==70 and [sum(r['label']==label for r in rows) for label in (0,1)]==[35,35]
    seen=set()
    for r in rows:
        p=ROOT/'input/train'/Path(r['zip_path']).name
        assert p.is_file() and sha_file(p)==r['sha256'],f'Bad 70-image file: {p}'
        assert r['sha256'] not in seen
        seen.add(r['sha256'])
    return rows

def verify_old_baseline():
    import collections
    cfg=config();test=read_json(ROOT/'input/test_manifest.json');assert len(test)==1004
    assert collections.Counter(r['label'] for r in test)=={0:894,1:110}
    reference=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            t=read_json(ROOT/f'input/baseline_train/target_train_{seed}_{arm}.json')
            assert t['seed']==seed and t['arm']==arm and len(t['epochs'])==12
            e=read_json(ROOT/f'input/baseline_eval/eval_{seed}_{arm}_epoch12.json')
            assert e['seed']==seed and e['arm']==arm and e['epoch']==12
            assert len(e['predictions'])==1004
            assert e['checkpoint_sha256']==t['checkpoint_sha256']
            assert [p['sha256'] for p in e['predictions']]==[r['sha256'] for r in test]
            assert [p['label'] for p in e['predictions']]==[r['label'] for r in test]
            y=[p['label'] for p in e['predictions']]
            p=[p['prediction'] for p in e['predictions']]
            cm=[[sum(a==i and b==j for a,b in zip(y,p)) for j in (0,1)] for i in (0,1)]
            macro=sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in (0,1))/2
            assert abs(macro-e['metrics']['macro_f1'])<1e-12
            reference.append({'size':70,'seed':seed,'loss':arm,'macro_f1':macro,'metrics':e['metrics'],'checkpoint_sha256':e['checkpoint_sha256'],'status':'historical, never retrained'})
    return reference

def verify_archive(path):
    cfg=config();path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Missing original KSDD2 ZIP: {path}')
    sha=sha_file(path)
    if sha!=cfg['target_archive_sha256']:raise ValueError(f'Original ZIP SHA256 mismatch: {sha}')
    return zipfile.ZipFile(path)

def nested_size_schedule(manifest,size,seed):
    """N>70: balanced 70 image-exposures/epoch, no-replacement class cycle until exhausted."""
    cfg=config();assert size in cfg['new_training_sizes']
    pool=manifest[str(size)];assert len(pool)==size
    base=read_json(ROOT/'input/target_manifest.json');assert len(base)==70
    positions={c:[i for i,r in enumerate(base) if r['label']==c] for c in (0,1)}
    schedule=[];sequences={}
    for label in (0,1):
        ids=[i for i,r in enumerate(pool) if r['label']==label];assert len(ids)==size//2
        rng=random.Random(20261108+seed*3+label*23+size)
        sequence=[]
        while len(sequence)<35*12:
            new=list(ids);rng.shuffle(new);sequence.extend(new)
        sequences[label]=sequence[:35*12]
    for epoch in range(12):
        mapping=[None]*70
        for label in (0,1):
            chosen=sequences[label][epoch*35:(epoch+1)*35]
            for slot,pool_idx in zip(positions[label],chosen):mapping[slot]=pool_idx
        assert all(x is not None for x in mapping)
        assert len({x for x in mapping})==70  # no duplicates within any epoch for n>=140
        schedule.append(mapping)
    return schedule

def old_initial_reference(seed):
    return read_json(ROOT/f'input/baseline_train/target_train_{seed}_ce.json')