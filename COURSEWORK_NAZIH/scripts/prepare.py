"""Load KSDD2 official archive, preserve previous 70, add nested 140/280/420 train pools.
Never inspect official test images/labels here.
"""
from pathlib import Path
import collections,hashlib,io,json,random,re,sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config,read_json,verify_source70,verify_archive,digest_bytes,nested_size_schedule

def prepare(archive):
    cfg=config();base=verify_source70();base_by_path={r['zip_path']:r for r in base}
    with verify_archive(archive) as z:
        names=sorted(n for n in z.namelist() if re.fullmatch(r'train/[0-9]+\.png',n))
        assert len(names)==2331
        extras={0:[],1:[]};labels=collections.Counter();base_count=0
        for i,name in enumerate(names,1):
            raw=z.read(name);sha=digest_bytes(raw)
            with Image.open(io.BytesIO(z.read(name[:-4]+'_GT.png'))) as im:label=int(np.array(im).max()>0)
            labels[label]+=1
            if name in base_by_path:
                old=base_by_path[name]
                assert old['sha256']==sha and old['label']==label
                base_count+=1
            else:
                extras[label].append({'zip_path':name,'sha256':sha,'label':label,'from_original70':False})
        assert labels=={0:2085,1:246},labels
        assert base_count==70
    base_pool=[{'zip_path':r['zip_path'],'sha256':r['sha256'],'label':r['label'],'from_original70':True} for r in base]
    rng=random.Random(cfg['extra_selection_seed'])
    for label in (0,1):rng.shuffle(extras[label])
    manifests={'70':base_pool}
    for size in cfg['new_training_sizes']:
        per_class=size//2-35
        pool=base_pool + extras[0][:per_class]+extras[1][:per_class]
        assert len(pool)==size and collections.Counter(r['label'] for r in pool)=={0:size//2,1:size//2}
        assert len(set(r['sha256'] for r in pool))==size
        manifests[str(size)]=pool
    # Ensure true nesting and no previously used official test images.
    test_hash={r['sha256'] for r in read_json(ROOT/'input/test_manifest.json')}
    for small,big in zip(cfg['sizes'],cfg['sizes'][1:]):
        smaller={r['sha256'] for r in manifests[str(small)]}
        larger={r['sha256'] for r in manifests[str(big)]}
        assert smaller.issubset(larger) and not larger&test_hash
    target=ROOT/'results/size_manifests.json'
    if target.exists():
        assert read_json(target)==manifests,'Manifest exists but changed. Never silently alter training population.'
    else:target.write_text(json.dumps(manifests,indent=2)+'\n')
    audit={'original_zip_sha256':cfg['target_archive_sha256'],'original70_preserved':True,'sizes':cfg['sizes'],
           'nested':True,'label_counts':{str(n):{'normal':n//2,'defective':n//2} for n in cfg['sizes']},
           'official_train_test_exact_hash_overlap':0,'test_read_during_selection':False}
    (ROOT/'results/data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('PREPARED original70 + nested 140/280/420, no test images opened');return manifests

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);a=p.parse_args()
    prepare(a.archive)