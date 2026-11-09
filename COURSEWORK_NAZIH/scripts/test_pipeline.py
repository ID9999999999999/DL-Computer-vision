"""Pre-training unit and integrity checks. No model training, test inference or external files needed."""
import collections,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config,verify_source70,verify_old_baseline,nested_size_schedule,verify_poly1

def main():
    cfg=config();base=verify_source70();ref=verify_old_baseline();assert len(ref)==6
    assert verify_poly1()
    classes={0:[r for r in base if r['label']==0],1:[r for r in base if r['label']==1]}
    # Synthetic metadata only: stand-ins to exercise nested size scheduling before the original archive is restored.
    pool=copy.deepcopy(base)
    for label in (0,1):
        for n in range(175):pool.append({'zip_path':f'train/TEST_{label}_{n}.png','label':label,
                                          'sha256':f'test_metadata_{label}_{n}','from_original70':False})
    for size in cfg['new_training_sizes']:
        for seed in cfg['seeds']:
            schedule=nested_size_schedule({str(size):pool[:70]+[r for r in pool[70:] if r['label']==0][:size//2-35]
                                                       +[r for r in pool[70:] if r['label']==1][:size//2-35]},size,seed)
            sub=pool[:70]+[r for r in pool[70:] if r['label']==0][:size//2-35]+[r for r in pool[70:] if r['label']==1][:size//2-35]
            assert len(schedule)==12
            for epoch in schedule:
                assert len(epoch)==70 and len(set(epoch))==70
                assert collections.Counter(sub[i]['label'] for i in epoch)=={0:35,1:35}
            seen=set(i for e in schedule for i in e)
            assert len(seen)==size, f'All {size} sources must appear over 12 epochs'
    print('PASS: 70 original image SHA256; 6 historical 1004-image evaluations; Poly-1 gradcheck; schedules 140/280/420 x3 seeds (NO TRAINING)')
if __name__=='__main__':main()