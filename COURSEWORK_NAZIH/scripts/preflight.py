"""Offline initial integrity audit. Runs without original KSDD2 ZIP and without GPU."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config,verify_source70,verify_old_baseline,sha_file

def main():
    cfg=config();rows=verify_source70();ref=verify_old_baseline()
    test=json.loads((ROOT/'input/test_manifest.json').read_text())
    assert not set(r['sha256'] for r in rows)&set(r['sha256'] for r in test)
    assert cfg['sizes']==[70,140,280,420] and cfg['rounds']==12
    from collections import defaultdict
    group=defaultdict(list)
    for r in ref:group[r['loss']].append(r['macro_f1'])
    report={
      'source_train_images_verified':len(rows),
      'historical_baseline_12epoch_evaluations_verified':len(ref),
      'test_images_metadata_verified':len(test),
      'baseline_macro_f1_mean':{arm:sum(a)/len(a) for arm,a in group.items()},
      'new_training_completed':False,
      'requires_original_zip_for_new_sizes':True,
      'strict_device':'cpu',
      'interpretation':'Exploratory only; 1004 test examples already reused.'
    }
    (ROOT/'results/preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()