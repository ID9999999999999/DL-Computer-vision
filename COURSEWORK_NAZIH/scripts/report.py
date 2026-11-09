"""Only writes results that truly exist. Never substitutes fabricated metrics."""
from pathlib import Path
import collections,csv,json,statistics,sys
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config,read_json,verify_old_baseline

KEYS=['macro_f1','defect_recall','normal_fpr','balanced_accuracy','auroc','average_precision','accuracy']

def normalize(m):
    return {k:float(m['defect_recall' if k=='defect_recall' and 'defect_recall' in m else k]
                    if k!='defect_recall' or 'defect_recall' in m else m['defective_recall']) for k in KEYS}

def main():
    cfg=config();old=verify_old_baseline();allrows=[]
    for r in old:
        row={'size':70,'seed':r['seed'],'loss':r['loss'],'kind':'historical_reference','checkpoint_sha256':r['checkpoint_sha256']}
        row.update(normalize(r['metrics']));allrows.append(row)
    for size in cfg['new_training_sizes']:
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                f=ROOT/f'results/eval_size{size}_seed{seed}_{arm}.json'
                if not f.exists():continue
                r=read_json(f);assert r['size']==size and r['seed']==seed and r['loss']==arm
                row={'size':size,'seed':seed,'loss':arm,'kind':'exploratory_new','checkpoint_sha256':r['checkpoint_sha256']}
                row.update(normalize(r['metrics']));allrows.append(row)
    path=ROOT/'results/observed_results.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['size','seed','loss','kind','checkpoint_sha256']+KEYS)
        w.writeheader();w.writerows(allrows)
    count=collections.Counter((r['size'],r['loss']) for r in allrows)
    lines=['# Training-set size — report (exploratory)',
           '', 'YASSER IDBOUZKRI | иикб-24 | 8 November 2026',
           '', '## Design',
           'The original 70-image, 12-epoch results are preserved exactly and are NOT retrained.',
           'New conditions use 140/280/420 balanced official KSDD2 training images, always 35 normal and 35 defective image exposures per epoch, 12 reference-equivalent 70-image rounds (6/3/2 full-dataset passes for sizes 140/280/420 respectively), 216 AdamW optimizer steps, original preprocessing, three paired seeds and both losses.',
           'BatchNorm recalibration uses the same original 70 training images for every size. The input samples, not the computational training budget or BN calibration dataset, are the intended intervention. Effective full-data passes change with N; numerical environment effects cannot be ruled out without rerunning the old baseline.',
           '', '**Evaluation caveat:** the 1,004 official test images have already been studied in earlier diagnosis. These are exploratory results, NOT an independent confirmatory test.',
           '**Specimen caveat:** official train/test isolation and exact-hash checks do not certify physical-part independence.',
           '', '## Actual 12-epoch evaluations (mean across 3 seeds)',
           '', '| Unique training images | Loss | Macro-F1 | Defective recall | Normal FPR | AUROC | Evidence |',
           '|---:|---|---:|---:|---:|---:|---|']
    for size in cfg['sizes']:
        for arm in cfg['arms']:
            sub=[r for r in allrows if r['size']==size and r['loss']==arm]
            if len(sub)==3:
                avg=lambda k:100*statistics.mean(r[k] for r in sub)
                kind='Archived 12-epoch baseline; not rerun' if size==70 else 'Newly evaluated; repeated test'
                lines.append(f'| {size} | {arm} | {avg("macro_f1"):.2f}% | {avg("defect_recall"):.2f}% | {avg("normal_fpr"):.2f}% | {avg("auroc"):.2f}% | {kind} |')
            else:
                lines.append(f'| {size} | {arm} | — | — | — | — | PENDING ({len(sub)}/3 seed evaluations) |')
    lines+=['','## Required before a strong causal claim',
            '- All predeclared training sizes complete for each paired seed and both losses.',
            '- Independently verify the absence of physical-part overlap; no proof presently available.',
            '- A new untouched confirmation set is necessary for confirmatory performance statements.',
            '- Do not choose the best size or hyperparameters using this previously inspected test set.',
            '', f'**Valid evaluated runs: {len(allrows)}/24.** Six of them are archived, and 18 are new runs requiring actual training and inference.',
            'No results are written for pending runs.']
    (ROOT/'results/REPORT_EN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('REPORT:',ROOT/'results/REPORT_EN.md','real evaluations',len(allrows),'of 24')
if __name__=='__main__':main()