"""Recalculate real error examples from archived, previously seen Study A test predictions.

Does not access images, run inference, change thresholds or select models.
Outputs are descriptive only; do not use to tune training or claim blind testing.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path


def load(z: zipfile.ZipFile, size: int, loss: str, seed: int):
    name = f'evaluations/eval_size{size}_seed{seed}_{loss}.json'
    return json.loads(z.read(name).decode('utf-8'))


def make(results_zip: Path, output: Path, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    with zipfile.ZipFile(results_zip) as z:
        assert z.testzip() is None
        for size in (140, 280, 420):
            ce = load(z, size, 'ce', seed)
            poly = load(z, size, 'poly1', seed)
            A = {x['path']: x for x in ce['predictions']}
            B = {x['path']: x for x in poly['predictions']}
            assert len(A) == len(B) == 1004 and A.keys() == B.keys()
            assert all(A[k]['label'] == B[k]['label'] and A[k]['sha256'] == B[k]['sha256'] for k in A)
            ce_bad = {k for k,r in A.items() if r['prediction'] != r['label']}
            poly_bad = {k for k,r in B.items() if r['prediction'] != r['label']}
            cm_a, cm_b = ce['metrics']['confusion_matrix'], poly['metrics']['confusion_matrix']
            assert len(ce_bad) == cm_a[0][1] + cm_a[1][0]
            assert len(poly_bad) == cm_b[0][1] + cm_b[1][0]
            assert all(A[k]['prediction'] == int(A[k]['defect_score'] > 0) and B[k]['prediction'] == int(B[k]['defect_score'] > 0) for k in A)
            summary.append(dict(size=size, ce_FPs=cm_a[0][1], ce_FNs=cm_a[1][0],
                                poly1_FPs=cm_b[0][1], poly1_FNs=cm_b[1][0],
                                both_wrong=len(ce_bad & poly_bad), ce_only=len(ce_bad-poly_bad),
                                poly1_only=len(poly_bad-ce_bad)))
            if size != 420:
                continue  # Detailed case study is frozen at size=420, seed=20261028.
            groups = {
                'Both wrong': ce_bad & poly_bad,
                'CE wrong / Poly-1 correct': ce_bad - poly_bad,
                'CE correct / Poly-1 wrong': poly_bad - ce_bad,
            }
            for group, keys in groups.items():
                for label, name in ((0, 'False Positive'), (1, 'False Negative')):
                    candidates = [k for k in keys if A[k]['label'] == label]
                    # Select the smallest absolute incorrect margin first (boundary case),
                    # then the largest-magnitude incorrect margin (strong error).
                    def error_margin(k):
                        incorrect = A[k] if A[k]['prediction'] != label else B[k]
                        return abs(incorrect['defect_score'])
                    order = sorted(candidates, key=lambda k:(error_margin(k), k))
                    selected = order[:2] + (order[-1:] if len(order)>2 else [])
                    for path in dict.fromkeys(selected):
                        a,b = A[path],B[path]
                        rows.append(dict(image=path, image_sha256=a['sha256'], actual_label=name,
                                         group=group, ce_prediction='Defective' if a['prediction'] else 'Normal',
                                         ce_logit_margin=round(a['defect_score'],6),
                                         poly1_prediction='Defective' if b['prediction'] else 'Normal',
                                         poly1_logit_margin=round(b['defect_score'],6)))
            assert rows
    md = [
        '# Verified classification failure examples — exploratory evaluation',
        '',
        f'**Scope:** Study A, archived official TEST (1,004 images), seed `{seed}`; original test set had already been inspected. These are **not** full-epoch Study B validation predictions.',
        '',
        'All examples below are read directly from the authentic archived prediction arrays. No model retraining, rescaling, threshold adjustment or post-selection occurred.',
        '',
        '## Error totals across paired runs',
        '',
        '| Train images | CE FP | CE FN | Poly-1 FP | Poly-1 FN | Both wrong | CE only | Poly-1 only |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summary:
        md.append(f"| {r['size']} | {r['ce_FPs']} | {r['ce_FNs']} | {r['poly1_FPs']} | {r['poly1_FNs']} | {r['both_wrong']} | {r['ce_only']} | {r['poly1_only']} |")
    md.extend(['','## Selected original image-level failure examples (size=420)', '',
               '| Image in official ZIP | Ground truth | CE decision | CE margin | Poly-1 decision | Poly-1 margin | Case |',
               '|---|---|---|---:|---|---:|---|'])
    for r in rows:
        md.append(f"| `{r['image']}` | {('Normal' if r['actual_label']=='False Positive' else 'Defective')} | {r['ce_prediction']} | {r['ce_logit_margin']:+.3f} | {r['poly1_prediction']} | {r['poly1_logit_margin']:+.3f} | {r['group']} — {r['actual_label']} |")
    md.extend(['', '## Interpretation', '',
               '- **False positive (FP):** a genuinely normal image classified as defective. At 420 images, CE produces 23 and Poly-1 produces 17 such errors.',
               '- **False negative (FN):** a genuinely defective image classified as normal. At 420 images, CE produces 26 and Poly-1 produces 28 such errors. Missed defects matter in industrial inspection.',
               '- **Trade-off:** in this one paired run Poly-1 rejects 6 fewer normal images but misses 2 more defective images. This is an observed trade-off, not evidence of a universal advantage.',
               '- **Defect logit margin** equals defective-class logit minus normal-class logit. Positive predicts defective; negative predicts normal. Its magnitude is not a calibrated probability.',
               '- **Distinct failures:** some images are wrong for only one loss. The models do not simply make the same errors, so aggregate Macro-F1 alone hides useful differences.',
               '', '## Evidence boundaries', '',
               '- The archive contains predictions, labels and exact image hashes, **not the raw test image pixels**. No visual texture, scratch type, lighting cause, or defect morphology was assessed here.',
               '- To create a photographic gallery, obtain the official `KolektorSDD2.zip`, check its fixed SHA256 and open the individual `test/*.png` files. A separate command in `scripts/error_gallery.py` generates that gallery reproducibly without changing any model.',
               '- These test images were evaluated previously. This is an **exploratory post-hoc error audit**, not a new independent blind-test assessment or a basis for hyperparameter tuning.',
               '- Physical-specimen grouping and near-duplicate independence were not verified by exact image hashes alone.',
               '', 'Dataset: Božič et al., KolektorSDD2, ViCoS (CC BY-NC-SA 4.0).', ''])
    (output/'ERROR_CASES.md').write_text('\n'.join(md),encoding='utf-8')
    with (output/'ERROR_CASES.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'PASS: {len(rows)} verified individual error examples; {len(summary)} controlled size comparisons')
    print('Study A 420 CM:',summary[-1]);print('No official test reevaluation or model selection performed.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results-zip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261028)
    a=p.parse_args();make(a.results_zip,a.output,a.seed)
