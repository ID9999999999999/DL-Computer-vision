"""Generate Task 4 comparison solely from the four documented results.json records.

This script does not rerun training, change existing results, or fabricate any metric.
Usage:
  python scripts/collect_results.py --results results --out submission/comparison_from_json.csv
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

MODEL_DIRS=('mobilenetv2','yolov8n','yolov8s','efficientnetb0')
FIELDS=('model','regime','accuracy','precision_macro','recall_macro','f1_macro',
        'latency_ms_per_image','model_size_mb','params','train_time_s',
        'epochs_planned','epochs_run','batch_size','input_size','test_samples')

def collect(root:Path):
    rows=[]
    for dirname in MODEL_DIRS:
        file=root/dirname/'results.json'
        if not file.is_file():raise FileNotFoundError(f'Missing source JSON: {file}')
        obj=json.loads(file.read_text(encoding='utf-8'))
        metrics=obj['metrics']
        if not 0<=metrics['accuracy']<=1:raise ValueError(f'Accuracy not a fraction: {file}')
        row={
          'model':obj['model'],'regime':obj['regime'],
          'accuracy':metrics['accuracy'],
          'precision_macro':metrics['precision_macro'],
          'recall_macro':metrics['recall_macro'],
          'f1_macro':metrics['f1_macro'],
          'latency_ms_per_image':obj['latency_ms_per_image'],
          'model_size_mb':obj['model_size_mb'],'params':obj['params'],
          'train_time_s':obj['train_time_s'],
          'epochs_planned':obj['epochs_planned'],'epochs_run':obj['epochs_run'],
          'batch_size':obj.get('batch_size'),
          'input_size':'x'.join(map(str,obj['input_size'])),'test_samples':obj['test_samples']}
        if len(obj['confusion_matrix'])!=2 or len(obj['confusion_matrix'][0])!=2:
            raise ValueError(f'Unexpected confusion matrix {file}')
        if sum(sum(r) for r in obj['confusion_matrix'])!=obj['test_samples']:
            raise ValueError(f'Confusion matrix count mismatch: {file}')
        rows.append(row)
    return rows

def write(rows,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path('results'))
    ap.add_argument('--out',type=Path,default=Path('submission/comparison_from_json.csv'))
    args=ap.parse_args(argv)
    rows=collect(args.results);write(rows,args.out)
    print('Wrote',args.out,'from exactly',len(rows),'original-derived results.json files')

if __name__=='__main__':main()