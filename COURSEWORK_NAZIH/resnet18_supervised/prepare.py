import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
rows=[]
for folder in ['motor_comparison','motor_classification','motor_final']:
 name='test_manifest.json' if folder=='motor_final' else 'manifest.json'
 for r in json.loads((B/folder/name).read_text()):
  p=B/folder/'data'/r['img_path']
  if r['illumination']=='01' and p.exists():
   rows.append({'path':str(p.relative_to(B)),'object':r['object_name'],'group':r['object_name'].split('_')[-1],'label':int(r['image_anomaly']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(rows)==99 and len({r['path'] for r in rows})==99
ids=sorted({r['group'] for r in rows},key=lambda x:hashlib.sha256(('resnet18-pilot-v1:'+x).encode()).hexdigest())
val=set(ids[:15])
for r in rows:r['split']='validation' if r['group'] in val else 'train'
assert all({r['label'] for r in rows if r['split']==s}=={0,1} for s in ['train','validation'])
(R/'manifest.json').write_text(json.dumps(rows,indent=2))
cfg={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':'Full torchvision ResNet18, ImageNet1K V1 initialization, fc replaced with 512 -> 2','classes':['Normal','Defective'],'regime':'Supervised fine-tuning, all parameters trainable','input_size':448,'preprocessing':'RGB, full-frame bicubic resize, ImageNet normalization, no augmentation for this engineering pilot','epochs':3,'batch_size':4,'optimizer':'AdamW','learning_rate':0.0001,'weight_decay':0.0001,'loss':'Unweighted cross entropy','seed':20261028,'threads':2,'decision':'argmax of two logits; ties class 0','checkpoint_selection':'Highest validation macro-F1, earliest epoch on ties','status':'EXPLORATORY ENGINEERING PILOT; old test data reused openly; no independent test','grouping':'Conservative numeric suffix grouping, not verified physical identity','split_groups':{'train':len(ids)-15,'validation':15},'counts':{s:{str(c):sum(r['split']==s and r['label']==c for r in rows) for c in [0,1]} for s in ['train','validation']},'paper_addition':'Not yet chosen or implemented; baseline only','future_confirmation':'Requires independently held-out data and resolved provenance. No comparison to old anomaly scores as if same training regime.'}
(R/'config.json').write_text(json.dumps(cfg,indent=2));print(cfg['counts'])
