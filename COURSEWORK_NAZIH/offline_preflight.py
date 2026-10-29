"""Repeatable offline split integrity audit; no claims of physical independence."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'resnet18_supervised/manifest.json').read_text())
assert len(manifest)==99
assert set(r['split'] for r in manifest)=={'train','validation'}
assert set(r['label'] for r in manifest)=={0,1}
assert len({r['path'] for r in manifest})==len(manifest),'Duplicate path'
assert len({r['sha256'] for r in manifest})==len(manifest),'Exact duplicate image bytes'
tr=[r for r in manifest if r['split']=='train']; va=[r for r in manifest if r['split']=='validation']
assert len(tr)==70 and len(va)==29
assert set(r['group'] for r in tr).isdisjoint({r['group'] for r in va}),'Numeric group leakage'
assert {r['path'] for r in tr}.isdisjoint({r['path'] for r in va}),'Path leakage'
counts={name:dict(Counter(str(r['label']) for r in subset)) for name,subset in [('train',tr),('validation',va)]}
verified=[];missing=[]
for r in manifest:
  p=root/r['path']
  if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']:verified.append(r['path'])
  else: missing.append(r['path'])
assert counts=={'train':{'0':35,'1':35},'validation':{'0':15,'1':14}},counts
result={'check_status':'PASS — manifest and available bytes only','images_total':len(manifest),'verified_local_images':len(verified),'missing_images':len(missing),'per_split_label_counts':counts,'exact_duplicate_paths_or_hashes':False,'numeric_group_overlap':False,'physical_specimen_independence':'NOT VERIFIED','near_duplicate_visual_check':'NOT RUN: missing 95 source images','independent_holdout':'NOT AVAILABLE'}
print(json.dumps(result,indent=2))
(root/'offline_preflight_report.json').write_text(json.dumps(result,indent=2))
