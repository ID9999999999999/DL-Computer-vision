"""Synthetic fixture tests for newly added submission utilities; no training performed."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod);return mod
tr=load(BASE/'scripts/train_yolo_quality.py','train_yolo_quality')
co=load(BASE/'scripts/collect_results.py','collect_results')

class Tests(unittest.TestCase):
 def test_split_verification(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for split in ('train','val','test'):
    for label in ('good','bad'):
     p=root/split/label;p.mkdir(parents=True);(p/f'{split}_{label}.png').write_bytes(b'fixture')
   self.assertEqual(sum(tr.inspect_split(root).values()),6)
   rows=[{'source_id':f'{split}-doc','split':split} for split in ('train','val','test') for _ in (0,1)]
   manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
   self.assertEqual(sum(tr.inspect_split(root,manifest).values()),6)
 def test_source_leakage(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for split in ('train','val','test'):
    for label in ('good','bad'):
     p=root/split/label;p.mkdir(parents=True);(p/f'{split}_{label}.png').write_bytes(b'fixture')
   rows=[{'source_id':'same','split':'train'},{'source_id':'same','split':'val'},
         {'source_id':'testdoc','split':'test'}]+[{'source_id':'other','split':'train'} for _ in range(3)]
   manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
   with self.assertRaises(ValueError):tr.inspect_split(root,manifest)
 def test_results_generator_only_uses_json(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for name in co.MODEL_DIRS:
    p=root/name;p.mkdir()
    doc=dict(model=name,regime='fixture',input_size=[224,224],metrics=dict(accuracy=0.75,
     precision_macro=.75,recall_macro=.75,f1_macro=.75),latency_ms_per_image=12,
     model_size_mb=3,params=25,train_time_s=4,epochs_planned=50,epochs_run=7,
     batch_size=32,test_samples=4,confusion_matrix=[[2,0],[1,1]])
    (p/'results.json').write_text(json.dumps(doc))
   rows=co.collect(root);self.assertEqual(len(rows),4)
   out=root/'summary.csv';co.write(rows,out);self.assertEqual(len(out.read_text().splitlines()),5)

if __name__=='__main__':unittest.main()