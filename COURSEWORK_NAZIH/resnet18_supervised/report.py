from pathlib import Path
import json,csv,hashlib,zipfile
from sklearn.metrics import classification_report,confusion_matrix
R=Path(__file__).resolve().parent
r=json.loads((R/'result.json').read_text());hist=json.loads((R/'history.json').read_text());p=list(csv.DictReader((R/'validation_predictions.csv').open()));y=[int(x['label']) for x in p];pred=[int(x['prediction']) for x in p];metrics=classification_report(y,pred,labels=[0,1],target_names=['Normal','Defective'],output_dict=True,zero_division=0)
assert abs(metrics['macro avg']['f1-score']-r['validation']['macro_f1'])<1e-12
assert confusion_matrix(y,pred,labels=[0,1]).tolist()==r['validation']['confusion_matrix']
(R/'metric_verification.json').write_text(json.dumps({'independent_sklearn_agreement':True,'classification_report':metrics},indent=2))
lines=['# ResNet18 supervised engineering pilot','', '**Status: training completed. No independent final test; no paper addition implemented yet.**','',
'Full ResNet18 was fine-tuned on 70 already-accessed Motor images (35 Normal, 35 Defective). Validation uses 29 already-accessed images (15 Normal, 14 Defective). Numeric suffix groups are disjoint, but physical specimen independence remains unverified. This run verifies the supervised implementation; it does not resolve the data audit blocker.', '',
'| Epoch | Training loss | Validation loss | Validation macro-F1 | Validation accuracy |','|---|---:|---:|---:|---:|']
for h in hist:lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['validation']['loss']:.4f} | {h['validation']['macro_f1']:.2%} | {h['validation']['accuracy']:.2%} |")
lines+=['',f"Selected epoch: {r['best_epoch']} (validation macro-F1; earliest tie). Selection makes this validation result optimistic as a final-performance estimate.",'',f"Selected confusion matrix (rows true Normal/Defective, columns predicted Normal/Defective): {r['validation']['confusion_matrix']}.",'',
'Independent scikit-learn recomputation matches the in-training confusion matrix and macro-F1. Convolution and classifier weights changed, gradients remained finite, and fresh model reload reproduced logits within tolerance. See result.json for numerical evidence and software versions.', '',
f"Training runtime: {r['runtime_seconds']:.1f} seconds on two CPU threads. Full classifier parameters: {r['parameter_count']:,}.",'',
'Do not interpret this result as the old 83.70% frozen-feature result improving or deteriorating: labels, learning regime and split are different. The first epoch predicting every validation image as Defective is retained in the history. No extra epochs or alternative settings were tried after seeing the outcomes.', '',
'Next: establish a defensible supervised development/confirmation dataset, choose a specific paper idea and implement a matched baseline/addition comparison. Current baseline settings remain a pilot starting point, not a final optimized protocol.', '',
'Read DECISION.md for source review, limitations and reproduction commands. Model weights, configuration, split manifest, per-image predictions and code are included.']
(R/'Pilot_Report_EN.md').write_text('\n'.join(lines)+'\n')
files=[f for f in R.iterdir() if f.is_file() and f.name!='SHA256.json'];(R/'SHA256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
with zipfile.ZipFile(R.parent/'ResNet18_Supervised_Pilot.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files+[R/'SHA256.json']:z.write(f,'resnet18_supervised/'+f.name)
with zipfile.ZipFile(R.parent/'ResNet18_Supervised_Pilot.zip') as z:assert z.testzip() is None
print(json.dumps(r,indent=2))
