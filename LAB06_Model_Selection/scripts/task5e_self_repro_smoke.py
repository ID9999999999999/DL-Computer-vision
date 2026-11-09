"""Task 5E self-reproducibility smoke-test on SYNTHETIC images only.
Checks the public share package and literal README-style CLI, not its accuracy.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import csv
import hashlib
import json
import shutil
import subprocess
import sys

root=Path("/tmp/lab06_task5e")
pkg=Path("/tmp/our_lab06_package/yolov8n_quality_gate")
weight=pkg/"yolov8n_quality_gate.pt"
script=pkg/"predict.py"
readme=pkg/"README.md"
expected="c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7"
for p in (weight,script,readme):
    assert p.exists(), f"Missing package file: {p}"
digest=hashlib.sha256(weight.read_bytes()).hexdigest()
assert digest==expected, f"Checkpoint SHA mismatch: {digest}"
assert "--source" in readme.read_text("utf-8"), "README missing CLI use"
dataset=root/"test"
for label in ("good","bad"):
    folder=dataset/label
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        im=Image.new("RGB",(448,448),("white" if label=="good" else "lightgray"))
        d=ImageDraw.Draw(im)
        d.rectangle((55,50,395,400),outline="black",width=3)
        for k in range(8):
            d.line([(100,110+30*k),(350,110+30*k)],fill="black",width=2)
        if label=="bad":
            d.ellipse((80+i*60,80,380,200),fill=(245,245,245))
        im.save(folder/f"{label}_{i}.jpg")
csvfile=root/"predictions.csv"
command=[sys.executable,str(script),"--source",str(dataset),"--out",str(csvfile)]
run=subprocess.run(command,cwd=pkg,text=True,capture_output=True,timeout=180)
(root/"stdout.txt").write_text(run.stdout,encoding="utf-8")
(root/"stderr.txt").write_text(run.stderr,encoding="utf-8")
if run.returncode:
    raise RuntimeError(f"README-style predict.py command failed {run.returncode}: {run.stderr[-3000:]} {run.stdout[-3000:]}")
assert csvfile.is_file(), "Prediction CSV not written"
with csvfile.open(newline="",encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    rows=list(reader)
    fields=reader.fieldnames
assert len(rows)==4, f"Expected four synthetic images in CSV; got {len(rows)}"
assert fields and len(fields)>=3, f"Unexpected CSV structure: {fields}"
summary={
  "task":"5E","date":"2026-11-09",
  "test_type":"OWN share package README CLI smoke test, NOT PEER experiment and NOT accuracy benchmark",
  "package_files":["README.md","predict.py","yolov8n_quality_gate.pt"],
  "original_checkpoint_sha256":digest,
  "synthetic_input_images":4,"synthetic_predictions_rows":len(rows),
  "script_returncode":run.returncode,"csv_columns":fields,
  "real_test_accuracy":None,
  "limits":[
    "Images are fabricated for smoke-testing only; predictions are not used as accuracy evidence.",
    "Classmate original zip was not run by this workflow.",
    "No original Dataset B images were provided."]
}
(root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
