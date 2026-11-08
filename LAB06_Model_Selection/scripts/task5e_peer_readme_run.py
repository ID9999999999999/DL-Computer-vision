"""Optional Task5E: invoke the classmate's *unchanged* predict.py README command.
Does not train or alter either checkpoint; requires local access to ZIP and labeled test images.
A script's successful exit does not prove the original peer training is reproducible.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--peer-zip",type=Path,required=True)
    ap.add_argument("--test",type=Path,required=True,help="our original yolo_data/test with good and bad")
    ap.add_argument("--out",type=Path,required=True,help="new output path")
    a=ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        ap.error("Choose an empty output directory to preserve older results")
    if not a.peer_zip.is_file(): ap.error(f"Missing peer ZIP: {a.peer_zip}")
    if not a.test.is_dir(): ap.error(f"Missing test folder: {a.test}")
    counts={}
    for cls in ("good","bad"):
        d=a.test/cls
        if not d.is_dir(): ap.error(f"Missing class directory: {d}")
        counts[cls]=len([p for p in d.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".webp")])
    if counts!={"good":135,"bad":135}:
        ap.error(f"Expected untouched 135 good + 135 bad Test images; got {counts}")
    a.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(a.peer_zip) as z:
            for info in z.infolist():
                rel=Path(info.filename)
                if rel.is_absolute() or ".." in rel.parts:
                    raise RuntimeError("Unsafe archive path")
            z.extractall(root)
        dirs=[d for d in root.iterdir() if d.is_dir() and (d/"predict.py").is_file() and (d/"README.md").is_file()]
        if len(dirs)!=1: raise RuntimeError("Expected one peer package directory with predict.py + README.md")
        pkg=dirs[0]
        pts=list(pkg.glob("*.pt"))
        if len(pts)!=1: raise RuntimeError("Expected exactly one peer checkpoint")
        import ultralytics
        # This is the explicit README invocation. Use its default output location.
        cmd=[sys.executable,"predict.py","--source",str(a.test.resolve())]
        result=subprocess.run(cmd,cwd=pkg,capture_output=True,text=True)
        (a.out/"peer_stdout.txt").write_text(result.stdout,encoding="utf-8")
        (a.out/"peer_stderr.txt").write_text(result.stderr,encoding="utf-8")
        summary={"task":"5E","status":"peer README invocation executed" if result.returncode==0 else "peer README invocation failed",
          "exit_code":result.returncode,"command":cmd,
          "peer_zip_sha256":sha256(a.peer_zip),"checkpoint_sha256":sha256(pts[0]),
          "test_counts":counts,"platform":platform.platform(),"python":sys.version,
          "ultralytics":ultralytics.__version__,"accuracy_on_our_test":None,
          "warning":"Only our test is used; not the peer's original published held-out images."}
        for csvpath in pkg.rglob("*.csv"):
            if csvpath.name.lower().startswith("prediction"):
                target=a.out/"peer_predictions.csv"
                shutil.copy2(csvpath,target)
                with target.open(newline="",encoding="utf-8-sig") as f:
                    reader=csv.DictReader(f)
                    rows=list(reader)
                    summary["predictions_rows"]=len(rows)
                    summary["prediction_columns"]=reader.fieldnames
                break
        (a.out/"peer_readme_invocation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2))
        if result.returncode:
            raise SystemExit(result.returncode)

if __name__=="__main__":
    main()
