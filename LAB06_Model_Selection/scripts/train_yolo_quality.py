"""Lab 06 Task 3: reproducible YOLOv8 classification launcher.

NEW submission-compatible launcher (2026-11-09), NOT the original October
training script and NOT evidence that the historical 224/16/15/4 models were
retrained at the instructor's required 640/32/50/5 configuration.

Examples:
  python scripts/train_yolo_quality.py --data data/quality_split --verify-only
  python scripts/train_yolo_quality.py --data data/quality_split --out new_runs --train

Dataset layout: train/{bad,good}, val/{bad,good}, test/{bad,good}.
The original experiment had the same source-grouped split but used a folder
named val. Use the original manifest to audit group-level leakage.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SEED=2473986
CLASS_NAMES={'good','bad'}
SUFFIXES={'.jpg','.jpeg','.png','.bmp','.webp'}
MODELS={'n':'yolov8n-cls.pt','s':'yolov8s-cls.pt'}


def inspect_split(root: Path, manifest_path: Path|None = None, hash_check=False):
    """Validate directory structure and optional source-level split metadata."""
    root=Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset folder not found: {root}')
    counts={}; names_per_split={}; hashes_per_split={}
    for split in ('train','val','test'):
        folder=root/split
        if not folder.is_dir():
            extra=' (Ultralytics uses val/, not validation/)' if split=='val' else ''
            raise ValueError(f'Missing {folder}{extra}')
        found={p.name for p in folder.iterdir() if p.is_dir()}
        if found!=CLASS_NAMES:raise ValueError(f'{split}: class directories must be exactly bad/ and good/; got {sorted(found)}')
        names_per_split[split]=set();hashes_per_split[split]=set()
        for name in sorted(CLASS_NAMES):
            pictures=sorted(p for p in (folder/name).rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
            if not pictures:raise ValueError(f'No images: {folder/name}')
            counts[f'{split}/{name}']=len(pictures)
            for p in pictures:
                key=(name,p.name)
                if key in names_per_split[split]:raise ValueError(f'Duplicate image basename in {split}: {key}')
                names_per_split[split].add(key)
                if hash_check:hashes_per_split[split].add(hashlib.sha256(p.read_bytes()).hexdigest())
    if any(names_per_split[a] & names_per_split[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Filename overlap across partitions; verify source grouping')
    if hash_check and any(hashes_per_split[a]&hashes_per_split[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Byte-identical image appeared in multiple partitions')
    if manifest_path:
        rows=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        grouped=collections.defaultdict(set); by_split=collections.Counter()
        for row in rows:
            split=row['split']; group=row['source_id']
            if split=='validation':split='val'
            if split not in {'train','val','test'}:raise ValueError(f'Unknown manifest split {split}')
            grouped[group].add(split); by_split[split]+=1
        if any(len(s)>1 for s in grouped.values()):raise ValueError('SOURCE leakage across manifest partitions')
        for split in ('train','val','test'):
            n=sum(counts[f'{split}/{label}'] for label in sorted(CLASS_NAMES))
            if by_split[split]!=n:raise ValueError(f'Manifest count mismatch for {split}: manifest={by_split[split]}, folders={n}')
        print(f'Source-group integrity verified from {manifest_path} ({len(grouped)} unique sources)')
    else:
        print('WARNING: source-group leakage cannot be independently audited without --manifest')
    print(json.dumps(counts,indent=2));return counts


def train(data:Path,out:Path,which:str):
    try:from ultralytics import YOLO
    except ModuleNotFoundError as exc:raise SystemExit('Install Ultralytics first: python -m pip install ultralytics') from exc
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    kinds=['n','s'] if which=='both' else [which]
    for kind in kinds:
        name=f'week7_yolov8{kind}_cls_teacher_protocol'
        if (out/name).exists():raise FileExistsError(f'Refusing to overwrite {out/name}; choose a new --out')
        model=YOLO(MODELS[kind])
        model.train(data=str(data.resolve()),task='classify',epochs=50,patience=5,
                    imgsz=640,batch=32,seed=SEED,project=str(out),name=name,
                    exist_ok=False,verbose=True)
        print('TRAINING FINISHED:',out/name)
        print('Run independent validation/test evaluation before reporting results.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=None,help='Original manifest.json; checks source-group separation')
    ap.add_argument('--hash-check',action='store_true',help='SHA-256 images to detect exact duplicates across splits')
    ap.add_argument('--out',type=Path,default=Path('new_yolo_runs'))
    ap.add_argument('--model',choices=['n','s','both'],default='both')
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify-only',action='store_true',help='Checks data; does NOT train')
    g.add_argument('--train',action='store_true',help='Actually train with teacher-protocol settings')
    args=ap.parse_args(argv)
    inspect_split(args.data,args.manifest,args.hash_check)
    if args.train:train(args.data,args.out,args.model)
    else:print('VERIFY-ONLY: passed structure checks, no model trained.')

if __name__=='__main__':main()