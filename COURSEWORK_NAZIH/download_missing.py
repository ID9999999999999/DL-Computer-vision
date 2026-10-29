"""Recover the *specific* 99 source images in supervised manifest, from the pinned 15-GB ZIP via HTTP byte ranges.
The full 15GB archive is never downloaded. Every file is checked against the SHA256 recorded during the original pilot.
Network access required. The helper follows the original project's known range request format.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from archive_io import RemoteFile,read_member
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    assert 1<=args.workers<=6
    rows=json.loads((ROOT/'resnet18_supervised/manifest.json').read_text())
    pending=[]
    for row in rows:
        dest=ROOT/row['path']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==row['sha256']:continue
        pending.append(row)
    print(f'Images already verified: {len(rows)-len(pending)}/99; need to retrieve {len(pending)}',flush=True)
    if not pending:return
    with zipfile.ZipFile(RemoteFile()) as z:
        by_suffix={}
        for info in z.infolist():
            name=info.filename.replace('\\','/')
            if name.startswith('Motor/') or '/Motor/' in name:
                by_suffix.setdefault(name[name.index('Motor/'):],[]).append(info)
        jobs=[]
        for row in pending:
            key=row['path'].split('/data/',1)[1]
            candidates=by_suffix.get(key,[])
            if len(candidates)!=1: raise ValueError(f'Expected exactly one ZIP member for {key}, found {len(candidates)}')
            jobs.append((row,candidates[0]))
    def task(job):
        row,info=job
        b=read_member(RemoteFile(),info)
        if hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError(f'SHA256 mismatch: {row["path"]}')
        path=ROOT/row['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
        return row['path']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for j,fut in enumerate(as_completed([pool.submit(task,j) for j in jobs]),1):
            fut.result()
            if j%10==0 or j==len(jobs):print('Verified',j,'/',len(jobs),'downloaded images',flush=True)
    print('All 99 source-image hashes now match the pilot manifest.')
if __name__=='__main__':main()
