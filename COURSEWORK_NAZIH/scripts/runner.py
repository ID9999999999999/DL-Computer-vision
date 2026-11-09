"""Resume-safe batch orchestrator; never overwrites archived 70-image results."""
from pathlib import Path
import argparse,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent))
from core import ROOT,config
from evaluate import frozen_checkpoint_inventory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive');ap.add_argument('--stage',choices=['prepare','train','evaluate','report'],required=True);args=ap.parse_args()
    c=config()
    if args.stage!='report' and not args.archive:ap.error('--archive is required for this stage')
    def run(script,*params):subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--archive',str(args.archive),*map(str,params)],check=True,cwd=ROOT)
    if args.stage=='prepare':run('prepare.py');return
    if args.stage=='train':
        for size in c['new_training_sizes']:
            for seed in c['seeds']:
                for loss in c['arms']:
                    report=ROOT/f'results/size{size}_seed{seed}_{loss}.json'
                    pt=ROOT/f'results/size{size}_seed{seed}_{loss}.pt'
                    if report.exists() or pt.exists():
                        if report.exists() and pt.exists():print('PRESERVED:',size,seed,loss);continue
                        raise RuntimeError('Incomplete partial checkpoint; resolve without overwrite: '+str(report))
                    run('train.py','--size',size,'--seed',seed,'--loss',loss)
        return
    if args.stage=='evaluate':
        frozen_checkpoint_inventory()
        for size in c['new_training_sizes']:
            for seed in c['seeds']:
                for loss in c['arms']:
                    report=ROOT/f'results/eval_size{size}_seed{seed}_{loss}.json'
                    if report.exists():print('PRESERVED:',report);continue
                    run('evaluate.py','--size',size,'--seed',seed,'--loss',loss)
        return
    subprocess.run([sys.executable,str(ROOT/'scripts/report.py')],check=True,cwd=ROOT)
if __name__=='__main__':main()