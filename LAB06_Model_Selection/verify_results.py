"""Verify the supplied run records without loading images or model checkpoints."""
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

from lab06 import MODELS, choose


def verify(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    assert hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() == protocol['manifest_sha256']
    rows = json.loads((root / 'manifest.json').read_text())
    lookup = {r['image_id']: r for r in rows}
    assert len(lookup) == len(rows) == 1800
    sources, hashes = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows:
        sources[r['source_id']].add(r['split'])
        hashes[r['sha256']].add(r['split'])
    assert all(len(s) == 1 for s in sources.values())
    assert all(len(s) == 1 for s in hashes.values())
    assert collections.Counter(r['split'] for r in rows) == {'train':1260,'val':270,'test':270}
    evaluations = []
    for model in MODELS:
        e = json.loads((root / model / 'evaluation.json').read_text())
        predictions = json.loads((root / model / 'predictions.json').read_text())
        assert len(predictions) == len({r['image_id'] for r in predictions}) == 540
        for split in ('val', 'test'):
            subset = [r for r in predictions if r['split'] == split]
            assert {r['image_id'] for r in subset} == {r['image_id'] for r in rows if r['split'] == split}
            matrix = [[0, 0], [0, 0]]
            for r in subset:
                original = lookup[r['image_id']]
                assert original['source_id'] == r['source_id']
                assert r['y_true'] == int(original['quality_label'] == 'bad')
                assert 0 <= r['p_bad'] <= 1
                assert r['y_pred'] == int(r['p_bad'] >= protocol['config']['threshold'])
                matrix[r['y_true']][r['y_pred']] += 1
            tn, fp = matrix[0]
            fn, tp = matrix[1]
            metrics = dict(accuracy=(tn+tp)/len(subset), precision_bad=tp/(tp+fp),
                           recall_bad=tp/(tp+fn), f1_bad=2*tp/(2*tp+fp+fn),
                           macro_f1=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2)
            assert matrix == e[split]['confusion_matrix']
            assert e[split]['n'] == len(subset)
            assert all(math.isclose(v, e[split][k]) for k, v in metrics.items())
        assert len(e['latency_samples_ms']) == protocol['config']['latency_runs']
        assert math.isclose(statistics.mean(e['latency_samples_ms']), e['latency_ms'])
        evaluations.append(e)
        print(f"{model}: validation/test metrics, threshold and latency mean verified")
    selected, passed = choose(evaluations, protocol['config']['acceptance'])
    decision = json.loads((root/'decision.json').read_text())
    assert selected['model'] == decision['candidate'] == 'yolov8n'
    assert passed == decision['validation_acceptance_passed']
    print('Split: 1260 / 270 / 270 images, 210 / 45 / 45 source documents')
    print('No source ID or recorded image hash crosses partition boundaries')
    print('Validation-only selection reproduced: yolov8n')
    print('Scope: saved records verified. Image and checkpoint bytes are external.')
    return evaluations


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent / 'results')
