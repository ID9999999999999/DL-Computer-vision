"""Protocol/report tests. Fixtures are artificial and never experiment results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab06 as lab


def metadata():
    records = []
    for kind in ("inn", "passport", "snils"):
        for s in range(20):
            for quality in ("good", "bad"):
                for v in range(3):
                    records.append({"image_id": f"{kind}_{s}_{quality}_{v}", "source_id": f"{kind}_{s}",
                                    "class_name": kind, "quality_label": quality, "split": "train",
                                    "image_path": f"{quality}/{kind}_{s}_{v}.jpg", "seed": 2473986})
    return records


class ProtocolTests(unittest.TestCase):
    def test_exact_source_grouped_split(self):
        original = metadata()
        rows = lab.source_split(original, 2473986)
        self.assertEqual(Counter(r["split"] for r in rows), {"train": 252, "val": 54, "test": 54})
        groups = defaultdict(set)
        for r in rows: groups[r["source_id"]].add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        for kind in ("inn", "passport", "snils"):
            self.assertEqual(Counter(r["split"] for r in rows if r["class_name"] == kind), {"train": 84, "val": 18, "test": 18})
        self.assertTrue(all(r["split"] == "train" for r in original))
        self.assertEqual(rows, lab.source_split(original, 2473986))
        other = lab.source_split(original, 2473987)
        self.assertNotEqual([r["split"] for r in rows], [r["split"] for r in other])

    def test_reject_incompatible_group(self):
        rows = metadata()
        rows.pop()
        with self.assertRaises(ValueError): lab.source_split(rows, 2473986)

    def test_bad_is_positive_and_matrix_order(self):
        result = lab.metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision_bad"], 0.5)
        self.assertEqual(result["recall_bad"], 0.5)
        self.assertEqual(result["f1_bad"], 0.5)

    def test_selection_ignores_test_scores(self):
        limits = {"accuracy": .9, "f1_bad": .9, "recall_bad": .95, "latency_ms": 50, "artifact_mib": 100}
        base = {"val": {"accuracy": .95, "f1_bad": .94, "recall_bad": .96}, "artifact_mib": 20}
        a = dict(copy.deepcopy(base), model="fast", latency_ms=10, test={"f1_bad": .1})
        b = dict(copy.deepcopy(base), model="slow", latency_ms=20, test={"f1_bad": 1})
        self.assertEqual(lab.choose([a, b], limits)[0]["model"], "fast")
        a["test"] = {"f1_bad": 1}; b["test"] = {"f1_bad": 0}
        self.assertEqual(lab.choose([a, b], limits)[0]["model"], "fast")

    def test_no_passing_candidate_is_provisional(self):
        limits = {"accuracy": .9, "f1_bad": .9, "recall_bad": .95, "latency_ms": 50, "artifact_mib": 100}
        e = {"model": "low-recall", "val": {"accuracy": .95, "f1_bad": .91, "recall_bad": .8}, "latency_ms": 10, "artifact_mib": 20}
        self.assertFalse(lab.choose([e], limits)[1])

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"; dataset.mkdir()
            (root / "outside.jpg").write_bytes(b"not an image")
            with self.assertRaises(FileNotFoundError): lab.safe_image(dataset, "../outside.jpg")

    def test_prepare_checks_pixels_hashes_and_immutability(self):
        from PIL import Image
        cfg = lab.read_json(lab.ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"; dataset.mkdir()
            records = metadata()
            for n, r in enumerate(records):
                f = dataset / r["image_path"]; f.parent.mkdir(exist_ok=True)
                # Lossless fixtures keep adjacent artificial colors distinct.
                Image.new("RGB", (640, 640), (n % 256, n // 256 * 50, 30)).save(f, format="PNG")
            md = dataset / "metadata.jsonl"
            md.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            original = md.read_bytes()
            with patch("lab06.importlib.metadata.version", return_value="artificial-test-version"):
                lab.prepare(dataset, root / "out", cfg)
            p = lab.protocol(root / "out")
            lab.check_images(root / "out", p)
            self.assertEqual(md.read_bytes(), original)
            self.assertEqual(p["new_split"], {"train": 252, "val": 54, "test": 54})
            first = dataset / records[0]["image_path"]
            first.write_bytes(b"changed artificial image")
            with self.assertRaises(ValueError): lab.check_images(root / "out", p)

    def test_report_generation_on_artificial_fixtures(self):
        from PIL import Image
        cfg = lab.read_json(lab.ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            image = out / "artificial.jpg"
            Image.new("RGB", (640, 640), "white").save(image)
            lab.write_json(out / "manifest.json", [])
            lab.write_json(out / "protocol.json", {"dataset": str(out), "config": cfg,
                "manifest_sha256": lab.digest(out / "manifest.json"), "config_sha256": lab.digest(lab.ROOT / "config.json"),
                "new_split": {"train": 1260, "val": 270, "test": 270}, "old_split": {"train": 1230, "validation": 228, "test": 342}})
            for i, name in enumerate(lab.MODELS):
                folder = out / name; folder.mkdir()
                result = lab.metrics([0, 1], [.1, .8], .5)
                lab.write_json(folder / "evaluation.json", {"model": name, "val": result, "test": result,
                    "latency_ms": i + 1, "latency_p95_ms": i + 2, "artifact_mib": 10, "parameters": 100,
                    "training_seconds": 1})
                lab.write_json(folder / "predictions.json", [
                    {"split": "test", "y_true": 1, "y_pred": 0, "defect_type": "blur", "p_bad": .1, "image_path": image.name}])
                if name in lab.MODELS[:2]:
                    lab.write_json(folder / "history.json", {"loss": [.7, .5], "val_loss": [.6, .4], "accuracy": [.5, .8], "val_accuracy": [.5, .75]})
            lab.report(out)
            self.assertTrue((out / "Final_Report.md").is_file())
            self.assertTrue((out / "Presentation_5min.md").is_file())
            self.assertTrue((out / "confusion_matrices.png").is_file())
            self.assertTrue((out / "accuracy_vs_speed.png").is_file())
            self.assertEqual(lab.read_json(out / "decision.json")["candidate"], "mobilenetv2")


if __name__ == "__main__":
    unittest.main()
