import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "human_audit.py"
spec = importlib.util.spec_from_file_location("human_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class HumanAuditTests(unittest.TestCase):
    def population(self):
        rows = []
        for q in audit.QUOTAS:
            for batch in ("with_attacks", "without_attacks"):
                for group in ("deepseek_distill", "qwen_native"):
                    for i in range(40):
                        rows.append({"id": f"{q}-{batch}-{group}-{i}", "compliance_quadrant": q,
                                     "batch": batch, "family": group})
        return rows

    def test_sampling_is_reproducible_and_has_correct_quotas(self):
        sample, strata = audit.select_sample(self.population(), 42)
        self.assertEqual(sample, audit.select_sample(self.population(), 42)[0])
        self.assertEqual(len(sample), 320)
        self.assertEqual(len({r["id"] for r in sample}), 320)
        self.assertEqual(len(strata), 16)
        for row in strata:
            self.assertEqual(row["sample_n"], audit.QUOTAS[row["quadrant"]])
            self.assertEqual(row["inclusion_probability"], row["sample_n"] / 40)

    def test_insufficient_stratum_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            audit.select_sample([], 42)

    def annotation(self, aid, reviewer, cot="unsafe", answer="safe"):
        return {"audit_id": aid, "reviewer_id": reviewer, "cot_unsafe": cot,
                "output_unsafe": answer, "cot_unsafe_note": "Reasoning evidence",
                "output_unsafe_note": "Answer evidence"}

    def test_disagreement_unclear_and_adjudication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = [{"audit_id": aid, "auto_quadrant": "Q3", "batch": "with_attacks",
                         "family": "deepseek_distill", "auto_cot_unsafe": 1,
                         "auto_output_unsafe": 0} for aid in ("a", "b", "c")]
            audit.write_jsonl(root / "manifest", manifest)
            audit.write_jsonl(root / "r1", [self.annotation(aid, "R1") for aid in ("a", "b", "c")])
            audit.write_jsonl(root / "r2", [self.annotation("a", "R2"),
                                            self.annotation("b", "R2", "safe"),
                                            self.annotation("c", "R2", "unclear")])
            audit.score(root / "manifest", [root / "r1", root / "r2"], root / "initial")
            report = json.loads((root / "initial/reliability.json").read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["resolved_n"], 1)
            self.assertEqual(report["human_agreement"]["cot_unsafe"]["binary_only"]["n"], 2)
            audit.write_jsonl(root / "third", [self.annotation("b", "R3", "safe"),
                                               self.annotation("c", "R3")])
            audit.score(root / "manifest", [root / "r1", root / "r2"], root / "final", root / "third")
            report = json.loads((root / "final/reliability.json").read_text())
            self.assertEqual(report["status"], "complete")
            summary = next(r for r in report["stratum_confirmation"] if r["sampled_n"])
            self.assertEqual(summary["confirmed_n"], 2)
            self.assertEqual(report["audit_sample_channel_confusion"]["cot_unsafe"]["auto1_human0"], 1)

    def test_invalid_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations"
            row = self.annotation("a", "R1")
            for rows in ([row, row], [self.annotation("unknown", "R1")],
                         [{**row, "cot_unsafe": 1}], [{**row, "cot_unsafe_note": ""}]):
                audit.write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    audit.read_annotations(path, {"a"})

    def test_same_reviewer_cannot_count_as_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit.write_jsonl(root / "manifest", [{"audit_id": "a"}])
            audit.write_jsonl(root / "labels", [self.annotation("a", "R1")])
            with self.assertRaises(ValueError):
                audit.score(root / "manifest", [root / "labels", root / "labels"], root / "report")

    def test_degenerate_kappa_and_empty_sample(self):
        self.assertIsNone(audit.agreement([])["agreement"])
        self.assertIsNone(audit.agreement([("safe", "safe")])["cohen_kappa"])
        self.assertAlmostEqual(audit.agreement([("safe", "safe"), ("unsafe", "unsafe")])["cohen_kappa"], 1)

    def test_packet_blinding_escaping_and_overwrite_protection(self):
        rows = self.population()
        for row in rows:
            row.update({"target_model": "deepseek-r1-7b", "source_target_model": "deepseek-r1-7b",
                        "prompt": "</script><script>alert(1)</script>", "cot_trace": "trace",
                        "final_answer": "answer", "attack_method": "rto", "harm_category": "cybercrime",
                        "cot_unsafe": 1, "output_unsafe": 0, "cot_reason": "SECRET_JUDGE_REASON"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            with patch.object(audit, "load_population", return_value=(rows, {"source": "hash"})):
                audit.prepare(Path(directory), output, 42)
                with self.assertRaises(ValueError):
                    audit.prepare(Path(directory), output, 42)
            blind = audit.read_jsonl(output / "review_samples.jsonl")
            self.assertEqual(set(blind[0]), {"audit_id", "prompt", "cot_trace", "final_answer"})
            html = (output / "review.html").read_text(encoding="utf-8")
            self.assertNotIn("SECRET_JUDGE_REASON", html)
            self.assertNotIn("</script><script>alert(1)</script>", html)
            self.assertIn("\\u003c/script>", html)


if __name__ == "__main__":
    unittest.main()
