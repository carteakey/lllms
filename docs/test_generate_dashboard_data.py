import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs" / "generate_dashboard_data.py"
SPEC = importlib.util.spec_from_file_location("generate_dashboard_data", MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def valid_community_run():
    return {
        "id": "test-24gb-run",
        "submittedAt": "2026-07-17",
        "sourceUrl": "https://example.com/raw",
        "hardware": {"gpu": "GPU", "vramGb": 24, "cpu": "CPU", "ramGb": 64},
        "software": {
            "os": "Linux",
            "backend": "CUDA",
            "llamaCppCommit": "0123456789abcdef",
        },
        "model": {"name": "org/model", "quant": "Q4_K_M"},
        "benchmark": {
            "command": "llama-bench -m model.gguf -p 512 -n 128 -r 5",
            "testedContext": 65536,
            "cacheState": "cold",
            "promptTokens": 512,
            "generatedTokens": 128,
            "repetitions": 5,
        },
        "metrics": {"pp": 1000.0, "tg": 50.0, "draftAcceptance": None},
    }


class DashboardDataTest(unittest.TestCase):
    def test_every_local_profile_has_complete_evidence_shape(self):
        meta = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        for profile_id, display in meta["models"].items():
            with self.subTest(profile=profile_id):
                evidence = dashboard.profile_evidence(display)
                self.assertEqual(set(dashboard.EVIDENCE_FIELDS), set(evidence) - {"scope"})
                self.assertEqual("local", evidence["scope"])

    def test_community_run_requires_reproducibility_fields(self):
        run = valid_community_run()
        self.assertEqual([run], dashboard.validate_community_runs({"runs": [run]}))

        run["software"]["llamaCppCommit"] = None
        with self.assertRaises(SystemExit):
            dashboard.validate_community_runs({"runs": [run]})

    def test_community_ids_do_not_enter_local_profile_source(self):
        parsed = dashboard.parse_models((ROOT / "llama-swap.yaml").read_text())
        community = valid_community_run()
        self.assertNotIn(community["id"], parsed)


if __name__ == "__main__":
    unittest.main()
