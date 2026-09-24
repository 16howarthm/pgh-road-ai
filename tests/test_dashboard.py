import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from analysis_backend import generate_analysis
from dashboard_data import clean, parse_path, permit_level
from pattern_analysis import analyze


class DashboardTests(unittest.TestCase):
    def frame(self, statuses):
        return pd.DataFrame({"permit_id": [str(i) for i in range(len(statuses))], "active": statuses,
                             "from_date": "2025-01-01", "to_date": "2026-01-01", "geometry": "[[40,-80],[41,-79]]"})

    def test_only_requested_statuses_are_past_due(self):
        data = clean(self.frame(["t", " TRUE ", None, "null", True, "f", "false", False, "unknown", "1"]), "2026-01-02")
        self.assertEqual(data.past_due_schedule.tolist(), [True] * 5 + [False] * 5)

    def test_date_boundaries_and_bad_dates(self):
        data = self.frame(["t"] * 5)
        data["to_date"] = ["2026-01-01", "2026-01-02", "2026-01-03", "1024-10-20", None]
        data = clean(data, "2026-01-02")
        self.assertEqual(data.past_due_schedule.tolist(), [True, False, False, False, False])
        self.assertEqual(data.to_date_parsed.isna().sum(), 2)

    def test_deduplication_quality_and_summary(self):
        data = self.frame(["t", "t", "f", None])
        data["permit_id"] = ["A", "A", "B", None]
        data["to_date"] = ["2025-01-01", "2025-06-01", "2025-01-01", "2025-01-01"]
        data = clean(data, "2026-01-02")
        queue = data[data.past_due_schedule]
        self.assertEqual(len(permit_level(queue)), 1)
        summary, _, observations, proposals = analyze(data, queue, 1, "2026-01-02")
        self.assertEqual(summary["permit_count"], 1)
        self.assertEqual(summary["median_days_past_due"], 366)
        self.assertEqual(summary["data_quality"]["conflicting_status_or_end_date_permits"], 1)
        self.assertTrue(observations and proposals)
        json.dumps(summary, allow_nan=False)
        empty, _, _, _ = analyze(data, queue.iloc[:0], 1, "2026-01-02")
        self.assertIsNone(empty["median_days_past_due"])

    def test_geometry_validation(self):
        self.assertEqual(parse_path("[[40,-80],[41,-79]]"), [[-80., 40.], [-79., 41.]])
        for value in [None, "bad", "[]", "[[40,-80]]", "[[100,-80],[40,-80]]", [[40, float('nan')], [41, -79]]]:
            self.assertIsNone(parse_path(value))

    def test_model_and_prompt_reach_provider(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Report"))])
        self.assertEqual(generate_analysis({"permit_count": 3}, "provider/model", "Custom prompt", client=client), "Report")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "provider/model")
        self.assertEqual(kwargs["messages"][0]["content"], "Custom prompt")
        self.assertEqual(json.loads(kwargs["messages"][1]["content"]), {"permit_count": 3})
        client.chat.completions.create.return_value.choices[0].message.content = ""
        with self.assertRaises(ValueError):
            generate_analysis({}, "provider/model", "prompt", client=client)


if __name__ == "__main__":
    unittest.main()
