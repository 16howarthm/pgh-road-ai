import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from road_ai import (
    DEFAULT_CATEGORIES,
    FAILED_PREDICTION,
    NORMALIZED_DESCRIPTION_COL,
    balanced_sample,
    completed_result_keys,
    dedupe_permits,
    load_fixed_splits,
    load_results_checkpoint,
    make_examples,
    run_experiment,
    save_fixed_splits,
    save_results_checkpoint,
    score_results,
    split_prompt_test,
    validate_splits,
)


class CategoryExperimentTests(unittest.TestCase):
    @staticmethod
    def source_frame(records_per_category=6):
        rows = []
        row_id = 1
        for category in DEFAULT_CATEGORIES:
            for index in range(records_per_category):
                rows.append(
                    {
                        "_id": row_id,
                        "closure_id": f"C-{row_id}-1",
                        "permit_id": f"{category[:3]}-{index}",
                        "work_description": f"Unique {category} work {index}",
                        "work_type": category,
                    }
                )
                row_id += 1
        return pd.DataFrame(rows)

    def test_deduplicates_to_one_observation_per_permit(self):
        data = pd.DataFrame(
            [
                {"_id": 2, "closure_id": "P-2", "permit_id": "P", "work_description": "second"},
                {"_id": 1, "closure_id": "P-1", "permit_id": "P", "work_description": "first"},
                {"_id": 3, "closure_id": "Q-1", "permit_id": "Q", "work_description": "other"},
            ]
        )
        result = dedupe_permits(data)
        self.assertEqual(result.permit_id.tolist(), ["P", "Q"])
        self.assertEqual(result.loc[result.permit_id.eq("P"), "work_description"].item(), "first")

    def test_balanced_sample_excludes_ambiguous_descriptions(self):
        data = self.source_frame(4)
        ambiguous = pd.DataFrame(
            [
                {
                    "_id": 100,
                    "closure_id": "A-1",
                    "permit_id": "AMB-A",
                    "work_description": "same ambiguous text",
                    "work_type": DEFAULT_CATEGORIES[0],
                },
                {
                    "_id": 101,
                    "closure_id": "B-1",
                    "permit_id": "AMB-B",
                    "work_description": " SAME   AMBIGUOUS text ",
                    "work_type": DEFAULT_CATEGORIES[1],
                },
            ]
        )
        sample, audit = balanced_sample(
            pd.concat([data, ambiguous], ignore_index=True),
            "work_type",
            n_per_category=3,
            return_audit=True,
        )
        self.assertEqual(len(sample), 18)
        self.assertEqual(sample.groupby("work_type").size().to_dict(), {label: 3 for label in DEFAULT_CATEGORIES})
        self.assertEqual(audit["ambiguous_description_groups"], 1)
        self.assertEqual(audit["ambiguous_records_excluded"], 2)
        self.assertNotIn("SAME AMBIGUOUS TEXT", set(sample[NORMALIZED_DESCRIPTION_COL]))

    def test_splits_are_reproducible_and_nonoverlapping(self):
        source = self.source_frame(7)
        sample_a = balanced_sample(source, "work_type", n_per_category=5, random_state=17)
        sample_b = balanced_sample(source.iloc[::-1], "work_type", n_per_category=5, random_state=17)
        prompt_a, test_a = split_prompt_test(
            sample_a, "work_type", prompt_n=12, test_n=18, random_state=17
        )
        prompt_b, test_b = split_prompt_test(
            sample_b, "work_type", prompt_n=12, test_n=18, random_state=17
        )
        self.assertEqual(prompt_a.permit_id.tolist(), prompt_b.permit_id.tolist())
        self.assertEqual(test_a.permit_id.tolist(), test_b.permit_id.tolist())
        self.assertTrue(validate_splits(prompt_a, test_a, "work_type", 12, 18))
        self.assertFalse(set(prompt_a.permit_id) & set(test_a.permit_id))
        self.assertFalse(
            set(prompt_a[NORMALIZED_DESCRIPTION_COL]) & set(test_a[NORMALIZED_DESCRIPTION_COL])
        )

        with tempfile.TemporaryDirectory() as directory:
            save_fixed_splits(prompt_a, test_a, "work_type", directory)
            loaded_prompt, loaded_test = load_fixed_splits("work_type", directory, 12, 18)
            self.assertEqual(loaded_prompt.permit_id.tolist(), prompt_a.permit_id.tolist())
            self.assertEqual(loaded_test.permit_id.tolist(), test_a.permit_id.tolist())

    def test_committed_splits_have_agreed_sizes_and_balance(self):
        split_dir = Path(__file__).resolve().parents[1] / "category_splits"
        prompt, test = load_fixed_splits("work_type", split_dir)
        self.assertTrue(validate_splits(prompt, test, "work_type"))
        self.assertEqual(prompt.groupby("work_type").size().to_dict(), {label: 20 for label in DEFAULT_CATEGORIES})
        self.assertEqual(test.groupby("work_type").size().to_dict(), {label: 30 for label in DEFAULT_CATEGORIES})
        combined = pd.concat([prompt, test], ignore_index=True)
        self.assertEqual(combined.groupby("work_type").size().to_dict(), {label: 50 for label in DEFAULT_CATEGORIES})

    def test_prompt_conditions_use_exact_total_example_counts(self):
        sample = balanced_sample(self.source_frame(5), "work_type", n_per_category=5)
        prompt, _ = split_prompt_test(sample, "work_type", prompt_n=12, test_n=18)
        self.assertEqual(len(make_examples(prompt, "work_type", n_examples=0)), 0)
        self.assertEqual(len(make_examples(prompt, "work_type", n_examples=1)), 1)
        several = make_examples(prompt, "work_type", n_examples=6)
        self.assertEqual(len(several), 6)
        self.assertEqual({example["category"] for example in several}, set(DEFAULT_CATEGORIES))

    @patch("road_ai.classify_one")
    def test_run_maps_api_missing_and_oov_failures(self, classify_one):
        classify_one.side_effect = [
            {"category": "A"},
            {},
            {"category": "NOT-A-LABEL"},
            RuntimeError("provider error"),
        ]
        test = pd.DataFrame(
            {
                "permit_id": ["1", "2", "3", "4"],
                "work_description": ["a", "b", "c", "d"],
                "label": ["A", "B", "A", "B"],
            }
        )
        persisted = []
        progress = []
        result = run_experiment(
            test,
            "label",
            "model",
            categories=["A", "B"],
            client=object(),
            on_result=persisted.append,
            progress_callback=lambda: progress.append(1),
        )
        self.assertEqual(result.predicted.tolist(), ["A", FAILED_PREDICTION, FAILED_PREDICTION, FAILED_PREDICTION])
        self.assertEqual(result.error.notna().sum(), 3)
        self.assertEqual(len(persisted), 4)
        self.assertEqual(len(progress), 4)
        self.assertEqual(persisted[-1]["predicted"], FAILED_PREDICTION)

    def test_results_checkpoint_is_atomic_deduplicated_and_resumable(self):
        rows = pd.DataFrame(
            [
                {"model": "m", "prompt_condition": "zero", "permit_id": "1", "n_examples": 0, "predicted": "A"},
                {"model": "m", "prompt_condition": "zero", "permit_id": "1", "n_examples": 0, "predicted": "B"},
                {"model": "m", "prompt_condition": "zero", "permit_id": "2", "n_examples": 0, "predicted": "A"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classification_results.csv"
            saved = save_results_checkpoint(rows, path)
            self.assertTrue(path.exists())
            self.assertFalse((Path(directory) / ".classification_results.csv.tmp").exists())
            self.assertEqual(len(saved), 2)

            loaded = load_results_checkpoint(path, ["m"], {"zero": 0}, ["1", "2", "3"])
            self.assertEqual(len(loaded), 2)
            self.assertEqual(
                completed_result_keys(loaded),
                {("m", "zero", "1"), ("m", "zero", "2")},
            )
            self.assertEqual(loaded.loc[loaded.permit_id.eq("1"), "predicted"].item(), "B")

            with self.assertRaises(ValueError):
                load_results_checkpoint(path, ["different-model"], {"zero": 0}, ["1", "2"])

    def test_scoring_keeps_failures_in_denominator_and_matrix(self):
        results = pd.DataFrame(
            {
                "actual": ["A", "B", "A"],
                "predicted": ["A", None, "OUTSIDE"],
            }
        )
        scores = score_results(results, categories=["A", "B"])
        self.assertAlmostEqual(scores["accuracy"], 1 / 3)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 3)
        self.assertEqual(scores["n_scored"], 3)
        self.assertEqual(scores["n_failures"], 2)
        self.assertEqual(int(scores["confusion_matrix"].to_numpy().sum()), 3)
        self.assertEqual(scores["confusion_matrix"].loc["A", FAILED_PREDICTION], 1)
        self.assertEqual(scores["confusion_matrix"].loc["B", FAILED_PREDICTION], 1)


if __name__ == "__main__":
    unittest.main()
