import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import classifier_demo
from road_ai import (
    DEFAULT_CATEGORIES,
    classification_prompt,
    load_fixed_splits,
    make_examples,
)


class FakeStreamlit:
    def __init__(self, description="crane lift"):
        self.description = description
        self.secrets = {"jetstream_api_key": "secret"}
        self.successes = []
        self.errors = []
        self.warnings = []
        self.captions = []

    def set_page_config(self, **kwargs):
        pass

    def title(self, value):
        pass

    def write(self, value):
        pass

    def text_area(self, label, **kwargs):
        return self.description

    def button(self, label, **kwargs):
        return True

    def warning(self, value):
        self.warnings.append(value)

    def error(self, value):
        self.errors.append(value)

    def spinner(self, value):
        return nullcontext()

    def subheader(self, value):
        pass

    def success(self, value):
        self.successes.append(value)

    def caption(self, value):
        self.captions.append(value)


class ClassifierDemoTests(unittest.TestCase):
    @staticmethod
    def response(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    def test_uses_exact_six_example_scout_condition(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self.response(
            '{"category":"CRANE"}'
        )

        category = classifier_demo.classify_description(
            "  crane lift  ", client=client
        )

        self.assertEqual(category, "CRANE")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "llama-4-scout")
        self.assertEqual(request["temperature"], 0)
        self.assertNotIn("response_format", request)
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("extra_body", request)

        prompt_dev, _ = load_fixed_splits(
            "work_type", classifier_demo.PROJECT_ROOT / "category_splits"
        )
        examples = make_examples(
            prompt_dev, "work_type", n_examples=6, random_state=42
        )
        self.assertEqual(len(examples), 6)
        self.assertEqual(
            request["messages"][0]["content"],
            classification_prompt(DEFAULT_CATEGORIES, examples),
        )

    @patch("classifier_demo.jetstream_client")
    def test_blank_input_does_not_create_client_or_call_api(self, jetstream_client):
        with self.assertRaisesRegex(ValueError, "Enter a work description"):
            classifier_demo.classify_description("   ", api_key="secret")
        jetstream_client.assert_not_called()

    def test_invalid_and_out_of_vocabulary_output_fail(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self.response("not JSON")
        with self.assertRaisesRegex(ValueError, "missing a category"):
            classifier_demo.classify_description("street work", client=client)

        client.chat.completions.create.return_value = self.response(
            '{"category":"OTHER"}'
        )
        with self.assertRaisesRegex(ValueError, "Out-of-vocabulary"):
            classifier_demo.classify_description("street work", client=client)

    def test_mocked_classification_renders_valid_category(self):
        streamlit = FakeStreamlit()
        with patch.object(classifier_demo, "st", streamlit), patch.object(
            classifier_demo, "classify_description", return_value="CRANE"
        ) as classify:
            classifier_demo.main()

        classify.assert_called_once_with("crane lift", api_key="secret")
        self.assertEqual(streamlit.successes, ["CRANE"])
        self.assertEqual(streamlit.errors, [])
        self.assertEqual(
            streamlit.captions, ["Llama 4 Scout · 6-example prompt"]
        )

    def test_blank_ui_input_shows_warning_without_classification(self):
        streamlit = FakeStreamlit("   ")
        with patch.object(classifier_demo, "st", streamlit), patch.object(
            classifier_demo, "classify_description"
        ) as classify:
            classifier_demo.main()

        classify.assert_not_called()
        self.assertEqual(len(streamlit.warnings), 1)

    def test_api_failure_shows_concise_error(self):
        streamlit = FakeStreamlit()
        with patch.object(classifier_demo, "st", streamlit), patch.object(
            classifier_demo,
            "classify_description",
            side_effect=RuntimeError("sensitive provider details"),
        ):
            classifier_demo.main()

        self.assertEqual(
            streamlit.errors, ["Classification failed. Please try again."]
        )
        self.assertNotIn("sensitive provider details", " ".join(streamlit.errors))

    def test_allowed_categories_are_unchanged(self):
        self.assertEqual(
            DEFAULT_CATEGORIES,
            [
                "MACHINERY",
                "DEMOLITION DUMPSTER",
                "CRANE",
                "BARRICADE",
                "MATERIALS",
                "SCAFFOLD",
            ],
        )


if __name__ == "__main__":
    unittest.main()
