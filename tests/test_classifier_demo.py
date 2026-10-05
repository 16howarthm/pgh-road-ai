import unittest
from contextlib import nullcontext
from pathlib import Path
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
    def test_streamlit_path_uses_jetstream_client_without_sdk_retries(
        self, jetstream_client
    ):
        request_client = MagicMock()
        request_client.chat.completions.create.return_value = self.response(
            '{"category":"CRANE"}'
        )
        jetstream_client.return_value.with_options.return_value = request_client

        category = classifier_demo.classify_description(
            "crane lift", api_key="secret"
        )

        self.assertEqual(category, "CRANE")
        jetstream_client.assert_called_once_with("secret")
        jetstream_client.return_value.with_options.assert_called_once_with(
            max_retries=0
        )
        source = Path(classifier_demo.__file__).read_text(encoding="utf-8")
        self.assertNotIn("openrouter", source.lower())

    @patch("classifier_demo.jetstream_client")
    def test_blank_input_does_not_create_client_or_call_api(self, jetstream_client):
        with self.assertRaisesRegex(ValueError, "Enter a work description"):
            classifier_demo.classify_description("   ", api_key="secret")
        jetstream_client.assert_not_called()

    @patch("classifier_demo.time.sleep")
    def test_invalid_and_out_of_vocabulary_output_never_fall_back(self, sleep):
        client = MagicMock()
        client.chat.completions.create.return_value = self.response("not JSON")
        with self.assertRaises(classifier_demo.InvalidModelOutputError):
            classifier_demo.classify_description("street work", client=client)
        self.assertEqual(client.chat.completions.create.call_count, 2)

        client.reset_mock()
        client.chat.completions.create.return_value = self.response(
            '{"category":"OTHER"}'
        )
        with self.assertRaises(classifier_demo.InvalidModelOutputError):
            classifier_demo.classify_description("street work", client=client)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        sleep.assert_called()

    @patch("classifier_demo.time.sleep")
    def test_503_then_success_retries_and_returns_category(self, sleep):
        unavailable = RuntimeError("provider body must not be shown")
        unavailable.status_code = 503
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            unavailable,
            self.response('{"category":"CRANE"}'),
        ]

        category = classifier_demo.classify_description(
            "crane lift", client=client
        )

        self.assertEqual(category, "CRANE")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("classifier_demo.time.sleep")
    def test_timeout_then_success_retries_and_returns_category(self, sleep):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            TimeoutError("sensitive timeout details"),
            self.response('{"category":"MATERIALS"}'),
        ]

        category = classifier_demo.classify_description(
            "material staging", client=client
        )

        self.assertEqual(category, "MATERIALS")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("classifier_demo.time.sleep")
    def test_repeated_transient_failures_stop_after_three_attempts(self, sleep):
        unavailable = RuntimeError("sensitive provider details")
        unavailable.status_code = 503
        client = MagicMock()
        client.chat.completions.create.side_effect = unavailable

        with self.assertRaises(classifier_demo.JetstreamUnavailableError):
            classifier_demo.classify_description("street work", client=client)

        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    @patch("classifier_demo.time.sleep")
    def test_authentication_failure_does_not_retry(self, sleep):
        for status in (401, 403):
            with self.subTest(status=status):
                denied = RuntimeError("secret provider response")
                denied.status_code = status
                client = MagicMock()
                client.chat.completions.create.side_effect = denied

                with self.assertRaises(classifier_demo.JetstreamAccessError):
                    classifier_demo.classify_description(
                        "street work", client=client
                    )

                client.chat.completions.create.assert_called_once()
        sleep.assert_not_called()

    @patch("classifier_demo.time.sleep")
    def test_repeated_rate_limits_use_specific_error(self, sleep):
        rate_limited = RuntimeError("provider response body")
        rate_limited.status_code = 429
        client = MagicMock()
        client.chat.completions.create.side_effect = rate_limited

        with self.assertRaises(classifier_demo.JetstreamRateLimitError):
            classifier_demo.classify_description("street work", client=client)

        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

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
            side_effect=classifier_demo.JetstreamUnavailableError,
        ):
            classifier_demo.main()

        self.assertEqual(
            streamlit.errors,
            ["Jetstream is temporarily unavailable. Please try again shortly."],
        )

    def test_authentication_failure_shows_actionable_safe_error(self):
        streamlit = FakeStreamlit()
        with patch.object(classifier_demo, "st", streamlit), patch.object(
            classifier_demo,
            "classify_description",
            side_effect=classifier_demo.JetstreamAccessError,
        ):
            classifier_demo.main()

        self.assertEqual(
            streamlit.errors,
            ["Jetstream rejected API access. Check the app's Jetstream secret."],
        )

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
