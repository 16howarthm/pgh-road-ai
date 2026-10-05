import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from vercel_classifier import classifier_demo_vercel as backup
from road_ai import classification_prompt, load_fixed_splits, make_examples, DEFAULT_CATEGORIES


class BackupClassifierTests(unittest.TestCase):
    def test_prompt_matches_evaluated_six_examples(self):
        p, _ = load_fixed_splits('work_type', 'category_splits')
        self.assertEqual(backup.PROMPT['system_prompt'], classification_prompt(
            DEFAULT_CATEGORIES, make_examples(p, 'work_type', 6)))
        self.assertEqual(backup.PROMPT['categories'], DEFAULT_CATEGORIES)

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'})
    @patch.object(backup, 'urlopen')
    def test_valid_category_and_provider_request(self, urlopen):
        urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
            'choices': [{'message': {'content': '```json\n{"category":"crane"}\n```'}}]
        }).encode())
        self.assertEqual(backup.classify_description(' crane lift ')['category'], 'CRANE')
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload['messages'][1]['content'], 'crane lift')
        self.assertTrue(payload['provider']['allow_fallbacks'])
        self.assertEqual(request.full_url, 'https://openrouter.ai/api/v1/chat/completions')

    @patch.object(backup, 'urlopen')
    def test_invalid_input_never_calls_provider(self, urlopen):
        for value in ['', ' ', None, 17, 'x' * 4001]:
            with self.subTest(value=str(value)[:20]), self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description(value)
            self.assertEqual(error.exception.status, 400)
        urlopen.assert_not_called()

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'})
    @patch.object(backup, 'urlopen')
    def test_errors_do_not_expose_provider_body(self, urlopen):
        urlopen.side_effect = HTTPError('url', 403, 'sensitive details', {}, BytesIO(b'secret'))
        with self.assertRaises(backup.ClassificationError) as error:
            backup.classify_description('crane lift')
        self.assertIn('API key and credit balance', str(error.exception))
        self.assertNotIn('secret', str(error.exception))

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'})
    @patch.object(backup, 'urlopen')
    def test_invalid_model_output_is_rejected(self, urlopen):
        for content in ['not json', '{"category":"OTHER"}', 'null']:
            urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
                'choices': [{'message': {'content': content}}]}).encode())
            with self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description('crane lift')
            self.assertEqual(error.exception.status, 502)
