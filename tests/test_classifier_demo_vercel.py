import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from vercel_classifier import classifier_demo_vercel as backup
from road_ai import classification_prompt, load_fixed_splits, make_examples, DEFAULT_CATEGORIES


class BackupClassifierTests(unittest.TestCase):
    @patch.dict(os.environ, {'JETSTREAM_API_KEY': 'test-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_reason_is_requested_and_returned_without_html_rendering(self, urlopen):
        urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
            'choices': [{'message': {'content': json.dumps({
                'category': 'CRANE', 'confidence': 0.9, 'reason': '  A crane lifts equipment.\nThis requires street space.  '
            })}}]
        }).encode())
        result = backup.classify_description('crane lift', provider='jetstream')
        self.assertEqual(result['reason'], 'A crane lifts equipment. This requires street space.')
        prompt = json.loads(urlopen.call_args.args[0].data)['messages'][0]['content']
        self.assertIn('1–2 short sentences', prompt)

    @patch.dict(os.environ, {'JETSTREAM_API_KEY': 'test-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_missing_or_invalid_reason_is_rejected(self, urlopen):
        for reason in [None, '', '   ', 12, ['explanation']]:
            urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
                'choices': [{'message': {'content': json.dumps({'category': 'CRANE', 'confidence': 0.9, 'reason': reason})}}]
            }).encode())
            with self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description('crane lift', provider='jetstream')
            self.assertIn('explanation', str(error.exception))

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'openai-secret',
                             'OPENROUTER_API_KEY': 'router-secret',
                             'JETSTREAM_API_KEY': 'jetstream-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_each_provider_uses_its_own_key_endpoint_and_selected_model(self, urlopen):
        cases = [('openai', 'gpt-4.1', 'openai-secret'),
                 ('openrouter', 'deepseek/deepseek-v3.2', 'router-secret'),
                 ('jetstream', 'gpt-oss-120b', 'jetstream-secret')]
        for provider, model, key in cases:
            with self.subTest(provider=provider):
                urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
                    'choices': [{'message': {'content': '{"category":"CRANE","confidence":0.9,"reason":"The description mentions a crane lift."}'}}]
                }).encode())
                result = backup.classify_description('crane lift', provider, model)
                request = urlopen.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(request.get_header('Authorization'), 'Bearer ' + key)
                self.assertEqual(request.full_url, backup.PROVIDERS[provider]['url'])
                self.assertEqual(result, {'category': 'CRANE', 'confidence': 0.9, 'reason': 'The description mentions a crane lift.', 'provider': provider, 'model': model})
                self.assertEqual(payload['model'], model)
                self.assertEqual('provider' in payload, provider == 'openrouter')
                if provider == 'jetstream':
                    self.assertEqual(payload['reasoning_effort'], 'low')

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'private-key-value'}, clear=True)
    def test_configuration_defaults_to_jetstream_and_never_exposes_secrets(self):
        config = backup.public_configuration()
        self.assertEqual(config['default_provider'], 'jetstream')
        self.assertEqual(config['providers']['jetstream']['default_model'], 'gpt-oss-120b')
        self.assertTrue(config['providers']['openai']['configured'])
        self.assertFalse(config['providers']['jetstream']['configured'])
        self.assertNotIn('private-key-value', json.dumps(config))
        self.assertNotIn('key_env', json.dumps(config))

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'openai-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_invalid_selections_and_missing_key_never_call_provider(self, urlopen):
        for provider, model, status in [('unknown', None, 400),
                                       ({}, None, 400),
                                       ('openai', '', 400),
                                       ('openai', ['gpt-4.1'], 400),
                                       ('openai', 'bad model', 400),
                                       ('openai', 'x' * 161, 400),
                                       ('jetstream', None, 503)]:
            with self.subTest(provider=provider, model=model), self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description('crane lift', provider, model)
            self.assertEqual(error.exception.status, status)
        urlopen.assert_not_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_custom_reasoning_model_does_not_send_temperature(self, urlopen):
        urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
            'choices': [{'message': {'content': '{"category":"CRANE","confidence":0.9,"reason":"The description mentions a crane lift."}'}}]
        }).encode())
        backup.classify_description('crane lift', provider='openai', model='gpt-5-mini')
        self.assertNotIn('temperature', json.loads(urlopen.call_args.args[0].data))

    @patch.object(backup, 'classify_description', return_value={'category': 'CRANE'})
    def test_http_handler_forwards_selected_provider_and_model(self, classify):
        payload = json.dumps({'description': 'crane lift', 'provider': 'openrouter',
                              'model': 'deepseek/deepseek-v3.2'}).encode()
        handler = backup.handler.__new__(backup.handler)
        handler.path = '/api/classify'
        handler.headers = {'Content-Length': str(len(payload))}
        handler.rfile = BytesIO(payload)
        with patch.object(handler, 'send_json') as send:
            handler.do_POST()
        classify.assert_called_once_with('crane lift', 'openrouter', 'deepseek/deepseek-v3.2')
        send.assert_called_once_with(200, {'category': 'CRANE'})

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_quota_exhaustion_and_rate_limiting_have_distinct_messages(self, urlopen):
        for body, expected in [
            ({'error': {'type': 'insufficient_quota', 'code': 'credit_balance_exhausted',
                        'message': 'sensitive'}}, 'insufficient API quota'),
            ({'error': {'code': 'rate_limit_exceeded'}}, 'Too many requests'),
        ]:
            urlopen.side_effect = HTTPError('url', 429, 'private', {}, BytesIO(json.dumps(body).encode()))
            with self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description('crane lift', provider='openai')
            self.assertIn(expected, str(error.exception))
            self.assertNotIn('sensitive', str(error.exception))

    def test_prompt_matches_evaluated_six_examples(self):
        p, _ = load_fixed_splits('work_type', 'category_splits')
        self.assertEqual(backup.PROMPT['system_prompt'], classification_prompt(
            DEFAULT_CATEGORIES, make_examples(p, 'work_type', 6)))
        self.assertEqual(backup.PROMPT['categories'], DEFAULT_CATEGORIES)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'})
    @patch.object(backup, 'urlopen')
    def test_valid_category_and_provider_request(self, urlopen):
        urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
            'choices': [{'message': {'content': '```json\n{"category":"crane","confidence":0.9,"reason":"The description mentions a crane lift."}\n```'}}]
        }).encode())
        self.assertEqual(backup.classify_description(' crane lift ', provider='openai')['category'], 'CRANE')
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload['messages'][1]['content'], 'crane lift')
        self.assertNotIn('provider', payload)
        self.assertEqual(payload['model'], 'gpt-4.1-mini')
        self.assertEqual(payload['max_completion_tokens'], 2048)
        self.assertNotIn('max_tokens', payload)
        self.assertEqual(request.full_url, 'https://api.openai.com/v1/chat/completions')

    @patch.object(backup, 'urlopen')
    def test_invalid_input_never_calls_provider(self, urlopen):
        for value in ['', ' ', None, 17, 'x' * 4001]:
            with self.subTest(value=str(value)[:20]), self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description(value)
            self.assertEqual(error.exception.status, 400)
        urlopen.assert_not_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'})
    @patch.object(backup, 'urlopen')
    def test_errors_do_not_expose_provider_body(self, urlopen):
        urlopen.side_effect = HTTPError('url', 403, 'sensitive details', {}, BytesIO(b'secret'))
        with self.assertRaises(backup.ClassificationError) as error:
            backup.classify_description('crane lift', provider='openai')
        self.assertIn('OPENAI_API_KEY', str(error.exception))
        self.assertNotIn('secret', str(error.exception))

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'})
    @patch.object(backup, 'urlopen')
    def test_invalid_model_output_is_rejected(self, urlopen):
        for content in ['not json', '{"category":"OTHER"}', 'null']:
            urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
                'choices': [{'message': {'content': content}}]}).encode())
            with self.assertRaises(backup.ClassificationError) as error:
                backup.classify_description('crane lift', provider='openai')
            self.assertEqual(error.exception.status, 502)

    @patch.dict(os.environ, {'JETSTREAM_API_KEY': 'test-secret'}, clear=True)
    @patch.object(backup, 'urlopen')
    def test_default_model_and_confidence_validation(self, urlopen):
        for confidence in [0, 0.9, 1, None, -0.1, 1.1, True, '0.9', float('nan')]:
            with self.subTest(confidence=confidence):
                urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({
                    'choices': [{'message': {'content': json.dumps({
                        'category': 'CRANE', 'reason': 'Crane lift.', 'confidence': confidence
                    })}}]
                }).encode())
                if confidence is None or isinstance(confidence, (bool, str)) or not 0 <= confidence <= 1:
                    with self.assertRaises(backup.ClassificationError) as error:
                        backup.classify_description('crane lift')
                    self.assertEqual(error.exception.status, 502)
                else:
                    result = backup.classify_description('crane lift')
                    self.assertEqual(result['confidence'], confidence)
                    self.assertEqual(result['provider'], 'jetstream')
                    self.assertEqual(result['model'], 'gpt-oss-120b')
