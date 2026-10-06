import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import openrouter_experiment as experiment
from road_ai import DEFAULT_CATEGORIES, FAILED_PREDICTION


class OpenRouterExperimentTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({
            'permit_id': [f'p{i}' for i in range(8)],
            'work_description': ['man lift'] * 8,
            'work_type': ['MACHINERY'] * 8,
        })

    def test_adapter_maps_models_and_limits_reasoning(self):
        from unittest.mock import MagicMock
        base = MagicMock()
        client = experiment.OpenRouterPromptClient('reviewed prompt', base_client=base)
        client.create(model='gpt-oss-120b', messages=[{'role': 'system', 'content': 'old'},
                      {'role': 'user', 'content': 'full description'}], reasoning_effort='low')
        request = base.chat.completions.create.call_args.kwargs
        self.assertEqual(request['model'], 'openai/gpt-oss-120b')
        self.assertEqual(request['messages'][0]['content'], 'reviewed prompt')
        self.assertEqual(request['messages'][1]['content'], 'full description')
        self.assertNotIn('reasoning_effort', request)
        self.assertEqual(request['max_tokens'], 1024)
        self.assertEqual(request['extra_body']['reasoning']['effort'], 'low')
        client.create(model='llama-4-scout', messages=[{'role': 'user', 'content': 'man lift'}],
                      response_format={'type': 'json_object'})
        scout_request = base.chat.completions.create.call_args.kwargs
        self.assertNotIn('response_format', scout_request)
        self.assertEqual(scout_request['model'], 'meta-llama/llama-4-scout')

    def test_concurrency_resume_and_failure_retry_preserve_order(self):
        active = peak = calls = 0
        lock = threading.Lock()
        def fake_run(frame, *args, **kwargs):
            nonlocal active, peak, calls
            with lock:
                calls += 1
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return pd.DataFrame([{'permit_id': frame.iloc[0].permit_id,
                                  'predicted': 'MACHINERY', 'error': None}])
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, 'run_experiment', fake_run):
            checkpoint = Path(directory) / 'run.jsonl'
            checkpoint.write_text(json.dumps({'permit_id': 'p1', 'predicted': FAILED_PREDICTION, 'error': 'old failure'}) + '\n')
            result = experiment.run_checkpointed(self.test, 'work_type', 'llama-4-scout', DEFAULT_CATEGORIES, object(), checkpoint, max_workers=3)
            self.assertEqual(result.permit_id.tolist(), self.test.permit_id.tolist())
            self.assertEqual(calls, 8)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 3)
            experiment.run_checkpointed(self.test, 'work_type', 'llama-4-scout', DEFAULT_CATEGORIES, object(), checkpoint)
            self.assertEqual(calls, 8)

    def test_preflight_failure_stops_bulk(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, 'run_experiment') as run:
            run.return_value = pd.DataFrame([{'permit_id': 'p0', 'predicted': FAILED_PREDICTION, 'error': '403 forbidden'}])
            with self.assertRaisesRegex(RuntimeError, 'preflight failed'):
                experiment.run_checkpointed(self.test, 'work_type', 'llama-4-scout', DEFAULT_CATEGORIES, object(), Path(directory) / 'run.jsonl')
            self.assertEqual(run.call_count, 1)

    def test_circuit_breaker_stops_scheduling(self):
        calls = 0
        def run(frame, *args, **kwargs):
            nonlocal calls
            calls += 1
            return pd.DataFrame([{'permit_id': frame.iloc[0].permit_id,
                'predicted': 'MACHINERY' if calls == 1 else FAILED_PREDICTION,
                'error': None if calls == 1 else 'server unavailable'}])
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, 'run_experiment', run):
            with self.assertRaisesRegex(RuntimeError, 'five consecutive failures'):
                experiment.run_checkpointed(self.test, 'work_type', 'llama-4-scout', DEFAULT_CATEGORIES, object(), Path(directory) / 'run.jsonl', max_workers=1)
            self.assertEqual(calls, 6)

if __name__ == '__main__':
    unittest.main()
