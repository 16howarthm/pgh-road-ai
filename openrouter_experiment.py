"""Concurrent, resumable OpenRouter classification for the local experiment."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from openai import OpenAI
from road_ai import FAILED_PREDICTION, run_experiment

MODEL_IDS = {
    'llama-4-scout': 'meta-llama/llama-4-scout',
    'gpt-oss-120b': 'openai/gpt-oss-120b',
    'muse-glimmer': 'meta/muse-glimmer-30b',
}


class OpenRouterPromptClient:
    def __init__(self, system_prompt, api_key=None, timeout=45, base_client=None):
        key = api_key or os.getenv('OPENROUTER_API_KEY') or os.getenv('openrouter_api_key')
        if base_client is None and not key:
            raise ValueError('Set OPENROUTER_API_KEY in .env.local, then rerun setup.')
        self.base_client = base_client or OpenAI(
            base_url='https://openrouter.ai/api/v1', api_key=key,
            timeout=timeout, max_retries=1,
        )
        self.system_prompt = system_prompt
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        model = request['model']
        request['model'] = MODEL_IDS[model]
        request['messages'] = [
            {**m, 'content': self.system_prompt} if m['role'] == 'system' else m.copy()
            for m in request['messages']
        ]
        request.pop('reasoning_effort', None)
        # Scout's non-BYOK endpoints do not accept response_format.
        if model == 'llama-4-scout':
            request.pop('response_format', None)
        else:
            request['response_format'] = {'type': 'json_object'}
        request['max_tokens'] = 1024 if model in ('gpt-oss-120b', 'muse-glimmer') else 256
        request['extra_body'] = {'provider': {'sort': 'latency'}}
        if model in ('gpt-oss-120b', 'muse-glimmer'):
            request['extra_body']['reasoning'] = {'effort': 'low', 'exclude': True}
        return self.base_client.chat.completions.create(**request)


def run_checkpointed(test, label_col, model, categories, client, checkpoint,
                     max_workers=8, on_progress=None):
    """Reuse successes, retry failures, and write completed rows on the main thread.

    One uncached record is tested before scheduling the remaining requests.
    Five consecutive failures stop further scheduling. Latest checkpoint row wins.
    """
    if max_workers < 1:
        raise ValueError('max_workers must be positive')
    checkpoint = Path(checkpoint)
    cached = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row['permit_id']] = row
    if not set(cached) <= set(test.permit_id):
        raise ValueError('Checkpoint contains unexpected records')
    successful = {pid: row for pid, row in cached.items()
                  if row.get('predicted') in categories and not row.get('error')}
    remaining = test[~test.permit_id.isin(successful)].copy()
    if on_progress:
        on_progress(len(successful))

    def classify(frame):
        return run_experiment(frame, label_col, model=model, categories=categories,
                              client=client).iloc[0].to_dict()

    def save(row):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        cached[row['permit_id']] = row
        if on_progress:
            on_progress(1)

    if not remaining.empty:
        first = classify(remaining.iloc[:1])
        save(first)
        if first['predicted'] == FAILED_PREDICTION:
            raise RuntimeError(f"{model} preflight failed; bulk run stopped: {first['error']}")
        frames = iter(remaining.iloc[i:i+1] for i in range(1, len(remaining)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = set()
            def fill():
                while len(pending) < max_workers:
                    frame = next(frames, None)
                    if frame is None:
                        break
                    pending.add(executor.submit(classify, frame))
            fill()
            consecutive_failures = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    save(row)
                    consecutive_failures = consecutive_failures + 1 if row['predicted'] == FAILED_PREDICTION else 0
                    if consecutive_failures >= 5:
                        # Drain already running requests to retain their results.
                        for other in pending:
                            if not other.cancel():
                                save(other.result())
                        raise RuntimeError(f'{model}: five consecutive failures; stopped. Check checkpoint errors before retrying.')
                fill()
    return pd.DataFrame([cached[pid] for pid in test.permit_id])
