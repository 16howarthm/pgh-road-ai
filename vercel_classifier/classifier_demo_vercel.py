"""Small dependency-free classifier backend, shared by local and Vercel handlers."""
import json
import logging
import os
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PROMPT = json.loads((ROOT / 'prompt.json').read_text())
PROVIDERS = {
    'openai': {
        'label': 'OpenAI', 'key_env': 'OPENAI_API_KEY',
        'url': 'https://api.openai.com/v1/chat/completions',
        'default_model': 'gpt-4.1-mini',
        'models': ['gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini'],
    },
    'openrouter': {
        'label': 'OpenRouter', 'key_env': 'OPENROUTER_API_KEY',
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'default_model': 'meta-llama/llama-4-scout',
        'models': ['meta-llama/llama-4-scout', 'meta-llama/llama-4-maverick',
                   'deepseek/deepseek-v3.2', 'deepseek/deepseek-v4.1-flash',
                   'deepseek/deepseek-r1'],
    },
    'jetstream': {
        'label': 'Jetstream', 'key_env': 'JETSTREAM_API_KEY',
        'url': 'https://llm.jetstream-cloud.org/api/chat/completions',
        'default_model': 'llama-4-scout',
        'models': ['llama-4-scout', 'gpt-oss-120b', 'muse-glimmer'],
    },
}


def public_configuration():
    """Only expose key availability, never key values or arbitrary environment data."""
    return {
        'default_provider': 'openai',
        'providers': {
            name: {'label': value['label'], 'models': value['models'],
                   'default_model': value['default_model'],
                   'configured': bool(os.getenv(value['key_env']))}
            for name, value in PROVIDERS.items()
        },
    }


class ClassificationError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


def classify_description(description, provider='openai', model=None):
    if not isinstance(description, str) or not description.strip():
        raise ClassificationError('Enter a work description.', 400)
    description = description.strip()
    if len(description) > 4000:
        raise ClassificationError('Limit your description to 4,000 characters.', 400)
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ClassificationError('Choose OpenAI, OpenRouter, or Jetstream.', 400)
    config = PROVIDERS[provider]
    model = config['default_model'] if model is None else model
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}', model):
        raise ClassificationError('Enter a valid model ID for the selected provider.', 400)
    key = os.getenv(config['key_env'])
    if not key:
        raise ClassificationError(f"{config['label']} is not configured. The administrator must set {config['key_env']}.")
    payload = {'model': model, 'messages': [
        {'role': 'system', 'content': PROMPT['system_prompt'] + '\nThe reason field must contain 1–2 short sentences explaining the category using only evidence in the description.'},
        {'role': 'user', 'content': description}]}
    if provider == 'openai':
        payload['max_completion_tokens'] = 2048
        # Reasoning models may reject temperature. Known non-reasoning presets
        # retain the evaluated deterministic setting.
        if model.startswith(('gpt-4.1', 'gpt-4o')):
            payload['temperature'] = 0
    else:
        payload.update(temperature=0, max_tokens=2048)
    if provider == 'openrouter':
        payload['provider'] = {'allow_fallbacks': True}
    if provider == 'jetstream' and model == 'gpt-oss-120b':
        payload.update(reasoning_effort='low', response_format={'type': 'json_object'})
    request = Request(config['url'],
                      data=json.dumps(payload).encode(),
                      headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=25) as response:
            body = json.load(response)
    except HTTPError as exc:
        logging.warning('Classification provider=%s status=%s', provider, exc.code)
        message = f"{config['label']} is unavailable (HTTP {exc.code}). Please try again shortly or choose another provider."
        if exc.code in (401, 402, 403):
            message = f"{config['label']} rejected API access (HTTP {exc.code}). Check {config['key_env']}, credit balance, and spending limits."
        elif exc.code in (400, 404):
            message = f"{config['label']} rejected this model or request (HTTP {exc.code}). Choose another model supported by this provider."
        elif exc.code == 429:
            message = 'Too many requests. Please try again shortly.'
            # Inspect only known machine-readable codes; never expose or log
            # the provider body, which may contain submitted text or secrets.
            try:
                details = json.loads(exc.read(16384)).get('error', {})
                if isinstance(details, dict) and (
                    details.get('type') == 'insufficient_quota' or
                    details.get('code') in ('insufficient_quota', 'credit_balance_exhausted')
                ):
                    message = f"{config['label']} has insufficient API quota or credits. Check billing and spending limits, or choose another provider."
            except (ValueError, AttributeError, TypeError):
                pass
        raise ClassificationError(message, 429 if exc.code == 429 else 503) from None
    except (URLError, TimeoutError, socket.timeout):
        raise ClassificationError('The classification provider could not respond. Please try again shortly.') from None
    try:
        content = body['choices'][0]['message']['content']
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.S)
            result = json.loads(match.group()) if match else {}
        category = str(result.get('category', '')).strip().upper()
        if category not in PROMPT['categories']:
            raise ValueError('Invalid category')
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ClassificationError('The model returned an invalid category. Please try again.', 502) from None
    reason = result.get('reason')
    if not isinstance(reason, str) or not reason.strip():
        raise ClassificationError('The model did not return an explanation. Please try again.', 502)
    reason = ' '.join(reason.split())
    return {'category': category, 'reason': reason, 'model': model, 'provider': provider}


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/api/config':
            self.send_json(200, public_configuration())
            return
        static_routes = {
            '/': ('index.html', 'text/html; charset=utf-8'),
            '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
        }
        if self.path not in static_routes:
            self.send_json(404, {'error': 'Not found.'})
            return
        filename, content_type = static_routes[self.path]
        data = (ROOT / 'public' / filename).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != '/api/classify':
            self.send_json(404, {'error': 'Not found.'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20000:
                raise ClassificationError('Invalid request size.', 413)
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('Expected object')
        except ClassificationError as exc:
            self.send_json(exc.status, {'error': str(exc)})
            return
        except (ValueError, UnicodeDecodeError):
            self.send_json(400, {'error': 'Send a JSON object with a description.'})
            return
        try:
            self.send_json(200, classify_description(
                payload.get('description'), payload.get('provider', 'openai'), payload.get('model')
            ))
        except ClassificationError as exc:
            self.send_json(exc.status, {'error': str(exc)})
        except Exception as exc:
            logging.error('Classification failure type=%s', type(exc).__name__)
            self.send_json(500, {'error': 'Classification failed. Please try again.'})


def main():
    # Load local credentials only when running locally, never bundle .env.local.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parent / '.env.local', override=False)
    except ImportError:
        pass
    print('Classifier running at http://localhost:8000')
    ThreadingHTTPServer(('127.0.0.1', 8000), handler).serve_forever()


if __name__ == '__main__':
    main()
