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


class ClassificationError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


def classify_description(description):
    if not isinstance(description, str) or not description.strip():
        raise ClassificationError('Enter a work description.', 400)
    description = description.strip()
    if len(description) > 4000:
        raise ClassificationError('Limit your description to 4,000 characters.', 400)
    key = os.getenv('OPENROUTER_API_KEY')
    if not key:
        raise ClassificationError('Classification is not configured. The administrator must set OPENROUTER_API_KEY.')
    model = os.getenv('CLASSIFIER_MODEL', 'meta-llama/llama-4-scout')
    payload = {'model': model, 'messages': [
        {'role': 'system', 'content': PROMPT['system_prompt']},
        {'role': 'user', 'content': description}],
        'temperature': 0, 'max_tokens': 300,
        'provider': {'allow_fallbacks': True}}
    request = Request('https://openrouter.ai/api/v1/chat/completions',
                      data=json.dumps(payload).encode(),
                      headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=25) as response:
            body = json.load(response)
    except HTTPError as exc:
        logging.warning('Classification provider status=%s', exc.code)
        message = 'The classification provider is unavailable. Please try again shortly.'
        if exc.code in (401, 402, 403):
            message = 'The administrator must check the provider API key and credit balance.'
        elif exc.code == 429:
            message = 'Too many requests. Please try again shortly.'
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
    return {'category': category, 'model': model}


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
        if self.path != '/':
            self.send_json(404, {'error': 'Not found.'})
            return
        data = (ROOT / 'public' / 'index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
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
            self.send_json(200, classify_description(payload.get('description')))
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
