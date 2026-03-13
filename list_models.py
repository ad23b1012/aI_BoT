import os, requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get('GOOGLE_API_KEY')
res = requests.get(f'https://generativelanguage.googleapis.com/v1beta/models?key={key}').json()
models = []
for m in res.get('models', []):
    if 'embedContent' in m.get('supportedGenerationMethods', []):
        models.append(m['name'])
print("Available Embedding Models:")
for name in models:
    print("-", name)
