import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

try:
    models = client.models.list().data
    print(f"Total models available: {len(models)}")
    for m in models:
        model_id = m.id
        print(f"Trying {model_id}...")
        try:
            res = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            print(f"SUCCESS with {model_id}!")
        except Exception as e:
            print(f"FAILED with {model_id}: {e}")
except Exception as e:
    print(f"Failed to list models: {e}")
