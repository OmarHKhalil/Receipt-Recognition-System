import json

def safe_json(text):
    try:
        return json.loads(text)
    except:
        return None