import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv(r"c:\Users\USER\Desktop\NLP\marathi-rag\.env")

# Test asking the exact question in Standard 8
payload = {
    "question": "भूमिगत पाठातील 'सामसूम होणे' या वाक्प्रचाराचा अर्थ काय आहे?",
    "standard": 8
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/ask",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        with open("scratch/ans_test.json", "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        print("Success! Response written to scratch/ans_test.json")
        print("Pages retrieved:", data.get("page_numbers"))
except Exception as e:
    print("Error:", e)
