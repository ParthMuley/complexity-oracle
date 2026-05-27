"""Quick smoke test against the live Cloud Run endpoint."""
import json
import urllib.request

URL = "https://complexity-oracle-1049073599817.us-central1.run.app/analyze"
API_KEY = input("Paste your Anthropic API key: ").strip()

CODE = """\
def find(data, target):
    for x in data:
        if x == target:
            return True
    return False
"""

payload = json.dumps({"code": CODE, "no_agent": False}).encode()

req = urllib.request.Request(
    URL,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "X-Anthropic-API-Key": API_KEY,
    },
    method="POST",
)

import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

with urllib.request.urlopen(req, context=ctx) as resp:
    result = json.loads(resp.read())

print(json.dumps(result, indent=2))
