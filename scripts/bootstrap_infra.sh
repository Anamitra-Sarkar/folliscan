#!/usr/bin/env bash
# One-time infrastructure bootstrap for Folliscan.
# Creates: GitHub repo (+secrets), HF Space (+secrets), Render service, Vercel project.
# Secrets are sourced from local key files ONLY and never echoed into logs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEYS="/home/anamitra/Downloads/API_Keys_and_Secrets/api keys for new set of projects"
GH_REPO="Anamitra-Sarkar/folliscan"
OWNER="Anamitra-Sarkar"

read_key() { cat "$KEYS/$1"; }

HF_TOKEN="$(read_key bhumika-hf.txt)"
VERCEL_TOKEN="$(read_key vercel.txt)"
RENDER_KEY="$(read_key render-api.txt)"
GROQ_KEY="$(read_key groq_api.txt)"
FIREBASE_B64="$(base64 -w0 "$KEYS/cabbage-guard-firebase-adminsdk-fbsvc-07fc830b13.json")"

echo "== 1. GitHub repo =="
if ! gh repo view "$GH_REPO" >/dev/null 2>&1; then
  gh repo create "$GH_REPO" --private --source "$ROOT" --remote origin --push
else
  echo "repo exists"
  git -C "$ROOT" remote add origin "https://github.com/$GH_REPO.git" 2>/dev/null || true
fi

echo "== 2. GitHub Actions secrets =="
gh secret set HF_TOKEN        -R "$GH_REPO" --body "$HF_TOKEN"
gh secret set VERCEL_TOKEN    -R "$GH_REPO" --body "$VERCEL_TOKEN"
gh secret set RENDER_API_KEY  -R "$GH_REPO" --body "$RENDER_KEY"
gh secret set GROQ_API_KEY    -R "$GH_REPO" --body "$GROQ_KEY"

echo "== 3. Resolve HF username & create Space =="
HF_USER="$(curl -sf -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 | python3 -c 'import json,sys;print(json.load(sys.stdin)["name"])')"
echo "HF user: $HF_USER"

python3 - "$HF_USER" "$HF_TOKEN" "$FIREBASE_B64" "$GROQ_KEY" <<'PY'
import sys
from huggingface_hub import HfApi
user, token, fb_b64, groq = sys.argv[1:5]
api = HfApi(token=token)
space = f"{user}/folliscan-api"
try:
    api.create_repo(space, repo_type="space", space_sdk="docker", private=False,
                    exist_ok=True)
except Exception as e:
    print("space create:", e)
for k, v in {
    "FIREBASE_CREDENTIALS_B64": fb_b64,
    "FIREBASE_PROJECT_ID": "cabbage-guard",
    "GROQ_API_KEY": groq,
    "HF_TOKEN": token,
    "HF_MODEL_REPO": f"{user}/folliscan-model",
}.items():
    try:
        api.add_space_secret(space, k, v)
    except Exception as e:
        print("secret", k, e)
print("space ready:", space)
PY

gh variable set HF_SPACE_ID -R "$GH_REPO" --body "${HF_USER}/folliscan-api"

echo "== 4. Render service =="
python3 - <<'PY'
import json, os, urllib.request

key = os.environ["RENDER_KEY"]
owner = os.environ["RENDER_OWNER_ID"]

def req(path, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(
        f"https://api.render.com/v1{path}",
        data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

status, owners = req("/owners?limit=1")
if status != 200 or not owners:
    print("render owner lookup failed:", status, owners); raise SystemExit(1)
owner_id = owner or owners[0]["id"]

existing = req("/services?limit=20")[1]
svc = next((s["service"] for s in existing.get("services", [])
            if s["service"].get("name") == "folliscan-user-api"), None)

if svc:
    print("render service exists:", svc["id"])
else:
    payload = {
        "type": "web_service",
        "name": "folliscan-user-api",
        "runtime": "image",
        "image": {"ownerId": owner_id,
                  "registryCredentialId": "",
                  "imagePath": f"ghcr.io/{os.environ['GH_OWNER'].lower()}/folliscan-user-api:latest"},
        "serviceDetails": {"envSpecificDetails": {}},
        "envVars": [],  # filled after creation below
    }
    # try creating without registry credential first (public package expected)
    status, created = req("/services", "POST", payload)
    if status not in (200, 201):
        # retry without image block -> docker runtime from repo is unavailable;
        # surface error clearly
        print("render create failed:", status, created); raise SystemExit(1)
    svc = created
    print("render service created:", svc["service"]["id"])

sid = svc["id"]
# environment vars
for k, v in [("FIREBASE_PROJECT_ID", "cabbage-guard"),
             ("ENV", "production")]:
    req(f"/services/{sid}/env-vars/{k}", "PUT",
        {"value": v})
fb = os.environ["FIREBASE_B64"]
req(f"/services/{sid}/env-vars/FIREBASE_CREDENTIALS_B64", "PUT", {"value": fb})
print("RENDER_SERVICE_ID=" + sid)
PY
export GH_OWNER="$OWNER"
RENDER_SERVICE_ID="$(python3 - <<'PY'
import json, os, urllib.request
r = urllib.request.Request("https://api.render.com/v1/services?limit=20",
    headers={"Authorization": "Bearer " + os.environ["RENDER_KEY"]})
with urllib.request.urlopen(r) as resp:
    services = json.loads(resp.read())
print(next(s["service"]["id"] for s in services if s["service"]["name"] == "folliscan-user-api"))
PY
)"
gh secret set RENDER_SERVICE_ID -R "$GH_REPO" --body "$RENDER_SERVICE_ID"
echo "Render service id: $RENDER_SERVICE_ID"

echo "== 5. Vercel project =="
cd "$ROOT/frontend"
npx vercel link --yes --project folliscan --token "$VERCEL_TOKEN" 2>/dev/null \
  || npx vercel link --yes --token "$VERCEL_TOKEN"
PROJECT_JSON="$(cat .vercel/project.json)"
V_ORG="$(python3 -c "import json;print(json.load(open('.vercel/project.json'))['orgId'])")"
V_PROJ="$(python3 -c "import json;print(json.load(open('.vercel/project.json'))['projectId'])")"
gh secret set VERCEL_ORG_ID     -R "$GH_REPO" --body "$V_ORG"
gh secret set VERCEL_PROJECT_ID -R "$GH_REPO" --body "$V_PROJ"

echo "NOTE: add frontend env vars (NEXT_PUBLIC_*) once backends are live:"
echo "  vercel env add NEXT_PUBLIC_API_URL production"
echo "  vercel env add NEXT_PUBLIC_USER_API_URL production"
echo "  vercel env add NEXT_PUBLIC_FIREBASE_* production"

echo "== bootstrap done =="
