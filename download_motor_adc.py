"""Download motor.zip using Application Default Credentials (has Drive scope)."""
import os, sys, json

FILE_ID = "1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf"
OUTPUT = r"D:\KDH\NvidiaNemo\motor_fsg.zip"

# Load ADC credentials
adc_path = os.path.join(os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json")
print(f"ADC path: {adc_path}")

with open(adc_path) as f:
    creds = json.load(f)

print(f"Client ID: {creds.get('client_id', '?')[:20]}...")
print(f"Type: {creds.get('type', '?')}")

# Get access token from refresh token
import requests

token_resp = requests.post("https://oauth2.googleapis.com/token", data={
    "client_id": creds["client_id"],
    "client_secret": creds["client_secret"],
    "refresh_token": creds["refresh_token"],
    "grant_type": "refresh_token",
})
token_data = token_resp.json()

if "access_token" not in token_data:
    print(f"Token refresh failed: {token_data}")
    sys.exit(1)

access_token = token_data["access_token"]
print(f"Access token obtained (len={len(access_token)})")

# Get file metadata
quota_project = creds.get("quota_project_id", "ai-cad")
print(f"Quota project: {quota_project}")
headers = {
    "Authorization": f"Bearer {access_token}",
    "x-goog-user-project": quota_project,
}
meta_resp = requests.get(
    f"https://www.googleapis.com/drive/v3/files/{FILE_ID}?fields=name,size,mimeType",
    headers=headers,
)
print(f"Metadata response: {meta_resp.status_code}")
if meta_resp.status_code == 200:
    meta = meta_resp.json()
    total_size = int(meta.get("size", 0))
    print(f"File: {meta.get('name')}")
    print(f"Size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Type: {meta.get('mimeType')}")
else:
    print(f"Metadata error: {meta_resp.text[:500]}")
    total_size = 0

# Download file
print(f"\nDownloading to {OUTPUT} ...")
dl_resp = requests.get(
    f"https://www.googleapis.com/drive/v3/files/{FILE_ID}?alt=media",
    headers=headers,
    stream=True,
)
print(f"Download response: {dl_resp.status_code}, Content-Type: {dl_resp.headers.get('Content-Type','?')}")

if dl_resp.status_code == 200:
    cl = int(dl_resp.headers.get('Content-Length', total_size))
    downloaded = 0
    with open(OUTPUT, 'wb') as f:
        for chunk in dl_resp.iter_content(chunk_size=131072):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if cl > 0 and downloaded % (5 * 1024 * 1024) < 131072:
                    pct = downloaded / cl * 100
                    print(f"  {downloaded / 1024 / 1024:.1f} / {cl / 1024 / 1024:.1f} MB ({pct:.0f}%)")
    
    fsize = os.path.getsize(OUTPUT)
    print(f"\nDone! File size: {fsize / 1024 / 1024:.1f} MB")
    
    if fsize < 10000:
        print("WARNING: File is very small, may be an error page")
        with open(OUTPUT, 'r', errors='ignore') as f:
            print(f.read()[:500])
else:
    print(f"Download failed: {dl_resp.status_code}")
    print(dl_resp.text[:500])
