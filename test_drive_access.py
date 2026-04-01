"""Test all three dataset file IDs from Merantix download_data.py"""
import os, json, requests

adc_path = os.path.join(os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json")
with open(adc_path) as f:
    creds = json.load(f)

token_resp = requests.post("https://oauth2.googleapis.com/token", data={
    "client_id": creds["client_id"],
    "client_secret": creds["client_secret"],
    "refresh_token": creds["refresh_token"],
    "grant_type": "refresh_token",
})
access_token = token_resp.json()["access_token"]

headers = {
    "Authorization": f"Bearer {access_token}",
    "x-goog-user-project": creds.get("quota_project_id", "ai-cad"),
}

file_ids = {
    "electromagnetics": "1x8xddySJO9sEnyc_qBUYcDDrSA7z5sV2",
    "darcy": "1OYoXIFhRbY_TJmce0cb0nSs5n9vO8z68",
    "motor": "1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf",
}

for name, fid in file_ids.items():
    # try normal
    url = f"https://www.googleapis.com/drive/v3/files/{fid}?fields=name,size,mimeType"
    resp = requests.get(url, headers=headers)
    print(f"\n{name} ({fid[:15]}...):")
    print(f"  Normal: {resp.status_code} - {resp.text[:200]}")
    
    # try with supportsAllDrives
    url2 = f"https://www.googleapis.com/drive/v3/files/{fid}?fields=name,size,mimeType&supportsAllDrives=true"
    resp2 = requests.get(url2, headers=headers)
    print(f"  Shared: {resp2.status_code} - {resp2.text[:200]}")
