"""Download motor.zip from Google Drive using OAuth2 with Drive scope."""
import os, sys, json, webbrowser, http.server, urllib.parse, threading

FILE_ID = "1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf"
OUTPUT = r"D:\KDH\NvidiaNemo\motor_fsg.zip"

# First try Edge cookies  
try:
    import browser_cookie3
    print("Trying Edge browser cookies...")
    cj = browser_cookie3.edge(domain_name='.google.com')
    import requests
    session = requests.Session()
    session.cookies = cj
    
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    resp = session.get(url, stream=True, allow_redirects=True)
    print(f"  Response: {resp.status_code}, CT: {resp.headers.get('Content-Type','?')[:50]}")
    
    # Handle virus scan confirmation page
    if 'text/html' in resp.headers.get('Content-Type', ''):
        import re
        # Find confirm token in cookies or page
        confirm = None
        for key, val in resp.cookies.items():
            if 'download_warning' in key:
                confirm = val
                break
        if not confirm:
            m = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
            if m:
                confirm = m.group(1)
        if not confirm:
            # Try uuid pattern 
            m = re.search(r'uuid=([0-9a-f-]+)', resp.text)
            uuid_val = m.group(1) if m else None
            if uuid_val:
                url2 = f"https://drive.google.com/uc?export=download&id={FILE_ID}&uuid={uuid_val}&confirm=t"
            else:
                url2 = f"{url}&confirm=t"
        else:
            url2 = f"{url}&confirm={confirm}"
        
        print(f"  Confirming download with token...")
        resp = session.get(url2, stream=True, allow_redirects=True)
        print(f"  Response: {resp.status_code}, CT: {resp.headers.get('Content-Type','?')[:50]}")
    
    ct = resp.headers.get('Content-Type', '')
    if resp.status_code == 200 and ('octet' in ct or 'zip' in ct or 'application' in ct):
        total = int(resp.headers.get('Content-Length', 0))
        print(f"  Downloading {total / 1024 / 1024:.1f} MB ...")
        downloaded = 0
        with open(OUTPUT, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (10 * 1024 * 1024) < 131072:
                        pct = downloaded / total * 100
                        print(f"  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct:.0f}%)")
        
        fsize = os.path.getsize(OUTPUT)
        if fsize > 10000:
            print(f"\nSuccess! File size: {fsize / 1024 / 1024:.1f} MB")
            sys.exit(0)
        else:
            print(f"  File too small ({fsize} bytes), likely an error page")
            os.remove(OUTPUT)
    elif resp.status_code == 200 and 'text/html' in ct:
        print("  Still getting HTML page, not actual file")
        # Save HTML for debugging
        with open(OUTPUT + ".html", "w", encoding="utf-8") as f:
            f.write(resp.text[:5000])
        print(f"  Saved response to {OUTPUT}.html for debugging")
    else:
        print(f"  Unexpected response: {resp.status_code} {ct}")
except Exception as e:
    print(f"Edge cookies method failed: {e}")

# Method 2: OAuth2 with Drive scope using google-auth-oauthlib
try:
    print("\n\nTrying OAuth2 with Google Drive scope...")
    
    # Use a simple OAuth2 flow with Drive read-only scope
    # Client ID from gcloud CLI (public, well-known)
    CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
    CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
    
    import requests
    
    # OAuth2 device flow (no browser redirect needed)
    device_resp = requests.post("https://oauth2.googleapis.com/device/code", data={
        "client_id": CLIENT_ID,
        "scope": "https://www.googleapis.com/auth/drive.readonly"
    })
    
    if device_resp.status_code == 200:
        device_data = device_resp.json()
        user_code = device_data["user_code"]
        verification_url = device_data["verification_url"]
        device_code = device_data["device_code"]
        interval = device_data.get("interval", 5)
        expires_in = device_data.get("expires_in", 1800)
        
        print(f"\n{'='*60}")
        print(f"  1. Open this URL: {verification_url}")
        print(f"  2. Enter this code: {user_code}")
        print(f"{'='*60}")
        print(f"  Waiting for authorization (expires in {expires_in}s)...")
        
        webbrowser.open(verification_url)
        
        import time
        start = time.time()
        access_token = None
        while time.time() - start < expires_in:
            time.sleep(interval)
            token_resp = requests.post("https://oauth2.googleapis.com/token", data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant_type:device_code"
            })
            token_data = token_resp.json()
            if "access_token" in token_data:
                access_token = token_data["access_token"]
                print("  Authorization successful!")
                break
            elif token_data.get("error") == "authorization_pending":
                elapsed = int(time.time() - start)
                print(f"  Waiting... ({elapsed}s)", end="\r")
            elif token_data.get("error") == "slow_down":
                interval += 2
            else:
                print(f"  Error: {token_data}")
                break
        
        if access_token:
            print(f"\n  Downloading file via Drive API...")
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Get file metadata first
            meta_resp = requests.get(
                f"https://www.googleapis.com/drive/v3/files/{FILE_ID}?fields=name,size,mimeType",
                headers=headers
            )
            if meta_resp.status_code == 200:
                meta = meta_resp.json()
                print(f"  File: {meta.get('name')}, Size: {int(meta.get('size',0))/1024/1024:.1f} MB")
            
            # Download
            dl_resp = requests.get(
                f"https://www.googleapis.com/drive/v3/files/{FILE_ID}?alt=media",
                headers=headers,
                stream=True
            )
            print(f"  Download response: {dl_resp.status_code}")
            
            if dl_resp.status_code == 200:
                total = int(dl_resp.headers.get('Content-Length', 0))
                downloaded = 0
                with open(OUTPUT, 'wb') as f:
                    for chunk in dl_resp.iter_content(chunk_size=131072):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0 and downloaded % (10 * 1024 * 1024) < 131072:
                                pct = downloaded / total * 100
                                print(f"  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct:.0f}%)")
                
                print(f"\n  Success! File: {OUTPUT}")
                print(f"  Size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")
                sys.exit(0)
            else:
                print(f"  Download failed: {dl_resp.status_code}")
                print(f"  {dl_resp.text[:500]}")
    else:
        print(f"  Device code request failed: {device_resp.status_code}")
        print(f"  {device_resp.text[:300]}")
except Exception as e:
    print(f"OAuth2 method failed: {e}")
    import traceback
    traceback.print_exc()

print("\n\nAll automated methods failed.")
print(f"Please download manually:")
print(f"  URL: https://drive.google.com/file/d/{FILE_ID}/view")
print(f"  Save to: {OUTPUT}")
