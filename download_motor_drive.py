"""Download motor.zip from Google Drive using browser cookies for authentication."""
import os
import sys

# Close Chrome first if needed - cookies may be locked
FILE_ID = "1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf"
OUTPUT = r"D:\KDH\NvidiaNemo\motor_fsg.zip"

print(f"Downloading Google Drive file {FILE_ID}")
print(f"Output: {OUTPUT}")

# Method 1: gdown with use_cookies=True
try:
    import gdown
    print("\nTrying gdown with use_cookies=True ...")
    gdown.download(
        id=FILE_ID,
        output=OUTPUT,
        quiet=False,
        use_cookies=True,
    )
    if os.path.exists(OUTPUT) and os.path.getsize(OUTPUT) > 1000:
        print(f"\nSuccess! File size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")
        sys.exit(0)
except Exception as e:
    print(f"gdown use_cookies failed: {e}")

# Method 2: requests with browser cookies
try:
    import browser_cookie3
    import requests
    
    print("\nTrying requests with Chrome cookies ...")
    cj = browser_cookie3.chrome(domain_name='.google.com')
    
    # Google Drive download URL
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    session = requests.Session()
    session.cookies = cj
    
    # First request may return a confirmation page
    resp = session.get(url, stream=True)
    print(f"  Initial response: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
    
    # Check if we got a download confirmation page
    if 'text/html' in resp.headers.get('Content-Type', ''):
        # Extract confirm token
        import re
        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                confirm = value
                break
        else:
            # Try to find confirm in page
            match = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
            confirm = match.group(1) if match else 't'
        
        url_confirmed = f"{url}&confirm={confirm}"
        print(f"  Confirming download...")
        resp = session.get(url_confirmed, stream=True)
        print(f"  Confirmed response: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
    
    if resp.status_code == 200 and 'application' in resp.headers.get('Content-Type', ''):
        total = int(resp.headers.get('Content-Length', 0))
        print(f"  Downloading {total / 1024 / 1024:.1f} MB ...")
        
        downloaded = 0
        with open(OUTPUT, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192 * 16):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (10 * 1024 * 1024) < 8192 * 16:
                        print(f"  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB")
        
        print(f"\nSuccess! File size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")
        sys.exit(0)
    else:
        print(f"  Failed: status={resp.status_code}")
        print(f"  Response text (first 500):\n{resp.text[:500]}")
except Exception as e:
    print(f"Method 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Method 3: Use Google Drive API with OAuth2
try:
    print("\nTrying Google Drive API with gcloud OAuth token ...")
    import subprocess
    result = subprocess.run(
        ['gcloud', 'auth', 'print-access-token'],
        capture_output=True, text=True,
        env={**os.environ, 'PATH': os.environ.get('PATH', '') + r';C:\Users\moa\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin'}
    )
    token = result.stdout.strip()
    if len(token) > 50:
        import requests
        # Enable Google Drive API scope - gcloud token may not have it
        # Try anyway
        headers = {'Authorization': f'Bearer {token}'}
        url = f'https://www.googleapis.com/drive/v3/files/{FILE_ID}?alt=media'
        resp = requests.get(url, headers=headers, stream=True)
        print(f"  Response: {resp.status_code}")
        if resp.status_code == 200:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(OUTPUT, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            print(f"Success! File size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")
            sys.exit(0)
        else:
            print(f"  Failed: {resp.text[:300]}")
except Exception as e:
    print(f"Method 3 failed: {e}")

print("\n\nAll methods failed. Please try:")
print("1. Open in browser: https://drive.google.com/uc?id=1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf")
print(f"2. Download manually to: {OUTPUT}")
