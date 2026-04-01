"""Download large file from Google Drive using requests with session cookies."""
import os
import sys
import zipfile
import requests

FILE_ID = "1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf"
DEST_DIR = "/workspace/multiscale-pde-operators/datasets"
DEST_ZIP = os.path.join(DEST_DIR, "motor.zip")

def download_file_from_google_drive(file_id, destination):
    """Download a large file from Google Drive, handling the virus scan warning."""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    
    # First request to get confirm token
    print(f"Step 1: Initial request to Google Drive (file_id={file_id})...")
    response = session.get(URL, params={"id": file_id}, stream=True)
    print(f"  Status: {response.status_code}")
    print(f"  Headers Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    print(f"  Cookies: {dict(session.cookies)}")
    
    # Check if we got a direct download or need confirmation
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            print(f"  Found download_warning token: {token}")
            break
    
    if token is None:
        # Try to find the confirm token in the HTML body
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            text = response.text
            # Look for various confirm patterns
            import re
            # Pattern 1: confirm=XXXX in URL
            match = re.search(r'confirm=([0-9A-Za-z_-]+)', text)
            if match:
                token = match.group(1)
                print(f"  Found confirm token in HTML: {token}")
            
            # Pattern 2: uuid=XXXX
            match2 = re.search(r'uuid=([0-9A-Za-z_-]+)', text)
            if match2:
                uuid_token = match2.group(1)
                print(f"  Found uuid in HTML: {uuid_token}")
                if not token:
                    token = uuid_token
            
            # Pattern 3: /uc?export=download&confirm=X&id=
            match3 = re.search(r'/uc\?export=download&amp;confirm=([^&"]+)', text)
            if match3:
                token = match3.group(1)
                print(f"  Found confirm in redirect URL: {token}")
            
            # Pattern 4: action="/uc?..." form
            match4 = re.search(r'action="(/uc\?[^"]+)"', text)
            if match4:
                action_url = match4.group(1).replace("&amp;", "&")
                print(f"  Found form action URL: {action_url}")
                # Try the action URL directly
                full_url = "https://drive.google.com" + action_url
                print(f"  Trying form URL: {full_url}")
                response = session.get(full_url, stream=True)
                print(f"  Status: {response.status_code}")
                print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                content_length = response.headers.get('Content-Length', 'unknown')
                print(f"  Content-Length: {content_length}")
                
                if "application/zip" in response.headers.get("Content-Type", "") or \
                   "application/octet-stream" in response.headers.get("Content-Type", ""):
                    print("  Got binary response from form URL!")
                    save_response_content(response, destination)
                    return True
            
            if not token:
                # Save HTML for debugging
                with open("/tmp/gdrive_debug.html", "w") as f:
                    f.write(text[:5000])
                print("  Could not find any token. Saved first 5000 chars of HTML to /tmp/gdrive_debug.html")
                print(f"  HTML preview (first 500 chars):")
                print(text[:500])
                
                # Try with confirm=t (sometimes works)
                print("\n  Trying with confirm=t...")
                token = "t"
        else:
            # It might be the actual file
            print(f"  Got non-HTML response, saving directly...")
            save_response_content(response, destination)
            return True
    
    if token:
        print(f"\nStep 2: Downloading with confirm token={token}...")
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
        print(f"  Status: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        content_length = response.headers.get('Content-Length', 'unknown')
        print(f"  Content-Length: {content_length}")
        save_response_content(response, destination)
        return True
    
    return False


def save_response_content(response, destination, chunk_size=32768):
    """Save streaming response to file."""
    total = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                if total % (1024 * 1024) < chunk_size:
                    print(f"  Downloaded: {total / (1024*1024):.1f} MB", end="\r")
    print(f"\n  Total downloaded: {total / (1024*1024):.2f} MB")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    success = download_file_from_google_drive(FILE_ID, DEST_ZIP)
    
    if not success:
        print("\nFAILED: Could not download file.")
        sys.exit(1)
    
    # Check file
    file_size = os.path.getsize(DEST_ZIP)
    print(f"\nFile size: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
    
    # Check if it's actually a zip
    try:
        with zipfile.ZipFile(DEST_ZIP, 'r') as zf:
            names = zf.namelist()
            print(f"Valid ZIP file with {len(names)} entries")
            # Show first 10 entries
            for name in names[:10]:
                print(f"  {name}")
            if len(names) > 10:
                print(f"  ... and {len(names)-10} more entries")
            
            # Extract
            print(f"\nExtracting to {DEST_DIR}...")
            zf.extractall(DEST_DIR)
            print("Extraction complete!")
            
            # Check for Bnorm files
            import glob
            bnorm_files = glob.glob(os.path.join(DEST_DIR, "**/*Bnorm_matinfo.txt"), recursive=True)
            print(f"Found {len(bnorm_files)} Bnorm_matinfo.txt files")
            for f in bnorm_files[:5]:
                print(f"  {f}")
    except zipfile.BadZipFile:
        print("ERROR: Downloaded file is NOT a valid ZIP!")
        with open(DEST_ZIP, 'rb') as f:
            header = f.read(200)
        print(f"File header (hex): {header[:20].hex()}")
        if header[:15] == b'<!DOCTYPE html>':
            print("File is actually HTML (Google Drive error/permission page)")
            with open(DEST_ZIP, 'r', errors='replace') as f:
                text = f.read(1000)
            print(f"HTML preview: {text[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
