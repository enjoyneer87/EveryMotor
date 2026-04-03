#!/bin/bash
set -e
FILEID="1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf"
DEST="/workspace/multiscale-pde-operators/datasets/motor.zip"
OUTDIR="/workspace/multiscale-pde-operators/datasets/motor"

echo "Downloading motor dataset from Google Drive (file ID: $FILEID)..."

# Method 1: curl with confirm=t
curl -L -o "$DEST" \
  "https://drive.google.com/uc?export=download&id=${FILEID}&confirm=t" 2>&1

FILESIZE=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
FILETYPE=$(file "$DEST" | head -1)
echo "Downloaded: $FILESIZE bytes"
echo "File type: $FILETYPE"

if echo "$FILETYPE" | grep -q "Zip archive"; then
    echo "Valid zip file. Extracting..."
    cd "$OUTDIR"
    unzip -o "$DEST"
    echo "Done extracting."
    NTXT=$(find "$OUTDIR" -name "*Bnorm_matinfo.txt" | wc -l)
    echo "Found $NTXT Bnorm_matinfo.txt files"
    find "$OUTDIR" -name "*Bnorm_matinfo.txt" | head -5
elif echo "$FILETYPE" | grep -q "HTML"; then
    echo "Got HTML page instead of zip. Trying with cookies..."
    # Save cookies
    wget -q --save-cookies /tmp/gdrive_cookies.txt --keep-session-cookies \
        --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=$FILEID" \
        -O /tmp/gdrive_confirm.html
    
    # Extract confirm token (uuid format used by newer Google Drive)
    CONFIRM=$(cat /tmp/gdrive_confirm.html | grep -oP 'uuid=[^"&]+' | head -1 | cut -d= -f2)
    echo "UUID: $CONFIRM"
    
    if [ -z "$CONFIRM" ]; then
        CONFIRM=$(cat /tmp/gdrive_confirm.html | grep -oP 'confirm=[^"&]+' | head -1 | cut -d= -f2)
        echo "Confirm: $CONFIRM"
    fi
    
    if [ -n "$CONFIRM" ]; then
        wget --load-cookies /tmp/gdrive_cookies.txt --no-check-certificate \
            "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$FILEID" \
            -O "$DEST" 2>&1 | tail -5
        FILESIZE2=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
        FILETYPE2=$(file "$DEST" | head -1)
        echo "Re-downloaded: $FILESIZE2 bytes, Type: $FILETYPE2"
    else
        echo "Could not extract confirmation token."
        echo "HTML content preview:"
        head -50 "$DEST"
    fi
else
    echo "Unknown file type: $FILETYPE"
    head -c 500 "$DEST"
fi
