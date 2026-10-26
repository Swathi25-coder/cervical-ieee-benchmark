import os
import zipfile
import shutil
from pathlib import Path

def unpack_from_downloads():
    downloads = Path("C:/Users/User/Downloads")
    data_dir = Path("data")
    sipak_dir = data_dir / "sipakmed"
    herlev_dir = data_dir / "herlev"
    sipak_dir.mkdir(parents=True, exist_ok=True)
    herlev_dir.mkdir(parents=True, exist_ok=True)
    
    print("Checking Downloads folder for SIPaKMeD and Herlev zip files...")
    zips = list(downloads.glob("*.zip"))
    
    found_any = False
    for z in zips:
        zname = z.name.lower()
        if "sipak" in zname:
            print(f"Found SIPaKMeD zip: {z.name}. Unpacking into {sipak_dir}...")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(sipak_dir)
            print("SIPaKMeD unpacked successfully!")
            found_any = True
        elif "herlev" in zname:
            print(f"Found Herlev zip: {z.name}. Unpacking into {herlev_dir}...")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(herlev_dir)
            print("Herlev unpacked successfully!")
            found_any = True
            
    if not found_any:
        print("No SIPaKMeD or Herlev zip files found in Downloads yet.")
        print("Once you download them, simply run this script again!")
    else:
        print("Done unpacking! Next step: Run 'python src/prep.py'")

if __name__ == "__main__":
    unpack_from_downloads()
