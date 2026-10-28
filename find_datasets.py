import os
from pathlib import Path

def search_for_datasets():
    print("Searching for SIPaKMeD and Herlev datasets on your computer...")
    potential_roots = [
        Path("D:/"),
        Path("C:/Users/User/Downloads"),
        Path("C:/Users/User/Desktop"),
        Path("C:/Users/User/Documents"),
    ]
    
    found = []
    for root in potential_roots:
        if not root.exists():
            continue
        try:
            # Check immediate children and 1 level down
            for p in root.iterdir():
                name = p.name.lower()
                if "sipak" in name or "herlev" in name or "cervical" in name:
                    found.append(p)
                elif p.is_dir():
                    try:
                        for sub in p.iterdir():
                            sub_name = sub.name.lower()
                            if "sipak" in sub_name or "herlev" in sub_name:
                                found.append(sub)
                    except Exception:
                        pass
        except Exception as e:
            pass
            
    if found:
        print("Found possible dataset files/folders:")
        for f in found:
            print(f" - {f}")
    else:
        print("No local SIPaKMeD or Herlev files found in Downloads, Desktop, or D: drive.")
    return found

if __name__ == "__main__":
    search_for_datasets()
