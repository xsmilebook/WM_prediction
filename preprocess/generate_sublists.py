import csv
import sys
from pathlib import Path

DATA_ROOT = Path(r"d:\code\WM_prediction\data")
DATASETS = ["ABCD", "CCNP", "EFNY", "HCPD", "PNC"]

def main():
    for name in DATASETS:
        dataset_dir = DATA_ROOT / name
        csv_path = dataset_dir / "table" / "rest_fd_summary.csv"
        # Output sublist.txt in the dataset root directory (e.g. data/ABCD/sublist.txt)
        out_txt = dataset_dir / "table" / "sublist.txt"
        
        if not csv_path.exists():
            print(f"Skipping {name}: {csv_path} not found")
            continue

        # Special handling for CCNP: exclude subjects < 6 years old
        excluded_ccnp_ids = set()
        if name == "CCNP":
            demo_csv = dataset_dir / "table" / "CCNP_demo.csv"
            if demo_csv.exists():
                try:
                    # Use utf-8-sig to handle potential BOM
                    with demo_csv.open("r", encoding="utf-8-sig") as f:
                        dreader = csv.DictReader(f)
                        # Standard DictReader should work if headers are clean.
                        # Based on cat output: subID,age,sex
                        for row in dreader:
                            sid = row.get("subID")
                            age_str = row.get("age")
                            if sid and age_str:
                                try:
                                    age = float(age_str)
                                    if age < 6:
                                        excluded_ccnp_ids.add(sid.strip())
                                except ValueError:
                                    pass # Skip invalid age
                    print(f"  Loaded CCNP demo exclusions (age < 6): {len(excluded_ccnp_ids)} subjects")
                except Exception as e:
                    print(f"  Error reading CCNP demo file: {e}")
            else:
                print(f"  Warning: CCNP demo file not found at {demo_csv}")
            
        print(f"Processing {name}...")
        valid_subs = set()
        
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if "valid_subject" not in fieldnames:
                    print(f"  Warning: 'valid_subject' column not found in {csv_path}")
                    # Try to see if there is any other indicator? 
                    # But for now rely on valid_subject as requested.
                    continue
                
                # Check for subid column
                subid_col = "subid"
                if "subid" not in fieldnames:
                    # Maybe it is named differently?
                    # But previous scripts used "subid".
                    if "Subject" in fieldnames:
                        subid_col = "Subject"
                    elif "subject_id" in fieldnames:
                        subid_col = "subject_id"
                    else:
                        print(f"  Warning: 'subid' column not found in {csv_path}. Available: {fieldnames}")
                        continue

                for row in reader:
                    val = row.get("valid_subject")
                    # Check for "1" (string) or 1 (int) just in case, though csv reads as str
                    if val == "1":
                        sid = row.get(subid_col)
                        if sid:
                            s_clean = sid.strip()
                            if name == "CCNP" and s_clean in excluded_ccnp_ids:
                                continue
                            valid_subs.add(s_clean)
            
            if not valid_subs:
                print(f"  No valid subjects found for {name}")
            
            # Sort for consistency
            sorted_subs = sorted(list(valid_subs))
            
            with out_txt.open("w", encoding="utf-8") as f:
                for s in sorted_subs:
                    f.write(s + "\n")
            
            print(f"  Generated {out_txt} with {len(sorted_subs)} unique subjects")
            
        except Exception as e:
            print(f"  Error processing {name}: {e}")

if __name__ == "__main__":
    main()
