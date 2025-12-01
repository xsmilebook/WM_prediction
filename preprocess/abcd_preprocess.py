#!/usr/bin/env python3
"""
ABCD Data Preprocessing Script
This script handles ABCD-specific data preprocessing tasks:
1. Converts src_subject_id to subid format (remove '_' and add 'sub-' prefix)
2. Generates cognition_sublist.txt from nc_y_nihtb_baseline.csv
3. Generates pfactor_sublist.txt from Pfactor_score_wx.csv
"""

import pandas as pd
import os
import sys


def convert_to_subid(src_subject_id):
    """Convert src_subject_id to subid format: remove '_' and add 'sub-' prefix"""
    if pd.isna(src_subject_id):
        return None
    # Remove underscores and add 'sub-' prefix
    clean_id = str(src_subject_id).replace('_', '')
    return f"sub-{clean_id}"


def load_existing_sublist(sublist_path):
    """Load existing sublist.txt file"""
    try:
        with open(sublist_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: {sublist_path} not found. Returning empty set.")
        return set()


def process_abcd_cognition_data(nihtb_file, sublist_file, output_file):
    """Process ABCD cognition data and generate cognition sublist"""
    print(f"Processing ABCD cognition data from: {nihtb_file}")
    
    # Load existing sublist
    valid_subjects = load_existing_sublist(sublist_file)
    print(f"Loaded {len(valid_subjects)} subjects from existing sublist")
    
    # Load NIHTB baseline data
    try:
        df = pd.read_csv(nihtb_file)
        print(f"Loaded {len(df)} subjects from NIHTB baseline data")
    except FileNotFoundError:
        print(f"Error: {nihtb_file} not found")
        return False
    
    # Convert src_subject_id to subid format
    df['subid'] = df['src_subject_id'].apply(convert_to_subid)
    
    # Filter subjects with valid cognition scores
    # Requirements: nihtbx_fluidcomp_fc, nihtbx_cryst_fc, nihtbx_totalcomp_fc all != 0
    # and subid must exist in the existing sublist
    valid_cognition = df[
        (df['nihtbx_fluidcomp_fc'] != 0) & 
        (df['nihtbx_cryst_fc'] != 0) & 
        (df['nihtbx_totalcomp_fc'] != 0) &
        (df['subid'].notna()) &
        (df['subid'].isin(valid_subjects))
    ].copy()
    
    print(f"Found {len(valid_cognition)} subjects with valid cognition scores")
    
    # Sort and save cognition sublist
    cognition_subjects = sorted(valid_cognition['subid'].tolist())
    
    with open(output_file, 'w') as f:
        for subid in cognition_subjects:
            f.write(f"{subid}\n")
    
    print(f"Saved cognition sublist to: {output_file}")
    print(f"Total subjects in cognition sublist: {len(cognition_subjects)}")
    
    return True


def process_abcd_pfactor_data(pfactor_file, sublist_file, output_file):
    """Process ABCD P-factor data and generate pfactor sublist"""
    print(f"Processing ABCD P-factor data from: {pfactor_file}")
    
    # Load existing sublist
    valid_subjects = load_existing_sublist(sublist_file)
    print(f"Loaded {len(valid_subjects)} subjects from existing sublist")
    
    # Load P-factor data
    try:
        df = pd.read_csv(pfactor_file)
        print(f"Loaded {len(df)} subjects from P-factor data")
        print(f"Columns in P-factor data: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: {pfactor_file} not found")
        return False
    
    # Check for the correct subject ID column
    if 'subjectkey' in df.columns:
        subject_col = 'subjectkey'
    elif 'src_subject_id' in df.columns:
        subject_col = 'src_subject_id'
    else:
        # Try the first column as subject ID
        subject_col = df.columns[0]
        print(f"Warning: Using first column '{subject_col}' as subject ID")
    
    # Convert subject ID to subid format
    df['subid'] = df[subject_col].apply(convert_to_subid)
    
    # Filter subjects with valid P-factor scores
    # Requirements: General, Ext, ADHD, Int all != 0
    # and subid must exist in the existing sublist
    valid_pfactor = df[
        (df['General'] != 0) & 
        (df['Ext'] != 0) & 
        (df['ADHD'] != 0) &
        (df['Int'] != 0) &
        (df['subid'].notna()) &
        (df['subid'].isin(valid_subjects))
    ].copy()
    
    print(f"Found {len(valid_pfactor)} subjects with valid P-factor scores")
    
    # Sort and save P-factor sublist
    pfactor_subjects = sorted(valid_pfactor['subid'].tolist())
    
    with open(output_file, 'w') as f:
        for subid in pfactor_subjects:
            f.write(f"{subid}\n")
    
    print(f"Saved P-factor sublist to: {output_file}")
    print(f"Total subjects in P-factor sublist: {len(pfactor_subjects)}")
    
    return True


def add_subid_to_file(input_file, output_file=None):
    """Add subid column to any ABCD CSV file"""
    if output_file is None:
        output_file = input_file  # Overwrite original file
    
    print(f"Adding subid to: {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {os.path.basename(input_file)}")
        
        # Check if subid column already exists
        if 'subid' in df.columns:
            print(f"subid column already exists in {os.path.basename(input_file)}, skipping")
            return True
        
        # Find the subject ID column (usually src_subject_id or subjectkey)
        if 'src_subject_id' in df.columns:
            subject_col = 'src_subject_id'
        elif 'subjectkey' in df.columns:
            subject_col = 'subjectkey'
        else:
            # Try the first column as subject ID
            subject_col = df.columns[0]
            print(f"Warning: Using first column '{subject_col}' as subject ID in {os.path.basename(input_file)}")
        
        # Add subid column
        df['subid'] = df[subject_col].apply(convert_to_subid)
        
        # Save the updated file
        df.to_csv(output_file, index=False)
        print(f"Successfully added subid column to: {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        return False
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False


def process_all_abcd_files(data_dir):
    """Process all ABCD files to add subid column"""
    files_to_process = [
        "abcd_p_demo_baseline.csv",
        "abcd_y_lt_baseline.csv", 
        "mri_y_qc_motion_baseline.csv",
        "nc_y_nihtb_baseline.csv",
        "Pfactor_score_wx.csv"
    ]
    
    success_count = 0
    
    for filename in files_to_process:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            if add_subid_to_file(file_path):
                success_count += 1
            print()
        else:
            print(f"Warning: {filename} not found, skipping")
            print()
    
    return success_count


def main():
    """Main function for ABCD data preprocessing"""
    
    # Define data paths
    data_dir = r"d:\code\WM_prediction\data\ABCD\table"
    
    print("=" * 60)
    print("ABCD Data Preprocessing Script - Enhanced Version")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()
    
    # Step 1: Add subid column to all ABCD files
    print("STEP 1: Adding subid column to all ABCD CSV files")
    print("-" * 50)
    file_success_count = process_all_abcd_files(data_dir)
    
    # Step 2: Generate sublists (original functionality)
    print("STEP 2: Generating cognition and P-factor sublists")
    print("-" * 50)
    
    # Input files for sublist generation
    nihtb_file = os.path.join(data_dir, "nc_y_nihtb_baseline.csv")
    pfactor_file = os.path.join(data_dir, "Pfactor_score_wx.csv")
    sublist_file = os.path.join(data_dir, "ABCD_fc_sublist.txt")
    
    # Output files
    cognition_output = os.path.join(data_dir, "cognition_sublist.txt")
    pfactor_output = os.path.join(data_dir, "pfactor_sublist.txt")
    
    sublist_success_count = 0
    
    # Process cognition data
    if os.path.exists(nihtb_file):
        if process_abcd_cognition_data(nihtb_file, sublist_file, cognition_output):
            sublist_success_count += 1
        print()
    else:
        print(f"Warning: {nihtb_file} not found, skipping cognition processing")
        print()
    
    # Process P-factor data
    if os.path.exists(pfactor_file):
        if process_abcd_pfactor_data(pfactor_file, sublist_file, pfactor_output):
            sublist_success_count += 1
        print()
    else:
        print(f"Warning: {pfactor_file} not found, skipping P-factor processing")
        print()
    
    print("=" * 60)
    print(f"ABCD preprocessing completed:")
    print(f"  - File processing: {file_success_count}/5 files processed successfully")
    print(f"  - Sublist generation: {sublist_success_count}/2 sublists generated successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()