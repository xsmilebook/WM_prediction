#!/usr/bin/env python3
"""
ABCD Data Preprocessing Script - Final Version with Complete NA Handling
Handles subid generation, inner joins, and filtering for cognition and P-factor data
Includes comprehensive NA handling for whitespace and string-to-numeric conversion
"""

import pandas as pd
import os
import sys
import numpy as np

def convert_to_subid(src_subject_id):
    """Convert src_subject_id to subid format: remove underscores and add 'sub-' prefix"""
    if pd.isna(src_subject_id):
        return None
    clean_id = str(src_subject_id).replace('_', '')
    return f"sub-{clean_id}"


def ensure_subid_column(file_path):
    """Ensure the file has a subid column, add if missing"""
    print(f"  Checking subid in {os.path.basename(file_path)}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Check if subid column already exists
        if 'subid' in df.columns:
            print(f"    ✓ subid column already exists")
            return df
        
        # Find the subject ID column (usually src_subject_id or subjectkey)
        if 'src_subject_id' in df.columns:
            subject_col = 'src_subject_id'
        elif 'subjectkey' in df.columns:
            subject_col = 'subjectkey'
        else:
            # Try the first column as subject ID
            subject_col = df.columns[0]
            print(f"    Warning: Using first column '{subject_col}' as subject ID")
        
        # Add subid column
        df['subid'] = df[subject_col].apply(convert_to_subid)
        
        # Save the updated file
        df.to_csv(file_path, index=False)
        print(f"    ✓ Added subid column successfully")
        return df
        
    except Exception as e:
        print(f"    ✗ Error processing {file_path}: {str(e)}")
        return None


def load_sublist(sublist_file):
    """Load existing sublist.txt file"""
    try:
        with open(sublist_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: {sublist_file} not found")
        return set()
    except Exception as e:
        print(f"Error loading sublist: {str(e)}")
        return set()


def process_cognition_data(data_dir, sublist_file):
    """Process cognition data with inner join and filtering criteria"""
    print("Processing cognition data...")
    
    # Required files for cognition processing
    required_files = [
        "abcd_y_lt_baseline.csv",
        "abcd_p_demo_baseline.csv", 
        "mri_y_adm_info_baseline.csv",
        "nc_y_nihtb_baseline.csv",
        "mri_y_qc_motion_baseline.csv"
    ]
    
    # Ensure all files have subid column and load them
    dataframes = {}
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Error: {filename} not found")
            return None
        
        df = ensure_subid_column(file_path)
        if df is None:
            return None
        
        # Keep only necessary columns to avoid conflicts
        if filename == "abcd_p_demo_baseline.csv":
            df = df[['subid', 'demo_sex_v2']].copy()
        elif filename == "nc_y_nihtb_baseline.csv":
            cognition_cols = [
                "nihtbx_picvocab_uncorrected",
                "nihtbx_reading_uncorrected",
                "nihtbx_picture_uncorrected",
                "nihtbx_flanker_uncorrected",
                "nihtbx_list_uncorrected",
                "nihtbx_cardsort_uncorrected",
                "nihtbx_pattern_uncorrected",
                "nihtbx_cryst_uncorrected",
                "nihtbx_fluidcomp_uncorrected",
                "nihtbx_totalcomp_uncorrected"
            ]
            available_cols = [col for col in cognition_cols if col in df.columns]
            df = df[['subid'] + available_cols].copy()
        else:
            # For other files, keep only subid
            df = df[['subid']].copy()
        
        dataframes[filename] = df
        print(f"  Loaded {len(df)} rows from {filename}")
    
    # Load sublist
    valid_subjects = load_sublist(sublist_file)
    print(f"  Loaded {len(valid_subjects)} subjects from sublist.txt")
    
    # Use a different approach: start with the first dataframe and iteratively merge
    result_df = dataframes[required_files[0]]
    print(f"  Starting with {required_files[0]}: {len(result_df)} subjects")
    
    for filename in required_files[1:]:
        df = dataframes[filename]
        # Merge using subid, but handle potential duplicates
        result_df = pd.merge(result_df, df, on='subid', how='inner', suffixes=('', '_drop'))
        # Drop any columns that were duplicated with _drop suffix
        drop_cols = [col for col in result_df.columns if col.endswith('_drop')]
        if drop_cols:
            result_df = result_df.drop(columns=drop_cols)
        print(f"  After inner join with {filename}: {len(result_df)} subjects")
    
    # Apply filtering criteria
    print("  Applying filtering criteria...")
    
    # 1. Exclude demo_sex_v2 != 1 and != 2
    if 'demo_sex_v2' in result_df.columns:
        initial_count = len(result_df)
        result_df = result_df[result_df['demo_sex_v2'].isin([1, 2])]
        print(f"  After sex filtering: {len(result_df)} subjects (excluded {initial_count - len(result_df)})")
    
    # 2. Check cognition columns are not null
    cognition_columns = [
        "nihtbx_picvocab_uncorrected",
        "nihtbx_reading_uncorrected",
        "nihtbx_picture_uncorrected",
        "nihtbx_flanker_uncorrected",
        "nihtbx_list_uncorrected",
        "nihtbx_cardsort_uncorrected",
        "nihtbx_pattern_uncorrected",
        "nihtbx_cryst_uncorrected",
        "nihtbx_fluidcomp_uncorrected",
        "nihtbx_totalcomp_uncorrected"
    ]
    
    available_cognition_cols = [col for col in cognition_columns if col in result_df.columns]
    if available_cognition_cols:
        initial_count = len(result_df)
        result_df = result_df.dropna(subset=available_cognition_cols)
        print(f"  After cognition non-null filtering: {len(result_df)} subjects (excluded {initial_count - len(result_df)})")
    
    # 3. Filter by sublist
    initial_count = len(result_df)
    result_df = result_df[result_df['subid'].isin(valid_subjects)]
    print(f"  After sublist filtering: {len(result_df)} subjects (excluded {initial_count - len(result_df)})")
    
    # Extract and save cognition sublist
    cognition_subjects = sorted(result_df['subid'].tolist())
    
    # Exclude specific subject due to rest_run-2 xcpd processing failure
    excluded_subject = "sub-NDARINVPETWZ0JC"
    if excluded_subject in cognition_subjects:
        cognition_subjects.remove(excluded_subject)
        print(f"  Excluded {excluded_subject} due to rest_run-2 xcpd processing failure")
    
    output_file = os.path.join(data_dir, "cognition_sublist.txt")
    
    with open(output_file, 'w') as f:
        for subid in cognition_subjects:
            f.write(f"{subid}\n")
    
    print(f"  ✓ Saved {len(cognition_subjects)} subjects to cognition_sublist.txt")
    return cognition_subjects


def process_pfactor_data(data_dir, sublist_file):
    """Process P-factor data with inner join and filtering criteria - Enhanced with NA handling"""
    print("Processing P-factor data with enhanced NA handling...")
    
    # Required files for P-factor processing (5个指定的表)
    required_files = [
        "Pfactor_score_wx.csv",
        "mri_y_qc_motion_baseline.csv",
        "mri_y_adm_info_baseline.csv", 
        "abcd_y_lt_baseline.csv",
        "abcd_p_demo_baseline.csv"
    ]
    
    # Ensure all files have subid column and load them
    dataframes = {}
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Error: {filename} not found")
            return None
        
        df = ensure_subid_column(file_path)
        if df is None:
            return None
        
        # Keep only necessary columns to avoid conflicts
        if filename == "abcd_p_demo_baseline.csv":
            df = df[['subid', 'demo_sex_v2']].copy()
        elif filename == "Pfactor_score_wx.csv":
            # Keep P-factor columns
            pfactor_cols = ['General', 'Ext', 'ADHD', 'Int']
            available_pfactor_cols = [col for col in pfactor_cols if col in df.columns]
            if not available_pfactor_cols:
                print("Error: No P-factor columns found in Pfactor_score_wx.csv")
                return None
            
            # Enhanced NA handling for P-factor columns
            for col in available_pfactor_cols:
                # Convert to string first to handle mixed types
                df[col] = df[col].astype(str)
                # Remove leading/trailing whitespace and convert empty/whitespace to NaN
                df[col] = df[col].str.strip()
                df.loc[df[col] == '', col] = pd.NA
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['subid'] + available_pfactor_cols].copy()
        else:
            # For other files, keep only subid
            df = df[['subid']].copy()
        
        dataframes[filename] = df
        print(f"  Loaded {len(df)} rows from {filename}")
    
    # Load sublist
    valid_subjects = load_sublist(sublist_file)
    print(f"  Loaded {len(valid_subjects)} subjects from sublist.txt")
    
    # Use a different approach: start with the first dataframe and iteratively merge
    result_df = dataframes[required_files[0]]
    print(f"  Starting with {required_files[0]}: {len(result_df)} subjects")
    
    for filename in required_files[1:]:
        df = dataframes[filename]
        # Merge using subid, but handle potential duplicates
        result_df = pd.merge(result_df, df, on='subid', how='inner', suffixes=('', '_drop'))
        # Drop any columns that were duplicated with _drop suffix
        drop_cols = [col for col in result_df.columns if col.endswith('_drop')]
        if drop_cols:
            result_df = result_df.drop(columns=drop_cols)
        print(f"  After inner join with {filename}: {len(result_df)} subjects")
    
    # Apply filtering criteria
    print("  Applying filtering criteria...")
    
    # 1. Exclude demo_sex_v2 != 1 and != 2
    if 'demo_sex_v2' in result_df.columns:
        initial_count = len(result_df)
        result_df = result_df[result_df['demo_sex_v2'].isin([1, 2])]
        print(f"  After sex filtering: {len(result_df)} subjects (excluded {initial_count - len(result_df)})")
    
    # 2. Check P-factor columns are not null (enhanced)
    pfactor_cols = ['General', 'Ext', 'ADHD', 'Int']
    available_pfactor_cols = [col for col in pfactor_cols if col in result_df.columns]
    if available_pfactor_cols:
        initial_count = len(result_df)
        
        # Check for NA values before filtering
        na_counts = result_df[available_pfactor_cols].isnull().sum()
        print(f"  NA counts before filtering:")
        for col, count in na_counts.items():
            print(f"    {col}: {count}")
        
        # Apply dropna filtering
        result_df = result_df.dropna(subset=available_pfactor_cols)
        print(f"  After P-factor non-null filtering: {len(result_df)} subjects (excluded {initial_count - len(result_df)})")
    
    # 3. Filter by sublist
    initial_count = len(result_df)
    result_df = result_df[result_df['subid'].isin(valid_subjects)]
    print(f"  After sublist filtering: {len(result_df)} subjects (excluded {initial_count - len(result_df)})")
    
    # Extract and save P-factor sublist
    pfactor_subjects = sorted(result_df['subid'].tolist())
    
    # Exclude specific subject due to rest_run-2 xcpd processing failure
    excluded_subject = "sub-NDARINVPETWZ0JC"
    if excluded_subject in pfactor_subjects:
        pfactor_subjects.remove(excluded_subject)
        print(f"  Excluded {excluded_subject} due to rest_run-2 xcpd processing failure")
    
    output_file = os.path.join(data_dir, "pfactor_sublist.txt")
    
    with open(output_file, 'w') as f:
        for subid in pfactor_subjects:
            f.write(f"{subid}\n")
    
    print(f"  ✓ Saved {len(pfactor_subjects)} subjects to pfactor_sublist.txt")
    return pfactor_subjects


def main():
    """Main function"""
    # Set up paths
    data_dir = r"d:\code\WM_prediction\data\ABCD\table"
    sublist_file = os.path.join(data_dir, "ABCD_fc_sublist.txt")
    
    print("=== ABCD Data Preprocessing - Final Enhanced Version ===")
    print(f"Data directory: {data_dir}")
    print(f"Sublist file: {sublist_file}")
    
    # Process cognition data
    cognition_subjects = process_cognition_data(data_dir, sublist_file)
    if cognition_subjects is None:
        print("Error processing cognition data")
        return 1
    
    # Process P-factor data with enhanced NA handling
    pfactor_subjects = process_pfactor_data(data_dir, sublist_file)
    if pfactor_subjects is None:
        print("Error processing P-factor data")
        return 1
    
    print(f"\n=== Summary ===")
    print(f"Cognition subjects: {len(cognition_subjects)}")
    print(f"P-factor subjects: {len(pfactor_subjects)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())