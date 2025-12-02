#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ABCD covariates file in the same format as PNC
Author: AI Assistant
Date: 2024-12-02
"""

import pandas as pd
import numpy as np
import os

def load_sublist(sublist_file):
    """Load subject list from text file"""
    with open(sublist_file, 'r') as f:
        subids = [line.strip() for line in f if line.strip()]
    return subids

def process_age(age_months):
    """Convert age from months to years"""
    if pd.isna(age_months):
        return np.nan
    return age_months / 12

def calculate_weighted_mean_fd(rest1_frame, rest1_fd, rest2_frame, rest2_fd):
    """Calculate weighted average of mean FD across runs"""
    # Only use valid runs (where restX_valid would be 1)
    # For simplicity, we'll use all available data and weight by number of frames
    
    total_frames = 0
    weighted_fd_sum = 0
    
    # Add rest1 data if available
    if not pd.isna(rest1_frame) and not pd.isna(rest1_fd) and rest1_frame > 0:
        weighted_fd_sum += rest1_frame * rest1_fd
        total_frames += rest1_frame
    
    # Add rest2 data if available  
    if not pd.isna(rest2_frame) and not pd.isna(rest2_fd) and rest2_frame > 0:
        weighted_fd_sum += rest2_frame * rest2_fd
        total_frames += rest2_frame
    
    if total_frames > 0:
        return weighted_fd_sum / total_frames
    else:
        return np.nan

def generate_abcd_covariates():
    """Main function to generate ABCD covariates file"""
    
    # Define file paths
    base_path = r'D:\code\WM_prediction\data\ABCD\table'
    
    sublist_file = os.path.join(base_path, 'ABCD_fc_sublist.txt')
    demo_file = os.path.join(base_path, 'abcd_p_demo_baseline.csv')
    age_file = os.path.join(base_path, 'abcd_y_lt_baseline.csv')
    fd_file = os.path.join(base_path, 'rest_fd_summary.csv')
    
    output_file = os.path.join(base_path, 'ABCD_covariates_subid_meanFD_age_sex.csv')
    
    print("Loading subject list...")
    target_subids = load_sublist(sublist_file)
    print(f"Found {len(target_subids)} subjects in the list")
    
    print("Loading demographic data...")
    # Load demographic data - use src_subject_id as key
    demo_df = pd.read_csv(demo_file)
    # Keep only baseline data and select relevant columns
    demo_df = demo_df[demo_df['eventname'] == 'baseline_year_1_arm_1']
    demo_df = demo_df[['src_subject_id', 'demo_sex_v2']].copy()
    demo_df = demo_df.dropna(subset=['src_subject_id', 'demo_sex_v2'])
    
    print("Loading age data...")
    # Load age data - use src_subject_id as key
    age_df = pd.read_csv(age_file)
    # Keep only baseline data and select relevant columns
    age_df = age_df[age_df['eventname'] == 'baseline_year_1_arm_1']
    age_df = age_df[['src_subject_id', 'interview_age']].copy()
    age_df = age_df.dropna(subset=['src_subject_id', 'interview_age'])
    
    print("Loading motion data...")
    # Load FD data - subid is already in the right format
    fd_df = pd.read_csv(fd_file)
    # Select relevant columns
    fd_df = fd_df[['subid', 'rest1_frame', 'rest1_fd', 'rest2_frame', 'rest2_fd']].copy()
    fd_df = fd_df.dropna(subset=['subid'])
    
    print("Processing data...")
    
    # Convert age from months to years
    age_df['age_years'] = age_df['interview_age'].apply(process_age)
    
    # Calculate weighted mean FD
    fd_df['meanFD'] = fd_df.apply(
        lambda row: calculate_weighted_mean_fd(
            row['rest1_frame'], row['rest1_fd'], 
            row['rest2_frame'], row['rest2_fd']
        ), axis=1
    )
    
    # Create a mapping from src_subject_id to subid format
    # src_subject_id: NDAR_INV00HEV6HB -> subid: sub-NDARINV00HEV6HB
    demo_df['subid'] = demo_df['src_subject_id'].apply(lambda x: f"sub-{x.replace('_', '')}")
    age_df['subid'] = age_df['src_subject_id'].apply(lambda x: f"sub-{x.replace('_', '')}")
    
    # Merge data on subid
    print("Merging data...")
    
    # Start with the target subject list
    result_df = pd.DataFrame({'subid': target_subids})
    
    # Merge with demographic data (sex)
    result_df = result_df.merge(demo_df[['subid', 'demo_sex_v2']], on='subid', how='left')
    result_df.rename(columns={'demo_sex_v2': 'sex'}, inplace=True)
    
    # Merge with age data
    result_df = result_df.merge(age_df[['subid', 'age_years']], on='subid', how='left')
    result_df.rename(columns={'age_years': 'age'}, inplace=True)
    
    # Merge with FD data
    result_df = result_df.merge(fd_df[['subid', 'meanFD']], on='subid', how='left')
    
    # Check for missing data
    missing_data = result_df.isnull().sum()
    if missing_data.any():
        print("Warning: Missing data found:")
        print(missing_data)
        
        # Show which subjects have missing data
        missing_subjects = result_df[result_df.isnull().any(axis=1)]
        if not missing_subjects.empty:
            print(f"Subjects with missing data: {len(missing_subjects)}")
            print(missing_subjects[['subid']].head())
    
    # Reorder columns to match PNC format
    result_df = result_df[['subid', 'age', 'sex', 'meanFD']]
    
    # Remove rows with any missing data (optional - you might want to keep them)
    result_df_clean = result_df.dropna()
    
    print(f"\nFinal dataset:")
    print(f"Total subjects: {len(result_df)}")
    print(f"Subjects with complete data: {len(result_df_clean)}")
    print(f"Subjects with missing data: {len(result_df) - len(result_df_clean)}")
    
    # Show summary statistics
    print(f"\nSummary statistics:")
    print(f"Age range: {result_df_clean['age'].min():.2f} - {result_df_clean['age'].max():.2f} years")
    print(f"Sex distribution: {result_df_clean['sex'].value_counts().to_dict()}")
    print(f"Mean FD range: {result_df_clean['meanFD'].min():.4f} - {result_df_clean['meanFD'].max():.4f}")
    
    # Save the complete dataset
    result_df_clean.to_csv(output_file, index=False)
    print(f"\nSaved covariates file to: {output_file}")
    
    # Also save a version with all subjects (including those with missing data)
    output_file_all = output_file.replace('.csv', '_all_subjects.csv')
    result_df.to_csv(output_file_all, index=False)
    print(f"Also saved file with all subjects to: {output_file_all}")
    
    return result_df_clean

if __name__ == "__main__":
    generate_abcd_covariates()