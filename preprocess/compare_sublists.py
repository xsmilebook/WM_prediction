#!/usr/bin/env python3
"""
Compare sublist files to identify differences between old and new lists.
Handles subid format conversion for existing files.
"""

import os
import sys

def convert_to_subid(subject_id):
    """Convert subject ID to subid format: remove '_' and add 'sub-' prefix"""
    if subject_id.startswith('sub-'):
        return subject_id
    # Remove underscores and add sub- prefix
    clean_id = subject_id.replace('_', '')
    return f'sub-{clean_id}'

def read_sublist_file(filepath):
    """Read a sublist file and return set of subject IDs"""
    try:
        with open(filepath, 'r') as f:
            # Read all non-empty lines and strip whitespace
            subjects = [line.strip() for line in f if line.strip()]
        return set(subjects)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return set()
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return set()

def compare_sublists(old_file, new_file, file_label=""):
    """Compare two sublist files and report differences"""
    print(f"\n{'='*60}")
    print(f"Comparing {file_label} sublists:")
    print(f"Old file: {os.path.basename(old_file)}")
    print(f"New file: {os.path.basename(new_file)}")
    print(f"{'='*60}")
    
    # Read files
    old_subjects = read_sublist_file(old_file)
    new_subjects = read_sublist_file(new_file)
    
    # Convert old subjects to subid format if needed
    old_subjects_converted = set()
    for subject in old_subjects:
        old_subjects_converted.add(convert_to_subid(subject))
    
    # Calculate differences
    added_subjects = new_subjects - old_subjects_converted
    removed_subjects = old_subjects_converted - new_subjects
    common_subjects = new_subjects & old_subjects_converted
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"  Old list size: {len(old_subjects_converted)}")
    print(f"  New list size: {len(new_subjects)}")
    print(f"  Common subjects: {len(common_subjects)}")
    print(f"  Added subjects: {len(added_subjects)}")
    print(f"  Removed subjects: {len(removed_subjects)}")
    
    # Show added subjects
    if added_subjects:
        print(f"\n➕ ADDED SUBJECTS ({len(added_subjects)}):")
        for subject in sorted(added_subjects)[:10]:  # Show first 10
            print(f"  {subject}")
        if len(added_subjects) > 10:
            print(f"  ... and {len(added_subjects) - 10} more")
    else:
        print(f"\n➕ No subjects added")
    
    # Show removed subjects
    if removed_subjects:
        print(f"\n➖ REMOVED SUBJECTS ({len(removed_subjects)}):")
        for subject in sorted(removed_subjects)[:10]:  # Show first 10
            print(f"  {subject}")
        if len(removed_subjects) > 10:
            print(f"  ... and {len(removed_subjects) - 10} more")
    else:
        print(f"\n➖ No subjects removed")
    
    # Show conversion examples if applicable
    if old_subjects != old_subjects_converted:
        print(f"\n🔧 FORMAT CONVERSION EXAMPLES:")
        original_sample = list(old_subjects)[:3]
        for orig in original_sample:
            converted = convert_to_subid(orig)
            print(f"  {orig} → {converted}")
    
    return {
        'old_count': len(old_subjects_converted),
        'new_count': len(new_subjects),
        'added': added_subjects,
        'removed': removed_subjects,
        'common': common_subjects
    }

def main():
    """Main function to run all comparisons"""
    
    # Define file paths
    data_dir = r"d:\code\WM_prediction\data\ABCD\table"
    
    cognition_old = os.path.join(data_dir, "subject_ids_used_n4388_cognition_motion2runFD.txt")
    cognition_new = os.path.join(data_dir, "cognition_sublist.txt")
    
    pfactor_old = os.path.join(data_dir, "subject_ids_used_pfactor_n4465_motion2runFD.txt")
    pfactor_new = os.path.join(data_dir, "pfactor_sublist.txt")
    
    print("🔍 SUBLIST COMPARISON TOOL")
    print("Comparing old subject lists with new generated sublists")
    
    # Compare cognition sublists
    cognition_results = None
    if os.path.exists(cognition_old) and os.path.exists(cognition_new):
        cognition_results = compare_sublists(cognition_old, cognition_new, "COGNITION")
    else:
        print(f"\n⚠️  Skipping cognition comparison - files not found")
        if not os.path.exists(cognition_old):
            print(f"   Missing: {cognition_old}")
        if not os.path.exists(cognition_new):
            print(f"   Missing: {cognition_new}")
    
    # Compare pfactor sublists
    pfactor_results = None
    if os.path.exists(pfactor_old) and os.path.exists(pfactor_new):
        pfactor_results = compare_sublists(pfactor_old, pfactor_new, "PFACTOR")
    else:
        print(f"\n⚠️  Skipping pfactor comparison - files not found")
        if not os.path.exists(pfactor_old):
            print(f"   Missing: {pfactor_old}")
        if not os.path.exists(pfactor_new):
            print(f"   Missing: {pfactor_new}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 OVERALL SUMMARY")
    print(f"{'='*60}")
    
    if cognition_results:
        print(f"Cognition: {cognition_results['old_count']} → {cognition_results['new_count']} "
              f"({len(cognition_results['added'])} added, {len(cognition_results['removed'])} removed)")
    
    if pfactor_results:
        print(f"P-factor:  {pfactor_results['old_count']} → {pfactor_results['new_count']} "
              f"({len(pfactor_results['added'])} added, {len(pfactor_results['removed'])} removed)")
    
    print(f"\n✅ Comparison complete!")

if __name__ == "__main__":
    main()