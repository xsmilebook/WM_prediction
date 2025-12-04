import pandas as pd
import os

def main():
    # Define paths
    base_dir = r'd:\code\WM_prediction\data\PNC\table'
    sublist_path = os.path.join(base_dir, 'sublist.txt')
    fd_summary_path = os.path.join(base_dir, 'rest_fd_summary.csv')
    participants_path = os.path.join(base_dir, 'study-PNC_desc-participants.tsv')
    output_path = os.path.join(base_dir, 'subid_meanFD_age_sex.csv')

    # 1. Load sublist
    print(f"Loading sublist from {sublist_path}...")
    df_sub = pd.read_csv(sublist_path, header=None, names=['subid'])
    print(f"Loaded {len(df_sub)} subjects.")

    # 2. Load FD summary
    print(f"Loading FD summary from {fd_summary_path}...")
    df_fd = pd.read_csv(fd_summary_path)
    # Ensure subid matches format if needed (looks like it already has 'sub-')
    # Keep only relevant columns
    if 'rest_fd' in df_fd.columns and 'subid' in df_fd.columns:
        df_fd = df_fd[['subid', 'rest_fd']]
    else:
        raise ValueError("rest_fd_summary.csv missing required columns")

    # 3. Load Participants info
    print(f"Loading participants info from {participants_path}...")
    df_demo = pd.read_csv(participants_path, sep='\t')
    
    # Preprocess participants data
    # Add 'sub-' prefix to participant_id
    df_demo['subid'] = 'sub-' + df_demo['participant_id'].astype(str)
    
    # Map sex: Male -> 1, Female -> 2
    sex_map = {'Male': 1, 'Female': 2}
    df_demo['sex'] = df_demo['sex'].map(sex_map)
    
    # Select relevant columns
    df_demo = df_demo[['subid', 'age', 'sex']]

    # 4. Merge Data
    # Merge sublist with FD
    print("Merging data...")
    df_merged = pd.merge(df_sub, df_fd, on='subid', how='inner')
    
    # Merge with Demographics
    df_merged = pd.merge(df_merged, df_demo, on='subid', how='inner')

    # Rename rest_fd to meanFD
    df_merged = df_merged.rename(columns={'rest_fd': 'meanFD'})

    # Reorder columns
    df_merged = df_merged[['subid', 'age', 'sex', 'meanFD']]

    # 5. Save output
    print(f"Saving output to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print("Done!")
    print(f"Final shape: {df_merged.shape}")
    print(df_merged.head())

if __name__ == "__main__":
    main()
