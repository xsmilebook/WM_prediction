#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"
SUBJ_LIST=${1:-"$PROJECT_ROOT/data/EFNY/table/sublist_new_left521.txt"}
LOG_DIR="$PROJECT_ROOT/log/hcp_pipeline/xcpd"

mkdir -p "$LOG_DIR"

while IFS= read -r subj || [[ -n "$subj" ]]; do
    subj=${subj//$'\r'/}
    [[ -n "$subj" ]] || continue
    echo "perform xcpd of subject: $subj"
    sbatch -J "$subj" \
        -o "$LOG_DIR/out.${subj}.txt" \
        -e "$LOG_DIR/error.${subj}.txt" \
        "xcpd_24p_csf_global.sh" "$subj"
done < "$SUBJ_LIST"
