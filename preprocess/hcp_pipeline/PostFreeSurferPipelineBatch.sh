#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/hcp_efny_batch_common.sh"

get_hcp_efny_batch_options "$@"

StudyFolder="$STUDY_FOLDER_DEFAULT"
Subjlist=""

if [[ -n "$command_line_specified_study_folder" ]]; then
    StudyFolder="$command_line_specified_study_folder"
fi

if [[ -n "$command_line_specified_subject" ]]; then
    Subjlist="$command_line_specified_subject"
elif [[ -n "$command_line_specified_session" ]]; then
    Subjlist="$command_line_specified_session"
else
    echo "ERROR: --Subject= or --Session= is required" >&2
    exit 1
fi

setup_hcp_efny_batch_env
echo "$@"

for Subject in $Subjlist; do
    echo "PostFreeSurferPipelineBatch.sh: Processing Subject: $Subject"
    run_postfreesurfer_subject "$StudyFolder" "$Subject"
done
