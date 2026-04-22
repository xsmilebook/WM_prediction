#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/hcp_efny_batch_common.sh"

get_hcp_efny_batch_options "$@"

StudyFolder="$STUDY_FOLDER_DEFAULT"
Sessionlist=""

if [[ -n "$command_line_specified_study_folder" ]]; then
    StudyFolder="$command_line_specified_study_folder"
fi

if [[ -n "$command_line_specified_session" ]]; then
    Sessionlist="$command_line_specified_session"
elif [[ -n "$command_line_specified_subject" ]]; then
    Sessionlist="$command_line_specified_subject"
else
    echo "ERROR: --Session= or --Subject= is required" >&2
    exit 1
fi

setup_hcp_efny_batch_env
echo "$@"

for Session in $Sessionlist; do
    echo "PreFreeSurferPipelineBatch.sh: Processing Session: $Session"
    run_prefreesurfer_session "$StudyFolder" "$Session"
done
