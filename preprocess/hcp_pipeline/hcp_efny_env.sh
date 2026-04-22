#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

setup_module_env() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi

    source /etc/profile >/dev/null 2>&1 || true

    if command -v module >/dev/null 2>&1; then
        return 0
    fi

    for candidate in \
        /etc/profile.d/modules.sh \
        /usr/share/Modules/init/bash \
        /usr/share/modules/init/bash
    do
        if [[ -f "$candidate" ]]; then
            # shellcheck disable=SC1090
            source "$candidate"
            if command -v module >/dev/null 2>&1; then
                return 0
            fi
        fi
    done

    return 1
}

resolve_workbench_dir() {
    local root="/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/workbench"
    local candidates=(
        "$root/bin_rh_linux64"
        "$root/exe_rh_linux64"
        "$root"
    )
    local dir
    for dir in "${candidates[@]}"; do
        if [[ -x "$dir/wb_command" ]]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done

    echo "Unable to find executable wb_command under $root" >&2
    exit 1
}

setup_workbench_runtime() {
    local root="/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/workbench"
    local primary_lib="$root/libs_rh_linux64"
    local ogl_lib="$root/libs_rh_linux64_software_opengl"
    local conda_lib=""
    local conda_sysroot_lib=""
    local library_path_parts=()

    if [[ ! -d "$primary_lib" ]]; then
        echo "Workbench library directory not found: $primary_lib" >&2
        exit 1
    fi

    export QT_PLUGIN_PATH="$primary_lib/plugins"

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        conda_lib="$CONDA_PREFIX/lib"
        conda_sysroot_lib="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"
    fi

    if [[ -d "$conda_sysroot_lib" ]]; then
        library_path_parts+=("$conda_sysroot_lib")
    fi

    if [[ -d "$conda_lib" ]]; then
        library_path_parts+=("$conda_lib")
    fi

    library_path_parts+=("$primary_lib")

    if [[ -d "$ogl_lib" ]]; then
        library_path_parts+=("$ogl_lib")
    fi

    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        library_path_parts+=("$LD_LIBRARY_PATH")
    fi

    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$(IFS=:; printf '%s' "${library_path_parts[*]}")
}

resolve_msm_bin_dir() {
    local root="/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/msm_centos_v3_bin"
    local candidates=(
        "."
        "bin"
        "build"
        "src/MSM"
    )
    local candidate
    local dir

    for candidate in "${candidates[@]}"; do
        dir="$root/$candidate"
        if [[ "$candidate" == "." ]]; then
            dir="$root"
        fi
        if [[ -x "$dir/msm" ]]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done

    local found
    found=$(find "$root" -maxdepth 5 -type f -name msm -perm -111 2>/dev/null | head -n 1 || true)
    if [[ -n "$found" ]]; then
        dirname "$found"
        return 0
    fi

    echo "Unable to find executable msm under $root" >&2
    echo "Install a runnable MSM binary and point MSMBINDIR to a directory containing msm." >&2
    exit 1
}

check_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "Required command not found on PATH: $name" >&2
        exit 1
    fi
}

if setup_module_env; then
    module load freesurfer/6.0.0
    module load fsl/6.3.0
    module load openblas/0.3.7
    export FS_LICENSE="/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/freesurfer_license/license.txt"
else
    echo "module command is unavailable; using existing PATH and environment variables" >&2
fi

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

export HCPPIPEDIR="$PROJECT_ROOT/HCPpipelines-5.0.0"
export CARET7DIR
CARET7DIR=$(resolve_workbench_dir)
setup_workbench_runtime
export PATH="$CARET7DIR:$PATH"
export MSMBINDIR
MSMBINDIR=$(resolve_msm_bin_dir)
export HCPCIFTIRWDIR="$HCPPIPEDIR/global/matlab/cifti-matlab"
export MSMCONFIGDIR="$HCPPIPEDIR/MSMConfig"

if [[ -z "${FSLDIR:-}" ]]; then
    found_fsl=$(command -v fslmaths || true)
    if [[ -n "$found_fsl" ]]; then
        export FSLDIR=$(dirname "$(dirname "$found_fsl")")
    fi
fi

export FSL_DIR="${FSLDIR:-}"

if [[ ! -f "$HCPPIPEDIR/global/scripts/finish_hcpsetup.shlib" ]]; then
    echo "finish_hcpsetup.shlib not found under HCPPIPEDIR=$HCPPIPEDIR" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$HCPPIPEDIR/global/scripts/finish_hcpsetup.shlib"

check_command fslmaths
check_command recon-all
check_command wb_command

if [[ ! -x "$MSMBINDIR/msm" ]]; then
    echo "MSMBINDIR does not contain executable msm: $MSMBINDIR" >&2
    exit 1
fi

python - <<'PY' >/dev/null
import importlib
for module_name in ("nibabel", "nilearn"):
    importlib.import_module(module_name)
PY
