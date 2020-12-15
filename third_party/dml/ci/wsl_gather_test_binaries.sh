#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if [[ $# -ne 3 ]]; then
    echo This script requires 3 positional arguments:
    echo 1: Name of the conda environment to activate
    echo 2: Path where the build output was generated
    echo 3: Path to store the gathered binaries
    echo Note: This script needs to run from the root of a tensorflow repo
    echo Example: ./wsl_gather_test_binaries.sh dml_ci_build /mnt/c/build_output /mnt/c/test_binaries
fi

SCRIPT_ROOT="`dirname \"$0\"`"
SCRIPT_ROOT="`( cd \"$SCRIPT_ROOT\" && pwd )`"

echo "Build conda environment: $1"
echo "Build output path: $2"
echo "Destination path: $3"

echo "Building config $2 in environment $1"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate $1
python "$SCRIPT_ROOT/gather_test_binaries.py" --source_root . --build_output $2 --destination $3
