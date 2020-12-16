#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if [[ $# -ne 5 ]]; then
    echo This script requires 5 positional arguments:
    echo 1: Name of the conda environment to activate
    echo 2: Name of the build configuration
    echo 3: Path to store build output
    echo 4: Include tests
    echo 5: Include C API
    echo Example: ./wsl_build.sh dml_ci_build release /mnt/c/build_output true
fi

echo "Build conda environment: $1"
echo "Build config: $2"
echo "Build output path: $3"

echo "Building config $2 in environment $1"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate $1

if [ "$4" == "True" ]; then
    python build.py --clean --package --tests --config $2 --build_output $3 --telemetry
else
    python build.py --clean --package --config $2 --build_output $3 --telemetry
fi

if [ "$5" == "True" ]; then
    python build.py --config $2 --build_output $3 --target //tensorflow/tools/lib_package:libtensorflow --telemetry
fi