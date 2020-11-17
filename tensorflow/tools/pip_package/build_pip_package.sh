#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function cp_external() {
  local src_dir=$1
  local dest_dir=$2

  pushd .
  cd "$src_dir"
  for f in `find . ! -type d ! -name '*.py' ! -path '*local_config_cuda*' ! -path '*local_config_tensorrt*' ! -path '*local_config_syslibs*' ! -path '*org_tensorflow*'`; do
    mkdir -p "${dest_dir}/$(dirname ${f})"
    cp "${f}" "${dest_dir}/$(dirname ${f})/"
  done
  popd

  mkdir -p "${dest_dir}/local_config_cuda/cuda/cuda/"
  cp "${src_dir}/local_config_cuda/cuda/cuda/cuda_config.h" "${dest_dir}/local_config_cuda/cuda/cuda/"
}

function move_to_root_if_exists () {
  arg_to_move="$1"
  if [ -e "${arg_to_move}" ]; then
    mv ${arg_to_move} ./
  fi
}

function reorganize_includes() {
  TMPDIR="${1%/}"
  pushd "${TMPDIR}/tensorflow/include/"

  move_to_root_if_exists external/com_google_absl/absl

  move_to_root_if_exists external/eigen_archive/Eigen
  move_to_root_if_exists external/eigen_archive/unsupported

  move_to_root_if_exists external/jsoncpp_git/include
  rm -rf external/jsoncpp_git

  move_to_root_if_exists external/com_google_protobuf/src/google
  rm -rf external/com_google_protobuf/python

  # Select DML files are copied in the copy_dml_redist_files function; we should
  # never include the entire contents of the redistributable package.
  rm -rf external/dml_redist

  popd
}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  if [[ "${PLATFORM}" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]; then
    true
  else
    false
  fi
}

function copy_dml_redist_files() {
  dml_redist_dir="${1%/}"

  if is_windows; then
    runfiles_manifest_path=bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles_manifest
  else
    runfiles_manifest_path=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles_manifest
  fi

  # Locate path to DirectMLConfig.h in the runfiles manifest
  dml_config_path=$(awk '/^dml_redist\/.*\/DirectMLConfig\.h/ {print $2}' $runfiles_manifest_path)
  if [ -z "$dml_config_path" ]; then
    echo "Could not find DirectMLConfig.h in runfiles"
    exit 1
  fi

  # Locate path to root of DirectML redist files
  dml_redist_root=$(echo $dml_config_path | sed 's/\/include\/DirectMLConfig\.h//')

  # Copy library and licenses
  if is_windows; then
    dml_version=$(awk '/^#define DIRECTML_SOURCE_VERSION "([abcdef0-9]+)"/ { gsub("\"","",$3); print $3 }' $dml_config_path)
    echo "DML Version = '$dml_version'"
    dml_dll=$(find "${dml_redist_root}/bin/x64-win/" -type f -name "*.dll" ! -name "DirectML.Debug.*")
    cp "$dml_dll" "${dml_redist_dir}"/DirectML.${dml_version}.dll
  else
    dml_version=$(awk -v RS='\r\n' '/^#define DIRECTML_SOURCE_VERSION "([abcdef0-9]+)"/ { gsub("\"","",$3); print $3 }' $dml_config_path)
    echo "DML Version = '$dml_version'"
    dml_so=$(find "$dml_redist_root/bin/x64-linux/" -type f -name "*.so" ! -name libdirectml.debug.*)
    cp "$dml_so" "${dml_redist_dir}"/libdirectml.${dml_version}.so
  fi
  cp "${dml_redist_root}/LICENSE.txt" "${dml_redist_dir}/DirectML_LICENSE.txt"
  cp "${dml_redist_root}/ThirdPartyNotices.txt" "${dml_redist_dir}/DirectML_ThirdPartyNotices.txt"
}

function prepare_src() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  TMPDIR="${1%/}"
  mkdir -p "$TMPDIR"
  EXTERNAL_INCLUDES="${TMPDIR}/tensorflow/include/external"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  if is_windows; then
    rm -rf ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    mkdir -p ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    echo "Unzipping simple_console_for_windows.zip to create runfiles tree..."
    unzip -o -q ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_windows.zip -d ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    echo "Unzip finished."
    # runfiles structure after unzip the python binary
    cp \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow/LICENSE \
      "${TMPDIR}"
    cp -R \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow/tensorflow \
      "${TMPDIR}"
    cp_external \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles \
      "${EXTERNAL_INCLUDES}/"
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow
  else
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow
    if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external ]; then
      # Old-style runfiles structure (--legacy_external_runfiles).
      cp \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/LICENSE \
        "${TMPDIR}"
      cp -R \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
        "${TMPDIR}"
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external \
        "${EXTERNAL_INCLUDES}"
      # Copy MKL libs over so they can be loaded at runtime
      so_lib_dir=$(ls $RUNFILES | grep solib) || true
      if [ -n "${so_lib_dir}" ]; then
        mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
        if [ -n "${mkl_so_dir}" ]; then
          mkdir "${TMPDIR}/${so_lib_dir}"
          cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
        fi
      fi
    else
      # New-style runfiles structure (--nolegacy_external_runfiles).
      cp \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/LICENSE \
        "${TMPDIR}"
      cp -R \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
        "${TMPDIR}"
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles \
        "${EXTERNAL_INCLUDES}"
      # Copy MKL libs over so they can be loaded at runtime
      so_lib_dir=$(ls $RUNFILES | grep solib) || true
      if [ -n "${so_lib_dir}" ]; then
        mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
        if [ -n "${mkl_so_dir}" ]; then
          mkdir "${TMPDIR}/${so_lib_dir}"
          cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
        fi
      fi
    fi
  fi

  mkdir -p ${TMPDIR}/third_party
  cp -R $RUNFILES/third_party/eigen3 ${TMPDIR}/third_party

  reorganize_includes "${TMPDIR}"

  cp tensorflow/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow/tools/pip_package/README ${TMPDIR}/README.md
  cp tensorflow/tools/pip_package/setup.py ${TMPDIR}

  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.so
  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.so.[0-9].*

  # TFDML will attempt to load the DirectML library from the same directory as the core
  # TF libary. This location differs based on platform. Note that tensorflow_core is
  # copied from "tensorflow" (see lines directly below).
  # - Windows : tensorflow_core/python/_pywrap_tensorflow_internal.pyd
  # - Linux   : tensorflow_core/libtensorflow_framework.so.1
  if is_windows
    copy_dml_redist_files "${TMPDIR}/tensorflow/python"
  else
    copy_dml_redist_files "${TMPDIR}/tensorflow"

  # In order to break the circular dependency between tensorflow and
  # tensorflow_estimator which forces us to do a multi-step release, we are
  # creating a virtual python package called tensorflow and moving all the tf
  # code into another python package called tensorflow_core:
  #
  #   * move code from tensorflow to tensorflow_core
  #   * create the virtual pip package: create folder and __init__.py file with
  #     needed code for transparent forwarding
  #
  # This is transparent to internal code or to code not using the pip packages.
  mv "${TMPDIR}/tensorflow" "${TMPDIR}/tensorflow_core"
  mkdir "${TMPDIR}/tensorflow"
  mv "${TMPDIR}/tensorflow_core/virtual_root.__init__.py" "${TMPDIR}/tensorflow/__init__.py"

  # In V1 API, we need to remove deprecation warnings from
  # ${TMPDIR}/tensorflow_core/__init__.py as those exists in
  # ${TMPDIR}/tensorflow/__init__.py which does an import* and if these don't
  # get removed users get 6 deprecation warning just on
  #
  #   import tensorflow as tf
  #
  # which is not ok. We disable deprecation by using sed to toggle the flag
  # TODO(mihaimaruseac): When we move the API to root, remove this hack
  # Note: Can't do in place sed that works on all OS, so use a temp file instead
  sed \
    "s/deprecation=True/deprecation=False/g" \
    "${TMPDIR}/tensorflow_core/__init__.py" > "${TMPDIR}/tensorflow_core/__init__.out"
  mv "${TMPDIR}/tensorflow_core/__init__.out" "${TMPDIR}/tensorflow_core/__init__.py"
}

function build_wheel() {
  if [ $# -lt 2 ] ; then
    echo "No src and dest dir provided"
    exit 1
  fi

  TMPDIR="$1"
  DEST="$2"
  PKG_NAME_FLAG="$3"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e tools/python_bin_path.sh ]]; then
    source tools/python_bin_path.sh
  fi

  pushd ${TMPDIR} > /dev/null

  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${PKG_NAME_FLAG} >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

function usage() {
  echo "Usage:"
  echo "$0 [--src srcdir] [--dst dstdir] [options]"
  echo "$0 dstdir [options]"
  echo ""
  echo "    --src                 prepare sources in srcdir"
  echo "                              will use temporary dir if not specified"
  echo ""
  echo "    --dst                 build wheel in dstdir"
  echo "                              if dstdir is not set do not build, only prepare sources"
  echo ""
  echo "  Options:"
  echo "    --project_name <name> set project name to name"
  echo "    --cpu                 build tensorflow_cpu"
  echo "    --gpu                 build tensorflow_gpu"
  echo "    --gpudirect           build tensorflow_gpudirect"
  echo "    --rocm                build tensorflow_rocm"
  echo "    --directml            build tensorflow_directml"
  echo "    --nightly_flag        build tensorflow nightly"
  echo ""
  exit 1
}

function main() {
  PKG_NAME_FLAG=""
  PROJECT_NAME=""
  GPU_BUILD=0
  PROJECT_NAME_CPU=0
  ROCM_BUILD=0
  DIRECTML_BUILD=0
  NIGHTLY_BUILD=0
  SRCDIR=""
  DSTDIR=""
  CLEANSRC=1
  while true; do
    if [[ "$1" == "--help" ]]; then
      usage
      exit 1
    elif [[ "$1" == "--nightly_flag" ]]; then
      NIGHTLY_BUILD=1
    elif [[ "$1" == "--gpu" ]]; then
      GPU_BUILD=1
    elif [[ "$1" == "--cpu" ]]; then
      # Check that --gpu has not been passed.
      if [[ ${GPU_BUILD} == "1" ]]; then
        echo "Specifying both --cpu and --gpu to build_pip_package is not allowed."
        usage
        exit 1
      fi

      PROJECT_NAME_CPU=1
    elif [[ "$1" == "--gpudirect" ]]; then
      PKG_NAME_FLAG="--project_name tensorflow_gpudirect"
    elif [[ "$1" == "--rocm" ]]; then
      ROCM_BUILD=1
    elif [[ "$1" == "--directml" ]]; then
      DIRECTML_BUILD=1
    elif [[ "$1" == "--project_name" ]]; then
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
    elif [[ "$1" == "--src" ]]; then
      shift
      SRCDIR="$(real_path $1)"
      CLEANSRC=0
    elif [[ "$1" == "--dst" ]]; then
      shift
      DSTDIR="$(real_path $1)"
    else
      DSTDIR="$(real_path $1)"
    fi
    shift

    if [[ -z "$1" ]]; then
      break
    fi
  done

  if [[ -z "$DSTDIR" ]] && [[ -z "$SRCDIR" ]]; then
    echo "No destination dir provided"
    usage
    exit 1
  fi

  if [[ -z "$SRCDIR" ]]; then
    # make temp srcdir if none set
    SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  fi

  prepare_src "$SRCDIR"

  if [[ -z "$DSTDIR" ]]; then
      # only want to prepare sources
      exit
  fi

  if [[ -n ${PROJECT_NAME} ]]; then
    PKG_NAME_FLAG="--project_name ${PROJECT_NAME}"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${GPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_gpu"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${ROCM_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_rocm"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${DIRECTML_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_directml"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${PROJECT_NAME_CPU} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_cpu"
  elif [[ ${NIGHTLY_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly"
  elif [[ ${GPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_gpu"
  elif [[ ${ROCM_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_rocm"
  elif [[ ${DIRECTML_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_directml"
  elif [[ ${PROJECT_NAME_CPU} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_cpu"
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG"

  if [[ $CLEANSRC -ne 0 ]]; then
    rm -rf "${TMPDIR}"
  fi
}

main "$@"
