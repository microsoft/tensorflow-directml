#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gathers the test binaries."""

import os
import argparse
import shutil
import zipfile
import json
from bazel_helpers import BazelEnv, TargetKind

def _parse_args():
  """Parses the arguments given to this script."""

  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--build_output",
      help="path to the build's output",
      required=True)

  parser.add_argument(
      "--source_root",
      help="path to tensorflow's source",
      required=True)

  parser.add_argument(
      "--destination",
      help="folder to copy the files to",
      required=True)

  parser.add_argument(
      "--test_prefix",
      default="tfdml_test",
      help="Prefix to use when creating a symlink to query the tests.")

  return parser.parse_args()

def _patch_main_python_file(file_path):
  """
  Patches the main python file to allow tests to be run on a different machine.
  """
  try:
    with open(file_path, "r", encoding="utf-8") as main_file:
      modified_source = main_file.read().replace(
          "python_program = FindPythonBinary(module_space)",
          "python_program = sys.executable")
  except UnicodeDecodeError:
    # This is an executable rather than a python file, so do nothing
    return

  with open(file_path, "w", encoding="utf-8") as main_file:
    main_file.write(modified_source)

def _patch_archive(zip_file_path):
  """
  Patches the zip archive to allow tests to be run on a different machine.
  """

  # When building python tests, bazel hardcodes the absolute path to the python
  # executable and assigns it to PYTHON_BINARY
  # (e.g. PYTHON_BINARY = 'C:/tools/miniconda3/envs/tfdml/python.exe'), which
  # makes it really hard to run the test executables on different machines. The
  # workaround that we use here is to change the 'python_program' variable
  # assignment to sys.executable.

  extracted_path, _ = os.path.splitext(zip_file_path)

  try:
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
      zip_file.extractall(extracted_path)
      _patch_main_python_file(os.path.join(extracted_path, "__main__.py"))

    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
      for root, _, files in os.walk(extracted_path):
        for file in files:
          abs_path = os.path.join(root, file)
          arc_path = os.path.relpath(abs_path, extracted_path)
          zip_file.write(abs_path, arc_path)
  finally:
    shutil.rmtree(extracted_path)

def main():
  args = _parse_args()

  test_groups = [
      (
          f"//{args.test_prefix}/tensorflow/python/...",
          TargetKind.PY_TEST,
          args.test_prefix,
          [
              "-tpu",
              "-no_dml",
              "-no_oss",
              "-no_pip",
              "-benchmark-test",
              "-manual",
          ]
      ),
      (
          f"//tensorflow/core/...",
          TargetKind.CC_TEST,
          None,
          [
              "-tpu",
              "-no_dml",
              "-no_oss",
              "-benchmark-test",
              "-manual",
              "dml",
          ]
      ),
  ]

  for _, _, _, tag_filters in test_groups:
    if os.name == "nt":
      tag_filters.append("-no_windows")

  for (target,
       target_kind,
       test_prefix,
       tag_filters) in test_groups:
    with BazelEnv(
        args.source_root,
        args.build_output,
        test_prefix) as bazel_env:

      tests_info = bazel_env.get_tests_info(
          target_bazel_path=target,
          target_kind=target_kind,
          tag_filters=tag_filters)

      if not os.path.exists(args.destination):
        os.makedirs(args.destination, exist_ok=True)
      elif not os.path.isdir(args.destination):
        raise Exception(f"'{args.destination}' is not a directory.")

      # Gather the tests binaries
      for test_info in tests_info:
        if not os.path.exists(test_info.exe_path):
          raise Exception(f"'{test_info.exe_path}' doesn't exist.")

        target_destination = os.path.join(args.destination,
                                          test_info.rel_folder_path)

        os.makedirs(target_destination, exist_ok=True)

        os.chmod(test_info.exe_path, 0o777)
        shutil.copy(test_info.exe_path, target_destination)

        if os.path.exists(test_info.runfiles_path):
          if os.name == "nt":
            os.chmod(test_info.runfiles_path, 0o777)
            shutil.copy(test_info.runfiles_path, target_destination)

            # Patch the zip file
            zip_file_name = os.path.split(test_info.runfiles_path)[1]
            new_zip_path = os.path.join(target_destination, zip_file_name)
            _patch_archive(new_zip_path)
          else:
            runfiles_name = os.path.split(test_info.runfiles_path)[-1]
            new_runfiles_path = os.path.join(target_destination, runfiles_name)
            shutil.copytree(test_info.runfiles_path, new_runfiles_path)

            # Patch the python executable
            exe_name = os.path.split(test_info.exe_path)[-1]
            new_exe_path = os.path.join(target_destination, exe_name)
            _patch_main_python_file(new_exe_path)

        # Create a json file with the parameters
        if test_info.params:
          _, exe_name = os.path.split(test_info.exe_path)
          json_name = os.path.splitext(exe_name)[0] + ".json"
          json_path = os.path.join(target_destination, json_name)

          with open(json_path, 'w') as params_file:
            json.dump(test_info.params, params_file)

        # Gather the test data
        for test_data_path in test_info.test_data_paths:
          if not os.path.exists(test_data_path):
            raise Exception(f"'{test_data_path}' doesn't exist.")

          # The test data needs to preserve its directory structure
          file_rel_path = os.path.relpath(test_data_path, args.source_root)
          dest_path = os.path.join(target_destination, file_rel_path)
          os.makedirs(os.path.dirname(dest_path), exist_ok=True)
          shutil.copy(test_data_path, dest_path)

if __name__ == "__main__":
  main()
