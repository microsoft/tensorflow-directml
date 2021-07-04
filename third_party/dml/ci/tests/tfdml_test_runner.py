#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Runs the tensorflow python tests."""

import argparse
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
import tempfile
import json
import re
import glob
from contextlib import ExitStack


def _parse_args():
  """Parses the arguments given to this script."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--test_binaries_path",
      help="path to the directory that contains the test binaries",
      required=True)

  parser.add_argument("--test_framework",
                      choices=["abseil", "gtest"],
                      help="test framework that the test binary is using",
                      required=True)

  parser.add_argument("--log_device_placement", action="store_true")

  parser.add_argument("--test_timeout", default=300, type=int)

  return parser.parse_args()


def _get_tf_env(exe_path, test_framework):
  env_copy = os.environ.copy()

  if os.name != "nt" and test_framework == "gtest":
    exe_path_head = os.path.splitext(exe_path)[0]
    tf_lib_path = f"{exe_path_head}.runfiles/org_tensorflow/tensorflow"
    deps_lib_path = f"{exe_path_head}.runfiles/org_tensorflow/_solib_k8"
    prev_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_library_path = f"{tf_lib_path}:{deps_lib_path}:{prev_library_path}"
    env_copy["LD_LIBRARY_PATH"] = ld_library_path

  return env_copy


def _run_test(exe_path, log_device_placement, shard_index, total_shard_count,
              test_framework, test_timeout):
  """Runs a test executable in its own process."""

  # Insert the shard index in the name of the xml file
  xml_path = f"{os.path.splitext(exe_path)[0]}_{shard_index}_test_result.xml"
  csv_path = f"{os.path.splitext(exe_path)[0]}_{shard_index}_test_placement.csv"

  # Since test data paths are relative, we need to run the executable from the
  # executable's directory
  prev_dir = os.getcwd()
  os.chdir(os.path.dirname(exe_path))

  env_copy = _get_tf_env(exe_path, test_framework)

  if log_device_placement:
    env_copy["TF_CPP_LOG_DEVICE_PLACEMENT"] = "1"

  if os.name != "nt":
    env_copy["TEST_SRCDIR"] = exe_path + ".runfiles"

  if test_framework == "abseil":
    env_copy["TEST_TOTAL_SHARDS"] = str(total_shard_count)
    env_copy["TEST_SHARD_INDEX"] = str(shard_index)
    xml_output_arg = f"--xml_output_file={xml_path}"
  elif test_framework == "gtest":
    env_copy["GTEST_TOTAL_SHARDS"] = str(total_shard_count)
    env_copy["GTEST_SHARD_INDEX"] = str(shard_index)
    xml_output_arg = f"--gtest_output=xml:{xml_path}"
  else:
    raise Exception("Unsupported test framework.")

  env_copy["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  try:
    with tempfile.TemporaryFile(mode="w+", encoding="iso-8859-1") as stdout, \
         tempfile.TemporaryFile(mode="w+", encoding="iso-8859-1") as stderr:
      # We only want to let KeyboardInterrupt exceptions propagate. Other
      # exceptions are test failures that should not make the parent process
      # crash.
      try:
        subprocess.run([exe_path, xml_output_arg],
                       stdout=stdout,
                       stderr=stderr,
                       env=env_copy,
                       stdin=subprocess.DEVNULL,
                       check=True,
                       universal_newlines=True,
                       timeout=test_timeout)
      except KeyboardInterrupt:  # pylint: disable=try-except-raise
        raise
      except Exception:  # pylint: disable=broad-except
        pass
      finally:
        print(f"Running '{exe_path}' with shard {shard_index}...")
        stdout.seek(0)
        stderr.seek(0)
        test_output = stdout.read() + stderr.read()

        if log_device_placement:
          output_lines = test_output.splitlines()
          csv_lines = []
          new_output_lines = []

          for line in output_lines:
            pattern = re.compile(
                r"tensorflow\/core\/common_runtime\/placer\.cc.* .*: \((.*)\): "
                r".*\/device:(.*):\d+")

            match = pattern.search(line)

            if match:
              op_type = match.group(1)
              op_device = match.group(2)
              csv_lines.append(f"{op_type},{op_device}\n")
            else:
              # Tensorflow prints 2 lines for every device placement, which is
              # redundant information. We need to remove both from the logs.
              pattern = re.compile(r".*: \(.*\): .*\/device:.*:\d+")

              if not pattern.search(line):
                new_output_lines.append(line)

          test_output = "\n".join(new_output_lines)

          if csv_lines:
            with open(csv_path, "w+") as csv_file:
              csv_file.writelines(csv_lines)

        print(test_output)
  finally:
    os.chdir(os.path.dirname(prev_dir))


def _is_distribute_test(exe_path):
  temp_path = exe_path

  while temp_path != "":
    temp_path, tail = os.path.split(temp_path)
    if tail == "distribute":
      return True

  return False


def _read_shard_count(exe_path):
  # Read the json file to know how many shards to split the test into
  shard_count = 1
  json_path = os.path.splitext(exe_path)[0] + ".json"

  if os.path.exists(json_path):
    with open(json_path) as json_file:
      params = json.load(json_file)
      if "shard_count" in params:
        shard_count = params["shard_count"]

  return shard_count


def main():
  args = _parse_args()
  absolute_binaries_path = os.path.join(sys.path[0], args.test_binaries_path)
  process_count = min(8, os.cpu_count())

  if os.name == "nt":
    exe_paths = glob.glob(f"{absolute_binaries_path}/**/*.exe", recursive=True)
  else:
    runfiles = glob.glob(f"{absolute_binaries_path}/**/*.runfiles",
                         recursive=True)
    exe_paths = [os.path.splitext(runfile)[0] for runfile in runfiles]

  futures = []

  try:
    with ExitStack() as stack:
      # Distribute tests can launch many servers over a limited number of ports,
      # so run them all in a sequential queue to avoid collisions
      sequential_executor = stack.enter_context(ThreadPoolExecutor(1))
      parallel_executor = stack.enter_context(ThreadPoolExecutor(process_count))

      for exe_path in exe_paths:
        shard_count = _read_shard_count(exe_path)

        for shard_index in range(shard_count):
          # Don't gather device placement data on WSL for now
          log_device_placement = os.name == "nt" and args.log_device_placement

          if _is_distribute_test(exe_path):
            futures.append(
                sequential_executor.submit(_run_test, exe_path,
                                           log_device_placement, shard_index,
                                           shard_count, args.test_framework,
                                           args.test_timeout))
          else:
            futures.append(
                parallel_executor.submit(_run_test, exe_path,
                                         log_device_placement, shard_index,
                                         shard_count, args.test_framework,
                                         args.test_timeout))

      for future in futures:
        future.result()
  finally:
    for future in futures:
      future.cancel()


if __name__ == "__main__":
  main()
