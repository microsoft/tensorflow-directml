#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Runs the tests from the given test groups."""

import argparse
import os
import json
import sys
from conda_helpers import CondaEnv

def _parse_args():
  """Parses the arguments given to this script."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--test_group",
      help="test group to run",
      required=True)

  parser.add_argument(
      "--tensorflow_wheel",
      help=(
          "path to the tensorflow wheel file - if not provided, uses conda's "
          "tensorflow 1.15 package"),
      default=None)

  parser.add_argument(
      "--env",
      help=(
          "the name of the temporary conda environment to create to run the "
          "tests"),
      default="tf_dml_tests")

  return parser.parse_args()

def main():
  args = _parse_args()

  with CondaEnv(args.env) as conda_env:
    with open(os.path.join(sys.path[0], "run_tests_data.json")) as json_file:
      data = json.load(json_file)
      json_test_group = data["testGroups"][args.test_group]
      script = json_test_group["script"]
      is_python_test = json_test_group["isPythonTest"]

      if is_python_test:
        if args.tensorflow_wheel is None:
          conda_env.install_package("tensorflow==1.15")
        else:
          conda_env.install_package(args.tensorflow_wheel)

        if "pipDeps" in json_test_group:
          for dep in json_test_group["pipDeps"]:
            conda_env.install_package(dep)

        if os.name == "nt" and "windowsPipDeps" in json_test_group:
          for dep in json_test_group["windowsPipDeps"]:
            conda_env.install_package(dep)

      command_line = [os.path.join(sys.path[0], "tests", script)]

      if "arguments" in json_test_group:
        for arg_name, arg_value in json_test_group["arguments"].items():
          command_line.append(f"{arg_name}={arg_value}")

      if "flags" in json_test_group:
        for flag in json_test_group["flags"]:
          command_line.append(flag)

      conda_env.run(command_line)

if __name__ == "__main__":
  main()
