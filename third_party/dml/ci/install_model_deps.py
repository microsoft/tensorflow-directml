# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Install the required dependencies to run the model tests."""

import subprocess
import json
import sys
import os

def main():
  with open(os.path.join(sys.path[0], "run_tests_data.json")) as json_file:
    data = json.load(json_file)

  json_test_group = data["testGroups"]["models"]

  deps = []

  if "pipDeps" in json_test_group:
    deps.extend(json_test_group["pipDeps"])

  if os.name == "nt" and "windowsPipDeps" in json_test_group:
    deps.extend(json_test_group["windowsPipDeps"])

  if deps:
    subprocess.run(f"pip install {' '.join(deps)}", shell=True, check=True)

if __name__ == "__main__":
  main()
