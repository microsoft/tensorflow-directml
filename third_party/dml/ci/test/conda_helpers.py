#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wrappers and context manager helpers around conda commands."""

import subprocess
import shutil
import os
import json

class CondaEnv:
  """Context Manager to automatically cleanup a temporary conda environment when
  an exception occurs or it's not needed anymore.
  """

  def __init__(self, env_name):
    self.env_name = env_name

    if os.name == "nt":
      self.hook_command = "conda_hook"
    else:
      self.hook_command = 'eval "$(conda shell.bash hook)"'

  def __enter__(self):
    subprocess.run(
        f"{self.hook_command}"
        f" && conda create python=3.6 -y -n {self.env_name}",
        shell=True,
        check=True)

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    conda_info = subprocess.run(
        f"{self.hook_command}"
        f" && conda activate {self.env_name}"
        f" && conda info --json",
        shell=True,
        check=True,
        stdout=subprocess.PIPE).stdout

    subprocess.run(
        f"{self.hook_command}"
        f" && conda env remove -n {self.env_name}",
        shell=True,
        check=True)

    # Some versions of conda packages publish .conda_trash files, which can't
    # be cleaned with "conda env remove"
    json_conda_info = json.loads(conda_info)
    shutil.rmtree(
        json_conda_info["env_vars"]["CONDA_PREFIX"],
        ignore_errors=True)

  def install_package(self, package_name):
    subprocess.run(
        f'{self.hook_command}'
        f' && conda activate {self.env_name}'
        f' && pip install "{package_name}"',
        shell=True,
        check=True)

  def run(self, args, variables=None):
    args_string = " ".join(args)

    env = None

    if variables is not None:
      env = {**os.environ, **variables}

    subprocess.run(
        f"{self.hook_command}"
        f" && conda activate {self.env_name}"
        f" && python {args_string}",
        shell=True,
        check=True,
        env=env)
