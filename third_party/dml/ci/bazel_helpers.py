#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wrappers and context manager helpers around bazel commands."""

import subprocess
import shutil
import os
import xml.etree.ElementTree as ET
from distutils.version import StrictVersion
from enum import Enum
from collections import namedtuple

class PythonVersion(Enum):
  PY2 = 1
  PY3 = 2

class TargetKind(Enum):
  PY_TEST = 1
  CC_TEST = 2

BazelTestInfo = namedtuple(
    "BazelTestInfo",
    "exe_path, runfiles_path, params, rel_folder_path, test_data_paths")

def _stringify_target_kind(target_kind):
  if target_kind == TargetKind.PY_TEST:
    return "py_test"

  if target_kind == TargetKind.CC_TEST:
    return "cc_test"

  raise Exception(f"Stringify for {target_kind} not implemented.")

class BazelEnv:
  """
  Helper to easily query tensorflow's bazel project and gather information on
  the tests. It's also a context manager that automatically creates and cleanups
  the tensorflow symlink.
  """

  def __init__(self, source_path, build_path, test_prefix=None):
    self.source_path = os.path.abspath(source_path)
    self.build_path = os.path.abspath(build_path)
    self.test_prefix = test_prefix

    if test_prefix:
      self.test_path = os.path.join(self.source_path, test_prefix)
      self.symlink_path = os.path.join(self.test_path, "tensorflow")
      self.symlink_target = os.path.join(self.source_path, "tensorflow")
    else:
      self.test_path = self.source_path

    self._py2_bin_path = None
    self._py3_bin_path = None

  def __enter__(self):
    # No need to create a symlink if the test path is the same as the source
    # path
    if self.source_path == self.test_path:
      return self

    os.mkdir(self.test_path)

    try:
      # We need to create a symlink before querying the tests. Otherwise, the
      # folder that will be included in the tests' runfiles files will be named
      # 'tensorflow', which will conflict when trying to import the real
      # tensorflow module.
      if os.name == "nt":
        # os.symlink throws a PermissionError on Windows, even if the script is
        # launched from an elevated prompt
        subprocess.run(
            ["mklink", "/J", self.symlink_path, self.symlink_target],
            shell=True,
            check=True)
      else:
        os.symlink(self.symlink_target, self.symlink_path)
    except Exception:
      shutil.rmtree(self.test_path)
      raise

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # No symlink was created if the test path is the same as the source path
    if self.source_path == self.test_path:
      return

    if os.name == "nt":
      # rmdir removes the symlink without deleting the files, as opposed to
      # rmtree that deletes the files pointed to by the symlink
      subprocess.run(
          ["rmdir", self.symlink_path],
          shell=True,
          check=True)
    else:
      os.unlink(self.symlink_path)

    shutil.rmtree(self.test_path)

  def _get_bin_path(self, py_version):
    """Returns the bin path corresponding to the tests' python version."""

    # We lazily initialize _py2_bin_path and _py3_bin_path
    if py_version == PythonVersion.PY2 and self._py2_bin_path is not None:
      return self._py2_bin_path

    if py_version == PythonVersion.PY3 and self._py3_bin_path is not None:
      return self._py3_bin_path

    bazel_version = self.get_version()

    # Before bazel 0.25.0, the default bin folder is py2 and py3 targets are
    # built in a folder with the '-py3' suffix. Since 0.25.0, it's the other way
    # around: https://github.com/bazelbuild/bazel/issues/7593
    if StrictVersion(bazel_version) < StrictVersion("0.25.0"):
      bazel_default_py_version = PythonVersion.PY2
    else:
      bazel_default_py_version = PythonVersion.PY3

    bin_path_process = subprocess.run(
        [
            "bazel",
            f"--output_user_root={self.build_path}",
            "info",
            "bazel-bin"
        ],
        check=True,
        stdout=subprocess.PIPE)

    bin_path = bin_path_process.stdout.decode("utf-8").rstrip()

    if py_version != bazel_default_py_version:
      bin_suffix = "-py3" if py_version == PythonVersion.PY3 else "-py2"

      bin_path_head, bin_path_tail = os.path.split(bin_path)
      bin_path_head, flavor = os.path.split(bin_path_head)
      flavor_parts = flavor.split("-")

      flavor_parts[0] += bin_suffix
      flavor = "-".join(flavor_parts)

      bin_path = os.path.join(bin_path_head, flavor, bin_path_tail)

    if py_version == PythonVersion.PY2:
      self._py2_bin_path = bin_path
    elif py_version == PythonVersion.PY3:
      self._py3_bin_path = bin_path
    else:
      raise Exception("Python version is not lazily inialized.")

    return bin_path

  def _bazel_to_source_path(self, bazel_path, extension=""):
    """
    Transforms a bazel path (e.g. //tensorflow/python) do an absolute path that
    points to tensorflow's source directory.
    """

    if bazel_path[:2] != "//":
      raise Exception(
          f"Unexpected prefix for the bazel path. Expected '//', but found"
          f" '{bazel_path[:2]}'.")

    colon_index = bazel_path.index(":")
    relative_path = bazel_path[2:colon_index].split("/")
    relative_path.append(bazel_path[colon_index + 1:] + extension)

    return os.path.join(self.source_path, os.path.join(*relative_path))

  def _bazel_to_build_path(
      self,
      bazel_path,
      bin_path,
      extension=""):
    """
    Transforms a bazel path (e.g. //tensorflow/python) do an absolute path that
    points to tensorflow's build directory.
    """

    if bazel_path[:2] != "//":
      raise Exception(
          f"Unexpected prefix for the bazel path. Expected '//', but found"
          f" '{bazel_path[:2]}'.")

    colon_index = bazel_path.index(":")
    relative_path = bazel_path[2:colon_index].split("/")
    relative_path.append(bazel_path[colon_index + 1:] + extension)

    return os.path.join(bin_path, os.path.join(*relative_path))

  def _get_tests_info(
      self,
      target_bazel_path,
      py_version,
      target_kind,
      include_attrs=(),
      exclude_attrs=()):
    """Gathers the paths to bazel tests based on attributes."""

    command = ["bazel", "query"]

    selectors = f"kind({_stringify_target_kind(target_kind)}, "
    selectors_end = f"{target_bazel_path})"

    # Append the attributes to include in the search
    for attr_name, attr_value in include_attrs:
      selectors += rf"attr({attr_name}, '\b{attr_value}\b', "
      selectors_end += ")"

    selectors += selectors_end
    command.append(selectors)

    # Append the attributes to exclude from the search
    for attr_name, attr_value in exclude_attrs:
      command.append("except")
      command.append(
          rf"attr({attr_name}, '\b{attr_value}\b', {target_bazel_path})")

    command.append("--output=xml")

    xml_string = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE).stdout.decode("utf-8")

    xml_root = ET.fromstring(xml_string)

    # Convert the bazel path to relative filesystem paths
    tests_info = []
    bazel_paths = set()

    for rule in xml_root.iter("rule"):
      # Save the executable path in the set so we can quickly check if
      # executables with the _gpu suffix have an equivalent executable without
      # the _gpu suffix
      bazel_path = rule.get("name")
      bazel_paths.add(bazel_path)

    bin_path = self._get_bin_path(py_version)

    for rule in xml_root.iter("rule"):
      # Get the executable and runfiles file paths
      bazel_path = rule.get("name")

      # The binaries that end with "_gpu" contain the same tests as the ones
      # without "_gpu", except that they have benchmarking enabled. Since we
      # don't need these benchmarking executables for our test loop, keeping
      # them doesn't give us anything.
      if bazel_path.endswith("_gpu") and bazel_path[:-4] in bazel_paths:
        continue

      if os.name == "nt":
        exe_path = self._bazel_to_build_path(bazel_path, bin_path, ".exe")
        runfiles_path = self._bazel_to_build_path(bazel_path, bin_path, ".zip")
      else:
        exe_path = self._bazel_to_build_path(bazel_path, bin_path)
        runfiles_path = self._bazel_to_build_path(bazel_path,
                                                  bin_path,
                                                  ".runfiles")

      data_element = rule.find('list[@name="data"]')
      test_data_bazel_paths = []

      # Get the test data
      if data_element is not None:
        for data_element in rule.find('list[@name="data"]').iter("label"):
          test_data_bazel_path = data_element.get("value")
          if test_data_bazel_path.endswith("_testdata"):
            test_data_bazel_paths.append(test_data_bazel_path)

      # Get the relevant parameters
      rule_params = {}
      shard_count_element = rule.find('int[@name="shard_count"]')
      if shard_count_element is not None:
        rule_params["shard_count"] = int(shard_count_element.get("value"))

      abs_folder_path = os.path.split(exe_path)[0]
      root_path = os.path.join(bin_path, self.test_prefix or "", "tensorflow")

      test_data_paths = []

      if test_data_bazel_paths:
        test_data_paths = self._get_test_data_paths(test_data_bazel_paths)

      tests_info.append(
          BazelTestInfo(
              rel_folder_path=os.path.relpath(abs_folder_path, root_path),
              exe_path=exe_path,
              runfiles_path=runfiles_path,
              test_data_paths=test_data_paths,
              params=rule_params))

    return tests_info

  def _get_test_data_paths(self, bazel_targets):
    """
    Queries the bazel targets and retrieves absolute paths to the test data
    files.
    """

    test_data_paths = []

    for bazel_target in bazel_targets:
      test_data_query_process = subprocess.run(
          ["bazel", "query", bazel_target, "--output=xml"],
          stdout=subprocess.PIPE,
          check=True)

      xml_string = test_data_query_process.stdout.decode("utf-8")
      xml_root = ET.fromstring(xml_string)

      for rule in xml_root.iter("rule"):
        for file_element in rule.find('list[@name="srcs"]').iter("label"):
          file_bazel_path = file_element.get("value")
          file_abs_path = self._bazel_to_source_path(file_bazel_path)
          test_data_paths.append(file_abs_path)

    return test_data_paths

  def get_version(self):
    version_process = subprocess.run(
        ["bazel", "version", "--gnu_format"],
        check=True,
        stdout=subprocess.PIPE)

    return version_process.stdout.decode("utf-8").rstrip().split()[-1]

  def get_tests_info(self, target_bazel_path, target_kind, tag_filters=()):
    """Gathers the paths to bazel tests based on tags."""

    exclude_attrs = []
    include_attrs = []

    for tag_filter in tag_filters:
      if tag_filter[0] == "-":
        exclude_attrs.append(("tags", tag_filter[1:]))
      else:
        include_attrs.append(("tags", tag_filter))

    py2_tests_info = self._get_tests_info(
        target_bazel_path=target_bazel_path,
        py_version=PythonVersion.PY2,
        target_kind=target_kind,
        include_attrs=include_attrs,
        exclude_attrs=exclude_attrs + [("python_version", "PY3")])

    py3_tests_info = self._get_tests_info(
        target_bazel_path=target_bazel_path,
        py_version=PythonVersion.PY3,
        target_kind=target_kind,
        include_attrs=include_attrs + [("python_version", "PY3")],
        exclude_attrs=exclude_attrs)

    return py2_tests_info + py3_tests_info
