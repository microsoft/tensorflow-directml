#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All Rights Reserved.
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

"""Helper script to build tensorflow packages and tests."""

import os
import sys
import site
import argparse
import subprocess
import shutil
import configure as tf_configure
from pathlib import Path

# Returns the path to .tf_configure.bazelrc, which is generated using configure.py.
def tf_configure_path():
  return os.path.join(
      os.path.dirname(os.path.realpath(__file__)),
      ".tf_configure.bazelrc")


# Returns true if configure.py needs to be re-run, or the user has
# explicitly asked for clean build.
def needs_clean(args):
  return args.clean or not os.path.isfile(tf_configure_path())


def clean(args):
  if os.path.isfile(tf_configure_path()):
    os.remove(tf_configure_path())

  subprocess.run(
      "bazel --output_user_root={} clean".format(args.build_output),
      shell=True,
      check=True)


def configure(args):
  """Runs configure.py with default options."""

  os.environ["PYTHON_BIN_PATH"] = sys.executable
  os.environ["PYTHON_LIB_PATH"] = site.getsitepackages()[-1]
  os.environ["TF_ENABLE_XLA"] = "0"
  os.environ["TF_NEED_ROCM"] = "0"
  os.environ["TF_NEED_OPENCL_SYCL"] = "0"
  os.environ["TF_NEED_MPI"] = "0"
  os.environ["TF_DOWNLOAD_CLANG"] = "0"
  os.environ["TF_SET_ANDROID_WORKSPACE"] = "0"
  os.environ["TF_NEED_CUDA"] = "0"

  # These defaults are just to suppress the interactive prompt and
  # specific to platform. We should adjust these in the future.
  if sys.platform == "win32":
    os.environ["CC_OPT_FLAGS"] = "/arch:AVX"
  elif sys.platform == "linux":
    os.environ["CC_OPT_FLAGS"] = "-mavx"
    # GCC doesn't support some of the MS extensions we rely on, such as __declspec(uuid(x)).
    # Setting this var will switch over to building with Clang (still uses GNU C/C++ libs).
    os.environ["TF_DOWNLOAD_CLANG"] = "1"

  # Quicker builds but slower CPU kernels. Revisit in the future.
  os.environ["TF_OVERRIDE_EIGEN_STRONG_INLINE"] = "1"

  # Invoke configure.py, but strip the command-line args that are
  # specific to dml_build.py (they will not be recognized and break).
  prev_argv = sys.argv
  sys.argv = [sys.argv[0]]
  tf_configure.main()
  sys.argv = prev_argv


def build_tests(args):
  """Builds the tests and their dependencies."""

  absolute_dir = os.path.dirname(os.path.realpath(__file__))
  absolute_test_prefix = os.path.join(absolute_dir, args.test_prefix)
  os.mkdir(absolute_test_prefix)
  symlink_path = os.path.join(absolute_test_prefix, "tensorflow")
  symlink_target = os.path.join(absolute_dir, "tensorflow")

  try:
    # We need to create a symlink before building the tests. Otherwise, the
    # folder that will be included in the tests' zip files will be named
    # 'tensorflow', which will conflict when trying to import the real
    # tensorflow module.
    if os.name == "nt":
      # os.symlink throws a PermissionError on Windows, even if the script is
      # launched from an elevated prompt
      subprocess.run(
          ["mklink", "/J", symlink_path, symlink_target],
          shell=True,
          check=True)
    else:
      os.symlink(symlink_target, symlink_path)

    targets_and_tag_filters = [
        (
            "//{}/tensorflow/python/...".format(args.test_prefix),
            [
                "-tpu",
                "-no_dml",
                "-no_oss",
                "-no_pip",
                "-benchmark-test",
                "-manual",
            ],
        ),
        (
            "//tensorflow/core/...",
            [
                "-tpu",
                "-no_dml",
                "-no_oss",
                "-benchmark-test",
                "-manual",
                "dml",
            ],
        )
    ]

    for _, tag_filters in targets_and_tag_filters:
      if os.name == "nt":
        tag_filters.append("-no_windows")

    cl = ["bazel"]
    cl.append("--output_user_root={}".format(args.build_output))
    cl.append("build")
    if args.subcommands:
      cl.append("--subcommands")
    cl.append("--config=opt")
    cl.append("--config=dml")
    if args.telemetry:
      cl.append("--config=dml_telemetry")
    cl.append("--test_lang_filters=py,cc")
    cl.append("--verbose_failures")
    cl.append("--build_tests_only")
    cl.append("--define=no_tensorflow_py_deps=true")
    if args.config == "debug":
      cl.append("--strip never")
    if args.config == "debug" and sys.platform == "win32":
      cl.append("--copt /wd4716")
      cl.append("--copt /Z7")
      cl.append("--copt /FS")
      cl.append("--linkopt /DEBUG:FASTLINK")

    # This is necessary because of name clashes when bazel tries to copy 2 DLLs
    # with the same name but different paths into the binary folder. This
    # doesn't affect the python package, but it's required to reliably build the
    # core tests.
    # https://github.com/bazelbuild/bazel/issues/11515
    if sys.platform == "win32":
      cl.append("--dynamic_mode=off")

    # For now, since we only want to run the core tests tagged as "dml", we need
    # to run the command multiple times. Even though bazel allows multiple
    # targets to be built from the same command, they will have the same
    # filtered tags. We want to build ALL the python tests, and not only the
    # ones tagged as "dml". The overhead should be very small since all the
    # other parameters are the same.
    for target, tag_filters in targets_and_tag_filters:
      target_cl = cl.copy()
      target_cl.append("--build_tag_filters={}".format(','.join(tag_filters)))
      target_cl.append("--test_tag_filters={}".format(','.join(tag_filters)))
      target_cl.append(target)
      subprocess.run(" ".join(target_cl), shell=True, check=True)
  finally:
    if os.name == "nt":
      # rmdir removes the symlink without deleting the files, as opposed to
      # rmtree that deletes all the files
      subprocess.run(
          ["rmdir", symlink_path],
          shell=True,
          check=True)
    else:
      os.unlink(symlink_path)

    shutil.rmtree(absolute_test_prefix)

def build(args):
  """Runs bazel to build TensorFlow's build_pip_package target."""

  cl = ["bazel"]
  cl.append("--output_user_root={}".format(args.build_output))
  cl.append("build")
  if args.force_debug:
    cl.append("-c dbg")
  if args.subcommands:
    cl.append("--subcommands")
  cl.append("--config=opt")
  cl.append("--config=dml")
  if args.telemetry:
    cl.append("--config=dml_telemetry")
  if args.config == "debug":
    cl.append("--strip never")
  if args.config == "debug" and sys.platform == "win32":
    cl.append("--copt /wd4716")
    cl.append("--copt /Z7")
    cl.append("--copt /FS")
    cl.append("--linkopt /DEBUG:FASTLINK")
  cl.append(args.target)
  subprocess.run(" ".join(cl), shell=True, check=True)


def create_package(args):
  """
  Creates an installable Python package from TensorFlow's build_pip_package
  target.
  """

  build_pip_package_path = os.path.join(
      os.path.dirname(os.path.realpath(__file__)),
      "bazel-bin",
      "tensorflow",
      "tools",
      "pip_package",
      "build_pip_package")

  if sys.platform == "win32":
    build_pip_package_path += ".exe"

  dst_path = os.path.join(args.build_output, "python_package")

  cl = [build_pip_package_path, "--dst", dst_path, "--directml"]

  subprocess.run(
      " ".join(cl),
      shell=True,
      check=True)


def install_package(args):
  """Installs the generated TensorFlow Python package."""

  package_dir = os.path.join(args.build_output, "python_package")

  # Find the most recently created package
  package_path = sorted(Path(package_dir).iterdir(), key=os.path.getmtime)[-1].as_posix()

  # Check if tensorflow is already installed
  reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
  installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

  # Install the package
  if "tensorflow-directml" in installed_packages:
    subprocess.run(
        ["pip", "install", "--force-reinstall", "--no-deps", package_path],
        check=True)
  else:
    subprocess.run(["pip", "install", package_path], check=True)


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--config", "-c",
      choices=("debug", "release"),
      default="debug",
      help="Build configuration.")

  parser.add_argument(
      "--clean", "-x",
      action="store_true",
      help="Configure and build from scratch.")

  parser.add_argument(
      "--package", "-p",
      action="store_true",
      help="Create Python package.")

  parser.add_argument(
      "--tests",
      action="store_true",
      help="Build tests.")

  parser.add_argument(
      "--test_prefix",
      default="tfdml_test",
      help="Prefix to use when creating a symlink to build the tests.")

  parser.add_argument(
      "--install", "-i",
      action="store_true",
      help="Install Python package using pip.")

  parser.add_argument(
      "--target", "-t",
      default="//tensorflow/tools/pip_package:build_pip_package",
      help="Bazel target to build.")

  parser.add_argument(
      "--subcommands", "-s",
      action="store_true",
      help="Display the subcommands executed during a build (e.g. compiler command line invocations).")

  parser.add_argument(
      "--force_debug", "-d",
      action="store_true",
      help="Build with dbg compilation mode. Useful for debugging individual targets on Linux (py package is too large).")

  parser.add_argument(
      "--telemetry",
      action="store_true",
      help="Allow builds to emit telemetry associated with the DMLTF client hint.")

  # Default to storing build output under <repo_root>/../tfdml_build/.
  default_build_output = os.path.join(
      os.path.dirname(os.path.realpath(__file__)),
      "..",
      "dml_build")

  parser.add_argument(
      "--build_output", "-o",
      default=default_build_output,
      help="Build output path. Defaults to {}.".format(default_build_output))

  args = parser.parse_args()

  # Clean
  if needs_clean(args):
    clean(args)

  # Configure
  if not os.path.isfile(tf_configure_path()):
    configure(args)

  # Build
  build(args)

  # Create Python package
  if args.package:
    create_package(args)
    if args.install:
      install_package(args)

  # Build the tests
  if args.tests:
    build_tests(args)

if __name__ == "__main__":
  main()
