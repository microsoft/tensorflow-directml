"""Loads the DirectML redistributable library"""

def _directml_nuget_repository_impl(repository_ctx):
  # Download NuGet CLI executable. WSL builds can use interop to invoke this executable
  # directly, so it "just works" on both Windows and Linux.
  nuget_path = repository_ctx.path("nuget.exe")
  repository_ctx.download(
    url = "https://dist.nuget.org/win-x86-commandline/v5.7.0/nuget.exe", 
    sha256 = "ae3bb02517b52a744833a4777e99d647cd80b29a62fd360e9aabaa34f09af59c",
    output = nuget_path,
    executable = True,
  )

  # On Linux, NuGet has trouble installing to a WSL path; instead, install to
  # a Windows path and copy it.
  output_directory = repository_ctx.path("")
  if repository_ctx.os.name == "linux":
    # Get path to %LOCALAPPDATA%\tfdml_redist in Windows
    cmd = ["powershell.exe", "echo $env:LOCALAPPDATA"]
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
      fail("Checking LOCALAPPDATA path failed: %s (%s)" % (result.stderr, " ".join(cmd)))
    install_directory = "{}\\tfdml_redist".format(result.stdout.strip())
    
    # Convert path to Linux path
    cmd = ["wslpath", "{}".format(install_directory)]
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
      fail("Converting install_directory path failed: %s (%s)" % (result.stderr, " ".join(cmd)))
    install_directory_wsl = result.stdout.strip()
  else:
    install_directory = output_directory

  # Install the DirectML NuGet package.
  cmd = [
    nuget_path,
    "install",
    "-Source", repository_ctx.attr.source,
    "-Version", repository_ctx.attr.version,
    "-OutputDirectory", install_directory,
    "-ExcludeVersion",
    repository_ctx.attr.package,
  ]
  result = repository_ctx.execute(cmd)
  if result.return_code != 0:
    fail("Downloading DirectML failed: %s (%s)" % (result.stderr, " ".join(cmd)))

  if repository_ctx.os.name == "linux":
    cmd = ["cp", "-R", "{}/{}".format(install_directory_wsl, repository_ctx.attr.package), output_directory]
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
      fail("Copying DirectML redist files failed: %s (%s)" % (result.stderr, " ".join(cmd)))

  # Overlay bazel BUILD file on top of the extracted DirectML NuGet.
  build_file_path = repository_ctx.path(repository_ctx.attr.build_file)
  repository_ctx.symlink(build_file_path, "BUILD")

dml_repository = repository_rule(
    implementation = _directml_nuget_repository_impl,
    attrs = {
      "source" : attr.string(mandatory=True),
      "version" : attr.string(mandatory=True),
      "package" : attr.string(mandatory=True),
      "build_file" : attr.label(),
    }
)