"""Loads the WinPixEventRuntime redistributable library"""

def _pix_nuget_repository_impl(repository_ctx):
  # Download NuGet CLI executable.
  nuget_path = repository_ctx.path("nuget.exe")
  repository_ctx.download(
    url = "https://dist.nuget.org/win-x86-commandline/v5.7.0/nuget.exe", 
    sha256 = "ae3bb02517b52a744833a4777e99d647cd80b29a62fd360e9aabaa34f09af59c",
    output = nuget_path,
    executable = True,
  )

  # Install the PIX NuGet package.
  output_directory = repository_ctx.path("")
  cmd = [
    nuget_path,
    "install",
    "-Source", repository_ctx.attr.source,
    "-Version", repository_ctx.attr.version,
    "-OutputDirectory", output_directory,
    "-ExcludeVersion",
    repository_ctx.attr.package,
  ]
  result = repository_ctx.execute(cmd)
  if result.return_code != 0:
    fail("Downloading WinPixEventRuntime failed: %s (%s)" % (result.stderr, " ".join(cmd)))

  # Overlay bazel BUILD file on top of the extracted NuGet.
  build_file_path = repository_ctx.path(repository_ctx.attr.build_file)
  repository_ctx.symlink(build_file_path, "BUILD")

pix_repository = repository_rule(
    implementation = _pix_nuget_repository_impl,
    attrs = {
      "source" : attr.string(mandatory=True),
      "version" : attr.string(mandatory=True),
      "package" : attr.string(mandatory=True),
      "build_file" : attr.label(),
    }
)