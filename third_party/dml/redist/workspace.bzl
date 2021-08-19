"""Loads the DirectML redistributable library"""

def _directml_nuget_repository_impl(repository_ctx):
  download_url = "https://www.nuget.org/api/v2/package/%s/%s" % (
    repository_ctx.attr.package,
    repository_ctx.attr.version
  )

  # Download and extract .nupkg in separate steps because .nupkg
  # is not recognized as an archive format by Bazle.
  nuget_path = repository_ctx.path("directml/directml.zip")
  repository_ctx.download(
    url = download_url, 
    sha256 = repository_ctx.attr.sha256,
    output = nuget_path,
    executable = True,
  )
  repository_ctx.extract(archive = nuget_path)

  # Overlay bazel BUILD file on top of the extracted DirectML NuGet.
  build_file_path = repository_ctx.path(repository_ctx.attr.build_file)
  repository_ctx.symlink(build_file_path, "BUILD")

dml_repository = repository_rule(
    implementation = _directml_nuget_repository_impl,
    attrs = {
      "sha256" : attr.string(mandatory=True),
      "version" : attr.string(mandatory=True),
      "package" : attr.string(mandatory=True),
      "build_file" : attr.label(),
    }
)