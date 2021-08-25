"""Loads the DirectML redistributable library"""

def _directml_nuget_repository_impl(repository_ctx):
  # Download and extract .nupkg in separate steps because bazel does not
  # recognize .nupkg as an archive format.
  archive_path = repository_ctx.path("directml/directml.zip")
  repository_ctx.download(
    url = repository_ctx.attr.url, 
    sha256 = repository_ctx.attr.sha256,
    output = archive_path,
    executable = True,
  )
  repository_ctx.extract(archive = archive_path, output = "directml/")

  # Overlay bazel BUILD file on top of the extracted DirectML NuGet.
  build_file_path = repository_ctx.path(repository_ctx.attr.build_file)
  repository_ctx.symlink(build_file_path, "BUILD")

dml_repository = repository_rule(
    implementation = _directml_nuget_repository_impl,
    attrs = {
      "url" : attr.string(mandatory=True),
      "sha256" : attr.string(mandatory=True),
      "build_file" : attr.label(),
    }
)