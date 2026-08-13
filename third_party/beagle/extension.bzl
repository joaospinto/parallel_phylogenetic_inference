"""Optional local BEAGLE repository for benchmark targets."""

def _first_existing(repository_ctx, paths):
    for path in paths:
        candidate = repository_ctx.path(path)
        if candidate.exists:
            return candidate
    return None

def _beagle_repository_impl(repository_ctx):
    prefix = repository_ctx.os.environ.get("BEAGLE_PREFIX", "")
    if not prefix:
        brew = repository_ctx.which("brew")
        if brew:
            result = repository_ctx.execute([brew, "--prefix", "beagle"])
            if result.return_code == 0:
                prefix = result.stdout.strip()

    header = None
    library = None
    if prefix:
        header = _first_existing(repository_ctx, [
            prefix + "/include/libhmsbeagle-1/libhmsbeagle/beagle.h",
            prefix + "/include/libhmsbeagle/beagle.h",
        ])
        library = _first_existing(repository_ctx, [
            prefix + "/lib/libhmsbeagle.dylib",
            prefix + "/lib/libhmsbeagle.so",
            prefix + "/lib64/libhmsbeagle.so",
            prefix + "/lib/x86_64-linux-gnu/libhmsbeagle.so",
        ])

    if header == None or library == None:
        repository_ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "beagle",
    target_compatible_with = ["@platforms//:incompatible"],
    visibility = ["//visibility:public"],
)
""")
        return

    repository_ctx.symlink(header.dirname.dirname, "include")
    extension = ".dylib" if str(library).endswith(".dylib") else ".so"
    repository_ctx.symlink(library, "lib/libhmsbeagle" + extension)
    repository_ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_import(
    name = "library",
    shared_library = "lib/libhmsbeagle%s",
)

cc_library(
    name = "beagle",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":library"],
)
""" % extension)

_beagle_repository = repository_rule(
    implementation = _beagle_repository_impl,
    environ = ["BEAGLE_PREFIX"],
    local = True,
)

def _beagle_extension_impl(module_ctx):
    _beagle_repository(name = "beagle")

beagle = module_extension(implementation = _beagle_extension_impl)
