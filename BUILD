load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

cmake_external(
   name = "openblas",
   cache_entries = {
       "NOFORTRAN": "on",
       "BUILD_WITHOUT_LAPACK": "no",
   },
   lib_source = "@openblas//:all",
   static_libraries = ["libopenblas.a"],
   visibility = ["//visibility:public"],
)

cmake_external(
   name = "eigen",
   cache_entries = {
       "BLA_VENDOR": "OpenBLAS",
       "BLAS_LIBRARIES": "$EXT_BUILD_DEPS/openblas/lib/libopenblas.a",
   },
   headers_only = True,
   lib_source = "@eigen//:all",
   # Dependency on other cmake_external rule; can also depend on cc_import, cc_library rules
   deps = [":openblas"],
   visibility = ["//visibility:public"],
)
