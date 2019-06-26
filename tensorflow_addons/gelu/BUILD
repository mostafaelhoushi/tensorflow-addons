licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "gelu",
    srcs = ([
        "__init__.py",
        "gelu.py",
    ]),
    data = [
        "//tensorflow_addons/custom_ops/gelu:_gelu_ops.so",
        "//tensorflow_addons/utils",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "gelu_test",
    size = "small",
    srcs = [
        "gelu_test.py",
    ],
    main = "gelu_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":gelu",
    ],
)