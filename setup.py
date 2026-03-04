"""Build script for the ane package.

Compiles bridge/ane_bridge.m -> libane.so, then builds
the Cython extension ane._bridge that links against it.
"""

import os
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_SRC = os.path.join(ROOT_DIR, "src", "ane", "bridge", "ane_bridge.m")
BRIDGE_HDR = os.path.join(ROOT_DIR, "src", "ane", "bridge", "ane_bridge.h")
LIB_NAME = "libane.so"


class build_ext(_build_ext):
    """Custom build_ext that compiles ane_bridge.m -> libane.so first."""

    def _compile_bridge(self, output_dir: str) -> str:
        """Compile the Objective-C bridge into a shared library (.so).

        Returns the path to the built shared library.
        """
        shared_lib_path = os.path.join(output_dir, LIB_NAME)

        # Check if rebuild is needed
        if os.path.exists(shared_lib_path):
            shared_lib_mtime = os.path.getmtime(shared_lib_path)
            src_mtime = max(os.path.getmtime(BRIDGE_SRC), os.path.getmtime(BRIDGE_HDR))
            if shared_lib_mtime > src_mtime:
                return shared_lib_path

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "xcrun",
            "clang",
            "-dynamiclib",
            "-O3",
            "-Wall",
            "-Wno-deprecated-declarations",
            "-fobjc-arc",
            "-fPIC",
            "-install_name",
            f"@rpath/{LIB_NAME}",
            "-o",
            shared_lib_path,
            BRIDGE_SRC,
            "-framework",
            "Foundation",
            "-framework",
            "IOSurface",
            "-ldl",
        ]
        print(f"Building {LIB_NAME}: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return shared_lib_path

    def build_extensions(self):
        # Place the shared library next to the source extension (works for editable and regular installs)
        src_pkg_dir = os.path.join(ROOT_DIR, "src", "ane")
        os.makedirs(src_pkg_dir, exist_ok=True)

        shared_lib_path = self._compile_bridge(src_pkg_dir)

        # Also place in build_lib for wheel installs
        if self.build_lib:
            build_pkg_dir = os.path.join(self.build_lib, *self.extensions[0].name.rsplit(".", 1)[0].split("."))
            os.makedirs(build_pkg_dir, exist_ok=True)
            build_shared_lib = os.path.join(build_pkg_dir, LIB_NAME)
            if not os.path.exists(build_shared_lib) or os.path.getmtime(shared_lib_path) > os.path.getmtime(
                build_shared_lib
            ):
                import shutil

                shutil.copy2(shared_lib_path, build_shared_lib)

        # Patch all extensions to link against the shared library
        for ext in self.extensions:
            ext.library_dirs.append(os.path.dirname(shared_lib_path))

        super().build_extensions()

    def get_outputs(self):
        outputs = super().get_outputs()
        # Include the shared library in the outputs so it gets installed
        ext = self.extensions[0]
        ext_dir = os.path.join(self.build_lib, *ext.name.rsplit(".", 1)[0].split("."))
        shared_lib_path = os.path.join(ext_dir, LIB_NAME)
        if shared_lib_path not in outputs:
            outputs.append(shared_lib_path)
        return outputs


extensions = [
    Extension(
        "ane._bridge",
        sources=[os.path.join("src", "ane", "_bridge.pyx")],
        include_dirs=[
            os.path.normpath(os.path.join(ROOT_DIR, "src", "ane", "bridge")),
            numpy.get_include(),
        ],
        libraries=["ane"],
        extra_link_args=["-Wl,-rpath,@loader_path"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    cmdclass={"build_ext": build_ext},
    package_data={"ane": [LIB_NAME]},
)
