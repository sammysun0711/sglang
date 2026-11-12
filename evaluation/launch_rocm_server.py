"""Launch with the system ROCm runtime before Torch loads its bundled libraries.

ROCm 7.2.4 contains graph activity-delivery fixes needed for complete ROCTracer
traces. This entrypoint also runs during multiprocessing spawn, so scheduler
workers select the same runtime.
"""

import ctypes
import os
from pathlib import Path
import runpy
import sys


def load_system_rocm_runtime():
    if "torch" in sys.modules:
        raise RuntimeError("Select the ROCm runtime before importing Torch.")

    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
    library_dir = (root / "lib").resolve()
    names = (
        "librocprofiler-register.so",
        "libhsa-runtime64.so",
        "libamdhip64.so",
        "libroctracer64.so",
        "libroctx64.so",
    )
    # Use the same unversioned names as Torch's DT_NEEDED dependencies. Loading
    # absolute paths alone allows Torch's $ORIGIN RPATH to load a second copy.
    # The existing ROCm environment must resolve these names to the system set.
    handles = [ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL) for name in names]
    mapped = {name: set() for name in names}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        for name in names:
            if path.name.startswith(name):
                mapped[name].add(path.resolve())
    if any(
        paths != {(library_dir / name).resolve()}
        for name, paths in mapped.items()
    ):
        raise RuntimeError(
            f"The ROCm loader must resolve all runtime libraries from {library_dir}; "
            f"observed {mapped}. Check the existing ROCm library search path."
        )
    return handles, root


# Keep handles alive and execute this before importing SGLang, including in
# spawned workers. Do not initialize a GPU here; the scheduler owns device setup.
_runtime_handles, _runtime_root = load_system_rocm_runtime()

if __name__ == "__main__":
    print(f"Using system ROCm runtime: {_runtime_root}", file=sys.stderr, flush=True)
    # Leave this bootstrap as multiprocessing's main module so it runs first
    # in children; run_module's default alter_sys=False preserves that identity.
    runpy.run_module("sglang.launch_server", run_name="__main__")
