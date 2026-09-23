#!/usr/bin/env bash
# Install the JetPack-native TensorRT stack, build the model, and benchmark it.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(uname -m)" != "aarch64" ] || [ ! -r /etc/nv_tegra_release ]; then
    echo "error: this installer must run natively on an NVIDIA Jetson" >&2
    exit 2
fi

echo "Jetson release: $(cat /etc/nv_tegra_release)"
echo "Installing NVIDIA's TensorRT runtime, builder, and Python bindings"
sudo apt-get update
sudo apt-get install -y tensorrt python3-libnvinfer libnvinfer-bin

python3 - <<'PY'
import ctypes
import ctypes.util
import tensorrt as trt

library = ctypes.util.find_library('cudart') or 'libcudart.so'
ctypes.CDLL(library)
print(f'TensorRT {trt.__version__}; CUDA runtime {library}')
if int(trt.__version__.split('.', 1)[0]) < 10:
    raise SystemExit('TensorRT 10 or newer is required for the JetPack 6 path')
PY

if [ "${1:-}" = "--packages-only" ]; then
    exit 0
fi
"$script_dir/build_tensorrt_engine.sh" "$@"
