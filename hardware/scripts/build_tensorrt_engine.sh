#!/usr/bin/env bash
# Build the detector engine on the Jetson. TensorRT engines are not portable.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
onnx_path="${1:-$repo_root/models/car_yolov8_640.onnx}"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/atlasautoware"
engine_path="${2:-$cache_root/car_yolov8_640.engine}"

if [ "$(uname -m)" != "aarch64" ] || [ ! -r /etc/nv_tegra_release ]; then
    echo "error: build this engine on the Jetson, not on the development computer" >&2
    exit 2
fi
if [ ! -r "$onnx_path" ]; then
    echo "error: model not found: $onnx_path" >&2
    exit 2
fi

trtexec_bin="$(command -v trtexec || true)"
if [ -z "$trtexec_bin" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    trtexec_bin=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$trtexec_bin" ]; then
    echo "error: trtexec is missing; run hardware/scripts/install_tensorrt.sh" >&2
    exit 2
fi

mkdir -p "$(dirname -- "$engine_path")"
temporary_engine="$(mktemp "${engine_path}.tmp.XXXXXX")"
cleanup() {
    rm -f -- "$temporary_engine"
}
trap cleanup EXIT

echo "Building a target-specific FP16 TensorRT engine"
echo "  source: $onnx_path"
echo "  output: $engine_path"
python3 "$repo_root/tools/build_tensorrt_engine.py" \
    "$onnx_path" "$temporary_engine" --workspace-mib 1024

# Deserialize and execute it before replacing any known-good engine.
"$trtexec_bin" \
    --loadEngine="$temporary_engine" \
    --warmUp=1000 \
    --duration=5

mv -- "$temporary_engine" "$engine_path"
trap - EXIT

python3 "$repo_root/tools/benchmark_camera_perception.py" \
    "$engine_path" --warmup 20 --runs 100
echo "TensorRT detector ready: $engine_path"
