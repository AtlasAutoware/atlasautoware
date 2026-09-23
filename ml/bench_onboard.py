#!/usr/bin/env python3
"""bench_onboard: latency of the student policy on the car's own compute.

Times the exact call policy_bridge makes (policy_io.make_feed + ONNX Runtime run) on
realistic inputs, for several thread counts, plus the full per-tick preprocessing
(front resize + lidar raster). Prints JSON.

    python3 ml/bench_onboard.py models/student.onnx [--n 500]
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'f1tenth_gym_ros'))
import policy_io as P  # noqa: E402


def pct(a, q): return round(float(np.percentile(a, q)), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--n', type=int, default=500)
    a = ap.parse_args()
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (360, 640, 3), np.uint8)             # OAK-D Pro preview size
    scan = rng.uniform(0.2, 12.0, 720).astype(np.float32)               # /scan: 720 bins
    ids = P.text_ids('turn left, then go straight to the end and stop')
    out = {'model': os.path.basename(a.model), 'n': a.n}
    t = []
    for _ in range(a.n):
        t0 = time.perf_counter()
        f = P.front_image(frame); b = P.bev_image(scan, -math.pi, 2 * math.pi / 720)
        t.append((time.perf_counter() - t0) * 1e3)
    out['preprocess_ms'] = {'p50': pct(t, 50), 'p99': pct(t, 99)}
    feed = P.make_feed(f, b, np.zeros(5, np.float32), ids)
    for th in (1, 2, 4):
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
        for _ in range(20): s.run(['action'], feed)
        t = []
        for _ in range(a.n):
            t0 = time.perf_counter(); s.run(['action'], feed); t.append((time.perf_counter() - t0) * 1e3)
        out[f'ort_cpu_{th}thr_ms'] = {'mean': round(float(np.mean(t)), 3), 'p50': pct(t, 50), 'p99': pct(t, 99)}
    print(json.dumps(out))


if __name__ == '__main__':
    main()
