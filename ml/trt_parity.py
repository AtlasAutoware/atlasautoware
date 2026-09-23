#!/usr/bin/env python3
"""trt_parity: does a TensorRT engine of the student give the same (speed, steer) as ONNX Runtime?

Feeds the same inputs (real-looking: a random camera frame, a rasterised random scan, zeros
for state, a real instruction) to both and reports the max / mean absolute difference.
Run on the Jetson:  python3 ml/trt_parity.py student.onnx student_fp16.engine [--n 200]
"""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'f1tenth_gym_ros'))
import policy_io as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('onnx'); ap.add_argument('engine'); ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    import onnxruntime as ort, tensorrt as trt
    sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(open(a.engine, 'rb').read()); ctx = eng.create_execution_context()
    import ctypes
    cudart = ctypes.CDLL('libcudart.so')
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    rng = np.random.default_rng(0); diffs = []
    instr = ['go straight to the end and stop', 'turn left, then go straight to the end and stop',
             'Take a right turn and stop', 'follow the track, bearing mostly left to the end and stop']
    for k in range(a.n):
        front = rng.integers(0, 255, (96, 128, 3), np.uint8)
        scan = rng.uniform(0.3, 8.0, 720).astype(np.float32)
        feed = P.make_feed(front, P.bev_image(scan, -math.pi, 2 * math.pi / 720), np.zeros(5, np.float32),
                           P.text_ids(instr[k % len(instr)]))
        ref = sess.run(['action'], feed)[0]
        bufs, host = {}, {}
        for n in names:
            shape = tuple(feed[n].shape) if n in feed else (1, 2)
            if n in feed: ctx.set_input_shape(n, shape)
            dt = trt.nptype(eng.get_tensor_dtype(n))
            host[n] = np.ascontiguousarray(feed[n].astype(dt)) if n in feed else np.zeros(shape, dt)
            ptr = ctypes.c_void_p(); cudart.cudaMalloc(ctypes.byref(ptr), host[n].nbytes); bufs[n] = ptr
            if n in feed: cudart.cudaMemcpy(ptr, host[n].ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(host[n].nbytes), 1)
            ctx.set_tensor_address(n, ptr.value)
        ctx.execute_async_v3(0); cudart.cudaDeviceSynchronize()
        out = [n for n in names if n not in feed][0]
        cudart.cudaMemcpy(host[out].ctypes.data_as(ctypes.c_void_p), bufs[out], ctypes.c_size_t(host[out].nbytes), 2)
        for p in bufs.values(): cudart.cudaFree(p)
        diffs.append(np.abs(host[out].astype(np.float32) - ref))
    d = np.concatenate(diffs)
    print(f'{os.path.basename(a.engine)} vs ONNX Runtime over {a.n} inputs: '
          f'speed |diff| max {d[:, 0].max():.4f} mean {d[:, 0].mean():.5f} m/s; '
          f'steer |diff| max {d[:, 1].max():.4f} mean {d[:, 1].mean():.5f} rad')


if __name__ == '__main__':
    main()
