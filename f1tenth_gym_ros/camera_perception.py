"""
Camera perception — detect other cars with a trained YOLO model and feed the
same race brain the lidar does.
=============================================================================

This runs on the **real car** (the f1tenth_gym sim has no camera). The hardware
configuration requires a device-built TensorRT engine. Frames go through the
TensorRT 10 named-I/O API and CUDA directly; PyTorch, PyCUDA, and ONNX Runtime
are absent from the live path. Build the engine once on the Jetson with
``hardware/scripts/build_tensorrt_engine.sh``.

The `auto`, `onnxruntime`, `cuda`, and `cpu` backends remain available for
development computers. Hardware uses `backend: tensorrt` so a broken GPU setup
is visible at startup instead of silently taking a CPU core during a race.

(Training happens separately — see `tools/train_car_detector.py`.  The
control loop stays on CPU on purpose: its QP is far too small to benefit
from a GPU; the camera pipeline is where the Jetson's GPU earns its keep.)

Pipeline:  image  ->  YOLO car boxes  ->  back-project each box to a position
relative to the car (pinhole + known car width)  ->  `race_brain.Opponent`
objects, in the **same format the lidar detector emits**, so they drop straight
into `RaceStrategist` or fuse with the lidar tracks.

The output is sensor-agnostic on purpose: camera gives "what + bearing", lidar
gives precise range; fusing them (camera class/bearing + lidar range) is the
robust setup, and both already speak the `Opponent` type.

Standalone (no ROS) test of the detector + geometry:
    from camera_perception import CarDetector, box_to_relative
"""

import ctypes
import ctypes.util
import math
import os

import numpy as np

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# Detectors — TensorRT (Jetson GPU) or YOLOv8 ONNX via cv2.dnn (CUDA/CPU)
# ─────────────────────────────────────────────────────────────────────────────

def parse_yolo_output(out, w0, h0, img_size, conf, nms, car_class):
    """Raw YOLOv8 head (1, 4+nc, N) -> [(x, y, w, h, confidence), ...] px."""
    out = np.squeeze(np.asarray(out))                   # (4+nc, N)
    if out.ndim != 2:
        return []
    if out.shape[0] < out.shape[1]:                     # (4+nc, N) -> (N, 4+nc)
        out = out.T
    cls = out[:, 4:]
    cids = np.argmax(cls, axis=1)
    confs = cls[np.arange(len(cls)), cids]
    keep = (confs > conf) & (cids == car_class)
    if not keep.any():
        return []
    rows, confs = out[keep], confs[keep]
    sx, sy = w0 / img_size, h0 / img_size
    cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.stack([(cx - bw/2)*sx, (cy - bh/2)*sy, bw*sx, bh*sy], 1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), conf, nms)
    if len(idx) == 0:
        return []
    idx = np.array(idx).flatten()
    return [(*boxes[i], float(confs[i])) for i in idx]


class CarDetector:
    """ONNX via cv2.dnn — CUDA target on the Jetson GPU, or CPU fallback."""

    def __init__(self, model_path, img_size=640, conf=0.35, nms=0.45,
                 car_class=0, use_cuda=False):
        if cv2 is None:
            raise RuntimeError('OpenCV not available')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'model not found: {model_path} — train one with '
                f'tools/train_car_detector.py and export to ONNX')
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.backend = 'cpu'
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.backend = 'cuda'
        self.sz, self.conf, self.nms, self.car_class = img_size, conf, nms, car_class

    def detect(self, img):
        """BGR image -> list of (x, y, w, h, confidence) car boxes in image px."""
        h0, w0 = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.sz, self.sz),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()                       # (1, 4+nc, N)
        return parse_yolo_output(out, w0, h0, self.sz, self.conf, self.nms,
                                 self.car_class)


class OrtDetector:
    """Portable YOLOv8 ONNX Runtime CPU backend for development and regression.

    The Jetson's Ubuntu OpenCV 4.5.4 cannot execute this YOLOv8 head. ONNX
    Runtime was the previous on-car workaround (~130 ms at 640, ~60 ms at 416
    on four Orin Nano CPU threads), but hardware.yaml now requires TensorRT.
    """

    backend = 'onnxruntime'

    def __init__(self, model_path, img_size=640, conf=0.35, nms=0.45, car_class=0, threads=4):
        import onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model not found: {model_path}')
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape           # e.g. [1, 3, 416, 416]
        if isinstance(shp[-1], int) and shp[-1] != img_size:
            img_size = shp[-1]                         # model is fixed-size: follow it
        self.sz, self.conf, self.nms, self.car_class = img_size, conf, nms, car_class

    def detect(self, img):
        h0, w0 = img.shape[:2]
        x = cv2.resize(img, (self.sz, self.sz))[:, :, ::-1]           # BGR -> RGB
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0]                     # (1, 4+nc, N)
        return parse_yolo_output(out, w0, h0, self.sz, self.conf, self.nms, self.car_class)


class _CudaRuntime:
    """Small CUDA Runtime wrapper so TensorRT needs no PyTorch or PyCUDA."""

    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self):
        candidates = [ctypes.util.find_library('cudart'), 'libcudart.so',
                      'libcudart.so.12']
        error = None
        for candidate in dict.fromkeys(c for c in candidates if c):
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                error = exc
        else:
            raise RuntimeError(f'CUDA runtime library not found ({error})')

        void_p = ctypes.c_void_p
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            void_p, void_p, ctypes.c_size_t, ctypes.c_int, void_p]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

    def _check(self, code, operation):
        if code == 0:
            return
        detail = self.lib.cudaGetErrorString(code)
        message = detail.decode() if detail else f'error {code}'
        raise RuntimeError(f'{operation} failed: {message}')

    def malloc(self, size):
        pointer = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(pointer), size),
                    'cudaMalloc')
        return int(pointer.value)

    def free(self, pointer):
        if pointer:
            self._check(
                self.lib.cudaFree(ctypes.c_void_p(pointer)), 'cudaFree')

    def create_stream(self):
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)),
                    'cudaStreamCreate')
        return int(stream.value)

    def destroy_stream(self, stream):
        if stream:
            self._check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)),
                        'cudaStreamDestroy')

    def copy_to_device(self, pointer, array, stream):
        self._check(self.lib.cudaMemcpyAsync(
            ctypes.c_void_p(pointer), ctypes.c_void_p(array.ctypes.data),
            array.nbytes, self.HOST_TO_DEVICE, ctypes.c_void_p(stream)),
            'host-to-device copy')

    def copy_from_device(self, array, pointer, stream):
        self._check(self.lib.cudaMemcpyAsync(
            ctypes.c_void_p(array.ctypes.data), ctypes.c_void_p(pointer),
            array.nbytes, self.DEVICE_TO_HOST, ctypes.c_void_p(stream)),
            'device-to-host copy')

    def synchronize(self, stream):
        self._check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)),
                    'cudaStreamSynchronize')


def _fixed_shape(shape, label):
    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise RuntimeError(f'unresolved TensorRT {label} shape: {shape}')
    return shape


class TRTDetector:
    """Run a device-built YOLOv8 engine directly with TensorRT and CUDA.

    TensorRT engines are not portable across JetPack/GPU versions. Build this
    one on the car with ``hardware/scripts/build_tensorrt_engine.sh``. Runtime
    inference reads only the serialized engine; ONNX Runtime is not involved.
    """

    backend = 'tensorrt'

    def __init__(self, engine_path, img_size=640, conf=0.35, nms=0.45,
                 car_class=0, _trt=None, _cuda=None):
        if cv2 is None:
            raise RuntimeError('OpenCV needed for TensorRT preprocessing')
        engine_path = os.path.abspath(os.path.expanduser(engine_path))
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f'TensorRT engine not found: {engine_path}')

        if _trt is None:
            import tensorrt as trt
        else:                                      # test seam; no GPU needed
            trt = _trt
        self._cuda = _cuda or _CudaRuntime()
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        if hasattr(trt, 'init_libnvinfer_plugins'):
            trt.init_libnvinfer_plugins(self._logger, '')
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, 'rb') as stream:
            self.engine = self._runtime.deserialize_cuda_engine(stream.read())
        if self.engine is None:
            raise RuntimeError(
                f'could not deserialize TensorRT engine: {engine_path}')
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError('could not create TensorRT execution context')

        self.stream = self._cuda.create_stream()
        self._device_allocations = []
        self._bindings = None
        self._api = 'v3' if hasattr(self.engine, 'num_io_tensors') else 'v2'
        try:
            self._input, self._output = self._allocate_io(img_size)
            in_shape = self._input['shape']
            if len(in_shape) != 4 or in_shape[0] != 1 or in_shape[1] != 3 \
                    or in_shape[2] != in_shape[3]:
                raise RuntimeError(
                    f'expected NCHW square YOLO input, got {in_shape}')
        except Exception:
            self.close()
            raise
        self.sz = in_shape[-1]
        self.conf, self.nms, self.car_class = conf, nms, car_class

    def _buffer(self, name, shape, dtype):
        shape = _fixed_shape(shape, name)
        host = np.empty(shape, dtype=np.dtype(dtype))
        device = self._cuda.malloc(host.nbytes)
        self._device_allocations.append(device)
        return {'name': name, 'shape': shape, 'host': host, 'device': device}

    def _allocate_io(self, img_size):
        """Allocate one-input/one-output YOLO engines for TRT 10 or TRT 8."""
        trt = self._trt
        if self._api == 'v3':
            names = [self.engine.get_tensor_name(i)
                     for i in range(self.engine.num_io_tensors)]
            inputs = [
                name for name in names
                if self.engine.get_tensor_mode(name)
                == trt.TensorIOMode.INPUT]
            outputs = [
                name for name in names
                if self.engine.get_tensor_mode(name)
                == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(
                    'YOLO engine must have exactly one input and one output; '
                    f'got {inputs=} {outputs=}')
            input_name, output_name = inputs[0], outputs[0]
            input_shape = tuple(self.ctx.get_tensor_shape(input_name))
            if any(int(value) < 0 for value in input_shape):
                requested = (1, 3, int(img_size), int(img_size))
                if not self.ctx.set_input_shape(input_name, requested):
                    raise RuntimeError(
                        f'cannot set TensorRT input to {requested}')
                input_shape = tuple(self.ctx.get_tensor_shape(input_name))
            output_shape = tuple(self.ctx.get_tensor_shape(output_name))
            input_buffer = self._buffer(
                input_name, input_shape,
                trt.nptype(self.engine.get_tensor_dtype(input_name)))
            output_buffer = self._buffer(
                output_name, output_shape,
                trt.nptype(self.engine.get_tensor_dtype(output_name)))
            for buffer in (input_buffer, output_buffer):
                if not self.ctx.set_tensor_address(
                        buffer['name'], buffer['device']):
                    raise RuntimeError('could not bind TensorRT tensor '
                                       f'{buffer["name"]}')
            return input_buffer, output_buffer

        count = self.engine.num_bindings
        input_indices = [i for i in range(count)
                         if self.engine.binding_is_input(i)]
        output_indices = [i for i in range(count)
                          if not self.engine.binding_is_input(i)]
        if len(input_indices) != 1 or len(output_indices) != 1:
            raise RuntimeError(
                'YOLO engine must have exactly one input and one output; '
                f'got {input_indices=} {output_indices=}')
        input_index, output_index = input_indices[0], output_indices[0]
        input_shape = tuple(self.ctx.get_binding_shape(input_index))
        if any(int(value) < 0 for value in input_shape):
            requested = (1, 3, int(img_size), int(img_size))
            if not self.ctx.set_binding_shape(input_index, requested):
                raise RuntimeError(f'cannot set TensorRT input to {requested}')
            input_shape = tuple(self.ctx.get_binding_shape(input_index))
        input_buffer = self._buffer(
            str(input_index), input_shape,
            trt.nptype(self.engine.get_binding_dtype(input_index)))
        output_buffer = self._buffer(
            str(output_index), self.ctx.get_binding_shape(output_index),
            trt.nptype(self.engine.get_binding_dtype(output_index)))
        self._bindings = [0] * count
        self._bindings[input_index] = input_buffer['device']
        self._bindings[output_index] = output_buffer['device']
        return input_buffer, output_buffer

    def detect(self, img):
        """Return car boxes after a direct TensorRT GPU inference."""
        h0, w0 = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.sz, self.sz),
                                     swapRB=True, crop=False)
        np.copyto(self._input['host'], blob,
                  casting='unsafe' if blob.dtype != self._input['host'].dtype
                  else 'no')
        self._cuda.copy_to_device(self._input['device'],
                                  self._input['host'], self.stream)
        if self._api == 'v3':
            ok = self.ctx.execute_async_v3(self.stream)
        else:
            ok = self.ctx.execute_async_v2(self._bindings, self.stream)
        if not ok:
            raise RuntimeError('TensorRT inference enqueue failed')
        self._cuda.copy_from_device(self._output['host'],
                                    self._output['device'], self.stream)
        self._cuda.synchronize(self.stream)
        return parse_yolo_output(self._output['host'], w0, h0, self.sz,
                                 self.conf, self.nms, self.car_class)

    def close(self):
        cuda = getattr(self, '_cuda', None)
        if cuda is None:
            return
        for pointer in reversed(getattr(self, '_device_allocations', [])):
            try:
                cuda.free(pointer)
            except Exception:
                pass
        self._device_allocations = []
        stream = getattr(self, 'stream', None)
        if stream:
            try:
                cuda.destroy_stream(stream)
            except Exception:
                pass
            self.stream = None

    def __del__(self):                         # pragma: no cover - shutdown
        self.close()


def make_detector(model_path, backend='auto', **kw):
    """Build a detector, optionally selecting the fastest available backend."""
    model_path = os.path.abspath(os.path.expanduser(model_path))
    engine = model_path if model_path.endswith(('.engine', '.trt')) \
        else os.path.splitext(model_path)[0] + '.engine'
    if backend in ('auto', 'tensorrt') and os.path.exists(engine):
        try:
            return TRTDetector(engine, **kw)
        except Exception:
            if backend == 'tensorrt':
                raise
    if backend == 'tensorrt':
        raise FileNotFoundError(f'no TensorRT engine at {engine}')
    if backend in ('auto', 'onnxruntime'):
        try:
            return OrtDetector(model_path, **kw)
        except Exception as e:
            if backend == 'onnxruntime':
                raise
            print('[camera_perception] onnxruntime unavailable '
                  f'({e}); trying cv2.dnn')
    has_cuda = (cv2 is not None
                and cv2.cuda.getCudaEnabledDeviceCount() > 0) \
        if backend == 'auto' else (backend == 'cuda')
    return CarDetector(model_path, use_cuda=has_cuda, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — image box -> position relative to the car
# ─────────────────────────────────────────────────────────────────────────────

def box_to_relative(box, fx, cx_img, car_width=0.30):
    """
    Pinhole back-projection with a known object width.
    box = (x, y, w, h) px.  Returns (x_forward, y_left, range) in metres, in the
    camera/base frame (ROS: +X forward, +Y left).
    """
    x, y, w, h = box[:4]
    u_center = x + w / 2.0
    depth = car_width * fx / max(w, 1.0)               # along camera optical axis
    y_left = -(u_center - cx_img) * depth / fx         # right of centre -> -Y
    rng = math.hypot(depth, y_left)
    return depth, y_left, rng


def relative_to_world(x_fwd, y_left, ego):
    ex, ey, eyaw = ego
    wx = ex + x_fwd * math.cos(eyaw) - y_left * math.sin(eyaw)
    wy = ey + x_fwd * math.sin(eyaw) + y_left * math.cos(eyaw)
    return wx, wy


# ─────────────────────────────────────────────────────────────────────────────
# ROS node (only imported/used on the car)
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import PoseArray, Pose
    from std_msgs.msg import ColorRGBA
    from transforms3d.euler import quat2euler

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from race_brain import Opponent, OpponentDetector  # noqa: reuse the tracker

    class CameraPerception(Node):
        def __init__(self):
            super().__init__('camera_perception')
            self.declare_parameter('image_topic', '/camera/color/image_raw')
            self.declare_parameter('model_path', os.path.join(
                os.path.dirname(__file__), '..', 'models', 'car_yolov8.onnx'))
            self.declare_parameter('fx', 600.0)        # camera intrinsics
            self.declare_parameter('cx', 320.0)
            self.declare_parameter('car_width', 0.30)
            self.declare_parameter('conf', 0.35)
            # plausibility guards: a real opponent is never closer than min_range (the
            # lidar owns that zone anyway) and never fills most of the frame. A hand or
            # face passing the lens otherwise becomes an 'opponent at 0.2 m'.
            self.declare_parameter('min_range', 0.5)      # m
            self.declare_parameter('max_box_frac', 0.6)   # box width / image width
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            self.declare_parameter('backend', 'auto')  # auto|tensorrt|onnxruntime|cuda|cpu
            self.declare_parameter('img_size', 640)    # model input side (416 = faster on CPU)
            topic = self.get_parameter('image_topic').value
            self.fx = float(self.get_parameter('fx').value)
            self.cx = float(self.get_parameter('cx').value)
            self.car_w = float(self.get_parameter('car_width').value)
            self.min_range = float(self.get_parameter('min_range').value)
            self.max_box_frac = float(self.get_parameter('max_box_frac').value)

            try:
                self.detector = make_detector(
                    self.get_parameter('model_path').value,
                    backend=self.get_parameter('backend').value,
                    img_size=int(self.get_parameter('img_size').value),
                    conf=float(self.get_parameter('conf').value))
                self.get_logger().info(
                    f'YOLO car detector loaded (backend: {self.detector.backend})')
            except Exception as e:
                self.detector = None
                self.get_logger().error(f'No detector ({e}); node idle until a model exists')

            # Reuse the lidar tracker purely for smoothing camera detections.
            self.tracker = OpponentDetector()
            self.ego = (0.0, 0.0, 0.0)
            odom_topic = self.get_parameter('odom_topic').value
            self.create_subscription(Odometry, odom_topic, self._odom, 10)
            self.create_subscription(Image, topic, self._image, 5)
            self.opp_pub = self.create_publisher(MarkerArray, '/camera_opponents', 5)
            self.pose_pub = self.create_publisher(PoseArray, '/camera_opponents_poses', 5)
            self.Marker, self.MarkerArray = Marker, MarkerArray
            self.PoseArray, self.Pose = PoseArray, Pose
            self.ColorRGBA = ColorRGBA
            self.get_logger().info(f'camera_perception subscribed to {topic}')

        def _odom(self, m):
            q = m.pose.pose.orientation
            from transforms3d.euler import quat2euler as q2e
            _, _, yaw = q2e([q.w, q.x, q.y, q.z])
            self.ego = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

        def _image(self, msg):
            if self.detector is None:
                return
            img = self._decode(msg)
            if img is None:
                return
            opps = []
            img_w = img.shape[1]
            for box in self.detector.detect(img):
                if box[2] > self.max_box_frac * img_w:
                    continue                                   # fills the frame: not a car at range
                xf, yl, rng = box_to_relative(box, self.fx, self.cx, self.car_w)
                if rng < self.min_range:
                    continue
                wx, wy = relative_to_world(xf, yl, self.ego)
                o = Opponent(wx, wy, rng, self.car_w, math.atan2(yl, xf))
                opps.append(o)
            self._publish(opps)

        @staticmethod
        def _decode(msg):
            try:
                from cv_bridge import CvBridge
                return CvBridge().imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                if msg.encoding in ('rgb8', 'bgr8'):
                    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
                    return a[:, :, ::-1] if msg.encoding == 'rgb8' else a
                return None

        def _publish(self, opps):
            pa = self.PoseArray()
            pa.header.frame_id = 'map'
            for o in opps:
                p = self.Pose()
                p.position.x, p.position.y = float(o.x), float(o.y)
                p.orientation.w = 1.0
                pa.poses.append(p)
            self.pose_pub.publish(pa)

            arr = self.MarkerArray()
            for k, o in enumerate(opps):
                m = self.Marker()
                m.header.frame_id = 'map'; m.ns = 'camera_opp'; m.id = k
                m.type = self.Marker.CUBE; m.action = self.Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = o.x, o.y, 0.1
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 0.4; m.scale.z = 0.2
                m.color = self.ColorRGBA(r=0.1, g=0.6, b=0.9, a=0.9)
                arr.markers.append(m)
            self.opp_pub.publish(arr)

    return CameraPerception


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = _make_node()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
