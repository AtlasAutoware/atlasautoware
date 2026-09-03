"""TensorRT detector tests with fake CUDA/TRT objects (no GPU required)."""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
import camera_perception as cp                              # noqa: E402


class FakeCuda:
    def __init__(self):
        self.next_pointer = 100
        self.allocations = {}
        self.destroyed = []

    def malloc(self, size):
        pointer = self.next_pointer
        self.next_pointer += 1
        self.allocations[pointer] = bytearray(size)
        return pointer

    def free(self, pointer):
        self.allocations.pop(pointer, None)

    @staticmethod
    def create_stream():
        return 77

    def destroy_stream(self, stream):
        self.destroyed.append(stream)

    def copy_to_device(self, pointer, array, stream):
        assert stream == 77
        self.allocations[pointer][:] = array.tobytes()

    def copy_from_device(self, array, pointer, stream):
        assert stream == 77
        raw = np.frombuffer(self.allocations[pointer], dtype=array.dtype)
        np.copyto(array.reshape(-1), raw)

    @staticmethod
    def synchronize(stream):
        assert stream == 77


class FakeContext:
    def __init__(self, cuda):
        self.cuda = cuda
        self.addresses = {}
        self.calls = []

    @staticmethod
    def get_tensor_shape(name):
        return (1, 3, 640, 640) if name == 'images' else (1, 5, 8400)

    def set_tensor_address(self, name, pointer):
        self.addresses[name] = pointer
        return True

    def execute_async_v3(self, stream):
        self.calls.append(('v3', stream))
        prediction = np.zeros((1, 5, 8400), np.float32)
        prediction[0, :, 0] = [320.0, 240.0, 100.0, 60.0, 0.9]
        output = self.cuda.allocations[self.addresses['output0']]
        output[:] = prediction.tobytes()
        return True


class FakeEngine:
    # Output deliberately comes first: TensorRT tensor order is not an I/O API.
    names = ('output0', 'images')
    num_io_tensors = 2

    def __init__(self, context, modes):
        self.context = context
        self.modes = modes

    def create_execution_context(self):
        return self.context

    def get_tensor_name(self, index):
        return self.names[index]

    def get_tensor_mode(self, name):
        return self.modes[name]

    @staticmethod
    def get_tensor_dtype(name):
        return np.float32


class FakeLogger:
    WARNING = 1

    def __init__(self, level):
        self.level = level


class FakeRuntime:
    def __init__(self, engine):
        self.engine = engine

    def deserialize_cuda_engine(self, payload):
        assert payload == b'fake-engine'
        return self.engine


def fake_trt(cuda):
    modes = SimpleNamespace(INPUT='input', OUTPUT='output')
    context = FakeContext(cuda)
    engine = FakeEngine(context, {'images': modes.INPUT,
                                  'output0': modes.OUTPUT})
    return SimpleNamespace(
        Logger=FakeLogger,
        Runtime=lambda logger: FakeRuntime(engine),
        TensorIOMode=modes,
        nptype=np.dtype,
        init_libnvinfer_plugins=lambda logger, namespace: True,
    ), context


def test_tensorrt_10_named_io_and_direct_inference(tmp_path):
    engine_path = tmp_path / 'car.engine'
    engine_path.write_bytes(b'fake-engine')
    cuda = FakeCuda()
    trt, context = fake_trt(cuda)

    detector = cp.TRTDetector(str(engine_path), img_size=416,
                              _trt=trt, _cuda=cuda)
    assert detector.backend == 'tensorrt'
    assert detector.sz == 640                 # static engine, not stale YAML
    assert set(context.addresses) == {'images', 'output0'}

    boxes = detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert context.calls == [('v3', 77)]
    assert len(boxes) == 1
    x, y, width, height, confidence = boxes[0]
    assert np.allclose([x, y, width, height], [270.0, 157.5, 100.0, 45.0])
    assert confidence > 0.89

    detector.close()
    assert cuda.allocations == {}
    assert cuda.destroyed == [77]


def test_fixed_shape_rejects_unresolved_dimensions():
    try:
        cp._fixed_shape((1, 3, -1, -1), 'images')
    except RuntimeError as exc:
        assert 'unresolved TensorRT images shape' in str(exc)
    else:
        raise AssertionError('dynamic shape should have been rejected')
