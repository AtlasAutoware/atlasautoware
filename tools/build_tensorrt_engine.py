#!/usr/bin/env python3
"""Compile an ONNX training export into a target-local TensorRT engine."""

import argparse
import os
import tempfile


def build_engine(onnx_path, engine_path, workspace_mib=1024):
    """Build a target-local engine and atomically install it."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    major = int(trt.__version__.split('.', 1)[0])
    flags = 0
    if major >= 11:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        errors = '\n'.join(str(parser.get_error(index))
                           for index in range(parser.num_errors))
        raise RuntimeError(f'TensorRT could not parse {onnx_path}:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_mib) << 20)
    if major < 11:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError(
                'this GPU does not expose fast FP16 to TensorRT')
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print('TensorRT 11 uses strongly typed networks; precision follows '
              'the model. JetPack 6 uses the FP16 TensorRT 10 branch.')

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    engine_bytes = bytes(serialized)

    engine_path = os.path.abspath(os.path.expanduser(engine_path))
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(engine_path) + '.',
        suffix='.tmp', dir=os.path.dirname(engine_path))
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(engine_bytes)
        os.replace(temporary, engine_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    inputs = [network.get_input(index)
              for index in range(network.num_inputs)]
    outputs = [network.get_output(index)
               for index in range(network.num_outputs)]
    print(f'TensorRT {trt.__version__}: wrote '
          f'{len(engine_bytes) / (1 << 20):.1f} MiB '
          f'engine to {engine_path}')
    for tensor in inputs:
        print(f'  input  {tensor.name}: {tuple(tensor.shape)} {tensor.dtype}')
    for tensor in outputs:
        print(f'  output {tensor.name}: {tuple(tensor.shape)} {tensor.dtype}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', help='fixed-shape YOLO ONNX export')
    parser.add_argument('engine', help='target-local output .engine')
    parser.add_argument('--workspace-mib', type=int, default=1024)
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, args.workspace_mib)


if __name__ == '__main__':
    main()
