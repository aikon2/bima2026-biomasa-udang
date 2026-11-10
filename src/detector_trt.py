"""TensorRT YOLOv8 detector. The only module importing tensorrt / pycuda (C2).

Pre/post-processing is delegated to the pure-numpy modules `preprocess` and
`postprocess` so it can be unit-tested without a GPU. Uses the TensorRT 8.2
binding API shipped with JetPack 4.6.
"""
import logging
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from .postprocess import Detection, postprocess_detections
from .preprocess import letterbox

LOGGER = logging.getLogger(__name__)

# Spatial dims of an NCHW input binding.
INPUT_HEIGHT_AXIS = 2
INPUT_WIDTH_AXIS = 3

Binding = namedtuple("Binding", "name host device shape")


class ShrimpDetectorTRT(object):
    """Run a serialized YOLOv8 TensorRT engine on BGR frames."""

    def __init__(self, engine_path: str, conf_threshold: float, iou_threshold: float,
                 num_classes: int) -> None:
        # Creating the CUDA context lazily keeps importing this module side-effect free.
        import pycuda.autoinit  # noqa: F401

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.num_classes = int(num_classes)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_binding, self.output_binding, self.bindings = self._allocate()
        self.input_hw = self._input_hw(self.input_binding.shape)
        LOGGER.info("Loaded engine %s (input %s, output %s)", engine_path,
                    self.input_binding.shape, self.output_binding.shape)

    @staticmethod
    def _load_engine(engine_path: str):
        trt_logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as engine_file, trt.Runtime(trt_logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_file.read())
        except (IOError, OSError) as error:
            raise RuntimeError("Cannot read TensorRT engine {}: {}".format(engine_path, error))
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine {}".format(engine_path))
        return engine

    @staticmethod
    def _input_hw(input_shape: Tuple[int, ...]) -> Tuple[int, int]:
        if len(input_shape) != 4 or min(input_shape) <= 0:
            raise RuntimeError("Unsupported (dynamic or non-NCHW) input shape {}".format(
                input_shape))
        return int(input_shape[INPUT_HEIGHT_AXIS]), int(input_shape[INPUT_WIDTH_AXIS])

    def _allocate(self) -> Tuple[Binding, Binding, List[int]]:
        inputs, outputs, device_pointers = [], [], []
        for binding_name in self.engine:
            shape = tuple(self.engine.get_binding_shape(binding_name))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_name))
            host = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device = cuda.mem_alloc(host.nbytes)
            device_pointers.append(int(device))
            binding = Binding(binding_name, host, device, shape)
            if self.engine.binding_is_input(binding_name):
                inputs.append(binding)
            else:
                outputs.append(binding)
        if len(inputs) != 1 or not outputs:
            raise RuntimeError("Expected 1 input and >=1 output binding, got {} / {}".format(
                len(inputs), len(outputs)))
        if len(outputs) > 1:
            LOGGER.warning("Engine has %d outputs; using '%s'", len(outputs), outputs[0].name)
        return inputs[0], outputs[0], device_pointers

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        np.copyto(self.input_binding.host, blob.ravel())
        cuda.memcpy_htod_async(self.input_binding.device, self.input_binding.host, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_binding.host, self.output_binding.device, self.stream)
        self.stream.synchronize()
        return self.output_binding.host.reshape(self.output_binding.shape)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect shrimp; returns boxes in original frame pixel coordinates."""
        blob, scale, pad_xy = letterbox(frame_bgr, self.input_hw)
        output = self._infer(blob)
        return postprocess_detections(output, self.num_classes, self.conf_threshold,
                                      self.iou_threshold, scale, pad_xy, frame_bgr.shape)
