
import numpy as np, cv2, tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit

class ShrimpDetectorTRT:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._alloc()

    def _alloc(self):
        ins, outs, binds = [], [], []
        stream = cuda.Stream()
        for b in self.engine:
            size = trt.volume(self.engine.get_binding_shape(b))
            dtype = trt.nptype(self.engine.get_binding_dtype(b))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            binds.append(int(dev))
            (ins if self.engine.binding_is_input(b) else outs).append((host, dev))
        return ins, outs, binds, stream

    def detect(self, frame):
        img = cv2.resize(frame, (640,640))
        img = img.astype(np.float32)/255.0
        img = np.transpose(img,(2,0,1)).ravel()
        np.copyto(self.inputs[0][0], img)
        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
        self.stream.synchronize()
        out = self.outputs[0][0].reshape(-1,6)
        return [(int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2))
                for x,y,w,h,c,_ in out if c>0.4]
