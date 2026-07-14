"""
Frame-level visual embeddings using MobileNetV3-Small (ONNX), pooled features
from the penultimate layer. Used only if config.ENABLE_VISUAL_EMBEDDINGS is
True -- lets you do "find frames that look like this" in addition to text RAG.
"""
import cv2
import numpy as np
import onnxruntime as ort

import config

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MobileNetEmbedder:
    def __init__(self, onnx_path=config.EMBEDDER_ONNX_PATH,
                 input_size=config.EMBEDDER_INPUT_SIZE):
        self.input_size = input_size
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)[None, ...]
        return img.astype(np.float32)

    def embed(self, frame_bgr):
        """Returns an L2-normalized 1D feature vector for the frame."""
        inp = self._preprocess(frame_bgr)
        out = self.session.run(None, {self.input_name: inp})[0]
        vec = out.reshape(-1)
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).tolist()
