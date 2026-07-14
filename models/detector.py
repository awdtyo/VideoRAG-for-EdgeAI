"""
Lightweight object detector using YOLOX-Nano exported to ONNX and run via
onnxruntime. Deliberately avoids the `ultralytics` package.

Model source: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0)
Download the ONNX export with scripts/download_models.sh before use.
"""
import cv2
import numpy as np
import onnxruntime as ort

import config


class YoloxNanoDetector:
    def __init__(self, onnx_path=config.DETECTOR_ONNX_PATH,
                 input_size=config.DETECTOR_INPUT_SIZE,
                 conf_thresh=config.DETECTOR_CONF_THRESH,
                 nms_thresh=config.DETECTOR_NMS_THRESH):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4  # Pi 5 has 4 cores
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        with open(config.COCO_CLASSES_PATH) as f:
            self.classes = [c.strip() for c in f if c.strip()]

    # ---- preprocessing ----
    def _preprocess(self, img):
        h, w = self.input_size
        img_h, img_w = img.shape[:2]
        r = min(h / img_h, w / img_w)
        resized = cv2.resize(img, (int(img_w * r), int(img_h * r)), interpolation=cv2.INTER_LINEAR)

        padded = np.ones((h, w, 3), dtype=np.uint8) * 114
        padded[: resized.shape[0], : resized.shape[1]] = resized

        padded = padded.astype(np.float32).transpose(2, 0, 1)[None, ...]  # BGR, NCHW, YOLOX expects raw 0-255
        return padded, r

    # ---- postprocessing (YOLOX-style decode + NMS) ----
    def _postprocess(self, outputs, ratio):
        preds = outputs[0][0]  # (num_boxes, 5 + num_classes)
        boxes = preds[:, :4]
        obj_conf = preds[:, 4]
        cls_scores = preds[:, 5:]

        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(len(cls_scores)), cls_id]
        scores = obj_conf * cls_conf

        mask = scores > self.conf_thresh
        boxes, scores, cls_id = boxes[mask], scores[mask], cls_id[mask]
        if len(boxes) == 0:
            return []

        # xywh -> xyxy
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy /= ratio

        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh
        )
        results = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                results.append({
                    "label": self.classes[cls_id[i]] if cls_id[i] < len(self.classes) else str(cls_id[i]),
                    "confidence": float(scores[i]),
                    "box": [float(v) for v in xyxy[i]],
                })
        return results

    def detect(self, frame_bgr):
        """Returns list of {label, confidence, box} for a single BGR frame."""
        inp, ratio = self._preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        return self._postprocess(outputs, ratio)
