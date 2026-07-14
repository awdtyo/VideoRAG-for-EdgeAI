#!/usr/bin/env bash
# Downloads the ONNX weights this pipeline needs. Run once before first use.
# YOLOX-Nano: official Megvii release (Apache-2.0), no ultralytics involved.
# MobileNetV3-Small: exported from torchvision, hosted in the ONNX Model Zoo mirror.
set -e

WEIGHTS_DIR="$(dirname "$0")/weights"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading YOLOX-Nano ONNX..."
curl -L -o "$WEIGHTS_DIR/yolox_nano.onnx" \
  "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"

echo "Downloading MobileNetV3-Small ONNX..."
curl -L -o "$WEIGHTS_DIR/mobilenetv3_small.onnx" \
  "https://huggingface.co/onnx-community/mobilenetv3_small_100.lamb_in1k/resolve/main/onnx/model.onnx"

echo "Done. Weights are in $WEIGHTS_DIR"
echo "NOTE: URLs for pretrained weights can move. If either download fails:"
echo "  - YOLOX-Nano: https://github.com/Megvii-BaseDetection/YOLOX/releases  (grab the 0.1.1rc0 yolox_nano.onnx asset)"
echo "  - MobileNetV3-Small: https://huggingface.co/onnx-community/mobilenetv3_small_100.lamb_in1k/tree/main/onnx"
echo "Place whichever file you get at the path shown above, matching the names in config.py."
echo "If you'd rather skip visual embeddings entirely, set ENABLE_VISUAL_EMBEDDINGS = False in config.py"
echo "and you won't need mobilenetv3_small.onnx at all."
