#!/bin/bash
# Model weights (all from GitHub releases)
mkdir -p weights && cd weights
[ -f vggface2.pt ] || curl -sL -o vggface2.pt https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
if [ ! -f w600k_mbf.onnx ]; then
  curl -sL -o buffalo_s.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip
  python -c "import zipfile;zipfile.ZipFile('buffalo_s.zip').extract('w600k_mbf.onnx','.')" && rm buffalo_s.zip
fi
ls -la
