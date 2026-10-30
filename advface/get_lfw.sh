#!/bin/bash
# Downloads exactly the 500 LFW (deep-funneled) photos used in the paper from a public GitHub mirror.
[ -d lfw_raw ] || git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/redwankarimsony/deepfunneled_lfw lfw_raw
cd lfw_raw && git sparse-checkout init --no-cone && git sparse-checkout set --no-cone $(cat ../data/lfw_selection.txt | tr '\n' ' ') && git checkout -q HEAD
echo "LFW images: $(find . -name '*.jpg' | wc -l)"
