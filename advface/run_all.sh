#!/bin/bash
# Full pipeline for one dataset. usage: bash run_all.sh lfw <image_root> | bash run_all.sh celeba <img_dir> <identity_CelebA.txt>
set -e
DS=$1; ROOT=$2; IDF=$3
[ -f data/${DS}_selection.txt ] || python select_subset.py $DS $ROOT $IDF
python preprocess.py --root $ROOT --list data/${DS}_selection.txt --split data/${DS}_split.txt --out data/${DS}.npz ${IDF:+--identity $IDF}
python calibrate.py $DS
for a in fgsm pgd_linf pgd_l2 gauss_matched mim cw_l2 sparse square nes uap spatial imperson rand_sign jpeg50; do
  [ -f data/${DS}_adv_$a.pt ] || python run_attack.py $DS $a
done
python evaluate.py $DS
python set_level.py $DS
python analyze.py $DS
