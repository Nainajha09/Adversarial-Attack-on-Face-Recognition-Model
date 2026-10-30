# Adversarial attacks on face recognition: beyond fooling rates

Measures how 11 adversarial attacks (+3 non-adversarial controls) change face images at three levels:
pixel/perceptual (SSIM, PSNR, SNR, UQI, LPIPS), statistical (MSE, TV, PCC, NMI, FID, KID) and
identity/embedding (cosine drift, verification flips, rank-1 identification, top-5 consistency, ArcFace transfer),
plus where on the face each perturbation concentrates.

## Pipeline
| Step | File |
|---|---|
| Photo selection (seed 0; disjoint calibration / attack identities) | `select_subset.py`, `data/lfw_selection.txt`, `data/lfw_split.txt` |
| MTCNN detection, 160×160 crop (margin 14), 5 landmarks | `preprocess.py` |
| FaceNet (facenet-pytorch InceptionResnetV1, VGGFace2, 512-d); verification threshold calibrated on 150 held-out identities | `common.py`, `calibrate.py` |
| Attacks (embedding-space objective, no dummy labels) | `attacks.py`, `run_attack.py` |
| Per-image metrics | `evaluate.py` |
| FID / KID | `set_level.py` |
| Tables (mean ± 95% bootstrap CI), tests, figures | `analyze.py` |
| Everything | `run_all.sh`, `AdvFace_pipeline.ipynb` (Colab) |

## Key conventions (state these in the paper)
* Pixels in [0,1]; ε = 8/255 in L∞ (= 16/255 in [−1,1]). FaceNet normalisation applied once, inside the model wrapper.
* Threat model: dodging. Loss = cosine similarity between f(x_adv) and the clean embedding f(x).
  Success = the clean pair (photo A, other photo B of the same person) is accepted but (A_adv, B) is rejected.
* The clean image is a stationary point of this loss, so gradient attacks start from a 1–2/255 random perturbation.
* Cosine distance range is [0, 2]. SNR/PSNR are in dB. TV is measured on the perturbation.
* LFW = deep-funneled LFW from a public GitHub mirror (500 photos: 150 calibration + 100 attack identities; MTCNN failed on 1).
* LPIPS needs internet to download AlexNet weights; it was not available in the environment that produced
  `results/lfw`, so those tables use SSIM as the perceptual metric. Running the notebook adds an LPIPS column.

## Attacks
FGSM (R+FGSM), PGD-L∞, MIM, PGD-L2 (ε = median L2 of PGD-L∞ = 6.41), C&W-L2 (min-norm, target = threshold + 0.1),
Sparse-PGD (1% of pixels, L0), Square (300 queries), NES (300 queries), Universal perturbation (trained on calibration
identities only), Spatial (worst of rotation ±15° × shift ±6 px), Impersonation (targeted feature adversary).
Controls: random ±8/255 noise, Gaussian noise at matched L2, JPEG q=50.

## Run
```
bash download_weights.sh
bash get_lfw.sh && bash run_all.sh lfw lfw_raw
bash run_all.sh celeba <img_align_celeba_dir> <identity_CelebA.txt>   # optional
```
