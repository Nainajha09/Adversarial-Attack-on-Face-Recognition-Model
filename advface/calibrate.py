"""Step 2: clean embeddings + verification thresholds on the CALIBRATION identities
(disjoint from the attacked identities)."""
import json, sys, numpy as np, torch
from common import FaceNet, ArcFace, calibrate_threshold, cos_dist
ds = sys.argv[1]
d = np.load(f'data/{ds}.npz')
X = torch.from_numpy(d['images']).permute(0, 3, 1, 2).float() / 255.
fn, af = FaceNet(), ArcFace()
cal = d['split'] == 'calib'
res = {}
for name, emb in [('facenet', fn.embed(X[cal])), ('arcface', af.embed(X[cal], d['landmarks'][cal]))]:
    res[name] = calibrate_threshold(emb, d['identity'][cal]); print(name, res[name])
json.dump(res, open(f'data/{ds}_thresholds.json', 'w'), indent=1)
