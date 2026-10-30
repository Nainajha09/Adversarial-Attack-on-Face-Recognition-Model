"""Step 5: distribution-level realism. FID and KID between the clean probe photos and their
adversarial versions, using the standard FID InceptionV3 (pytorch-fid, pool3 2048-d features).
With ~100 images FID is biased upward and should be read comparatively; KID (unbiased) is
the more reliable number at this sample size."""
import sys, glob, numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import linalg
from pytorch_fid.inception import InceptionV3

ds = sys.argv[1]
net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
@torch.no_grad()
def feats(x):
    return torch.cat([net(x[i:i + 25])[0].flatten(1) for i in range(0, len(x), 25)]).numpy().astype(np.float64)

def fid(a, b):
    m1, m2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))

def kid(a, b, n_sub=50, size=50, seed=0):
    rng = np.random.default_rng(seed); dd = a.shape[1]; vals = []
    k = lambda x, y: (x @ y.T / dd + 1) ** 3
    for _ in range(n_sub):
        x = a[rng.choice(len(a), size, False)]; y = b[rng.choice(len(b), size, False)]
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        m = size
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))

d = np.load(f'data/{ds}.npz')
X = torch.from_numpy(d['images']).permute(0, 3, 1, 2).float() / 255.
rows = []
first = True
for f in sorted(glob.glob(f'data/{ds}_adv_*.pt')):
    o = torch.load(f)
    if first:
        fc = feats(X[o['iA']]); first = False
    fa = feats(o['adv'])
    km, ks = kid(fc, fa)
    rows.append(dict(attack=f.split('_adv_')[1][:-3], dataset=ds, FID=fid(fc, fa), KIDx1000=km * 1000, KIDx1000_sd=ks * 1000))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'results/{ds}_set_level.csv', index=False)
