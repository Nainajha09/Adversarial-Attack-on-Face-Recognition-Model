"""Step 4: evaluate every saved attack. Writes one row per attacked image to results/<ds>_per_image.csv
and set-level distribution metrics (FID, KID) to results/<ds>_set_level.csv.

Identity metrics (all use photos the attack never saw):
  self_dist      cosine distance between FaceNet(x_adv) and FaceNet(x)              (drift)
  ref_dist       cosine distance between FaceNet(x_adv) and FaceNet(other photo B)  (what a system sees)
  dodge          1 if the clean pair (A,B) was accepted but the adversarial pair is rejected
  rank1_clean / rank1_adv   correct identity at rank 1 in a gallery of 249 identities
  top5_consist   |top5(x) & top5(x_adv)| / 5 using FaceNet's real 8,631-identity VGGFace2 head
  arc_*          same, on ArcFace (transfer: the attack never saw this model)
"""
import json, sys, glob, os, numpy as np, pandas as pd, torch
from common import FaceNet, ArcFace, cos_dist, image_metrics, region_concentration

ds = sys.argv[1]
only = sys.argv[2:]  # optional subset of attacks
os.makedirs('results', exist_ok=True)
d = np.load(f'data/{ds}.npz')
thr_all = json.load(open(f'data/{ds}_thresholds.json'))
T, TA = thr_all['facenet']['threshold'], thr_all['arcface']['threshold']
X = torch.from_numpy(d['images']).permute(0, 3, 1, 2).float() / 255.
ident, split, LM = d['identity'], d['split'], d['landmarks']
fn, af = FaceNet(), ArcFace()
try:  # LPIPS (AlexNet backbone, v0.1). Needs internet once to fetch weights (works in Colab).
    import lpips
    LP = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval()
except Exception as ex:
    LP = None; print('LPIPS unavailable here:', type(ex).__name__)

files = sorted(glob.glob(f'data/{ds}_adv_*.pt'))
names = [f.split('_adv_')[1][:-3] for f in files]
if only: files, names = zip(*[(f, n) for f, n in zip(files, names) if n in only])

iA, iB = torch.load(files[0])['iA'], torch.load(files[0])['iB']
XA, XB = X[iA], X[iB]
eA, eB = fn.embed(XA), fn.embed(XB)
aA, aB = af.embed(XA, LM[iA]), af.embed(XB, LM[iB])
# identification gallery: photo B of every attacked person + one photo of each calibration person
cal_first = np.array([np.where(ident == i)[0][0] for i in dict.fromkeys(ident[split == 'calib'])])
G = torch.cat([eB, fn.embed(X[cal_first])]); g_lab = np.arange(len(G))  # attacked person j <-> gallery j
top5_clean = fn.topk(XA)
clean_match = (cos_dist(eA, eB) < T).numpy(); clean_match_a = (cos_dist(aA, aB) < TA).numpy()
rank1_clean = (cos_dist(eA[:, None], G[None]).argmin(1).numpy() == np.arange(len(eA)))
tgt = np.roll(np.arange(len(iA)), -1)

rows = []
for f, name in zip(files, names):
    adv = torch.load(f)['adv']
    e = fn.embed(adv); a = af.embed(adv, LM[iA])
    self_d, ref_d = cos_dist(e, eA).numpy(), cos_dist(e, eB).numpy()
    a_self, a_ref = cos_dist(a, aA).numpy(), cos_dist(a, aB).numpy()
    rank1 = cos_dist(e[:, None], G[None]).argmin(1).numpy() == np.arange(len(e))
    t5 = fn.topk(adv)
    imp_d = cos_dist(e, eB[tgt]).numpy(); a_imp = cos_dist(a, aB[tgt]).numpy()
    if LP is not None:
        with torch.no_grad():
            lp = torch.cat([LP(adv[i:i + 32] * 2 - 1, XA[i:i + 32] * 2 - 1).flatten() for i in range(0, len(adv), 32)]).numpy()
    else:
        lp = np.full(len(adv), np.nan)
    for j in range(len(iA)):
        x, xa = XA[j].numpy(), adv[j].numpy()
        r = dict(attack=name, dataset=ds, person=ident[iA[j]], idx=int(iA[j]))
        r.update(image_metrics(x, xa)); r['LPIPS'] = float(lp[j])
        r.update(region_concentration(xa - x, LM[iA[j]]))
        r.update(self_dist=self_d[j], ref_dist=ref_d[j], clean_ref_dist=float(cos_dist(eA[j], eB[j])),
                 clean_match=bool(clean_match[j]), adv_match=bool(ref_d[j] < T),
                 dodge=bool(clean_match[j] and ref_d[j] >= T),
                 rank1_clean=bool(rank1_clean[j]), rank1_adv=bool(rank1[j]),
                 top5_consist=len(set(top5_clean[j].tolist()) & set(t5[j].tolist())) / 5,
                 imperson_dist=imp_d[j], imperson_success=bool(imp_d[j] < T),
                 arc_self_dist=a_self[j], arc_ref_dist=a_ref[j], arc_clean_match=bool(clean_match_a[j]),
                 arc_dodge=bool(clean_match_a[j] and a_ref[j] >= TA),
                 arc_imperson_success=bool(a_imp[j] < TA))
        rows.append(r)
    print(f'{name:14s} self_dist={self_d.mean():.3f} dodge={np.mean([x["dodge"] for x in rows[-len(iA):]]):.2f}', flush=True)

out = f'results/{ds}_per_image.csv'
df = pd.DataFrame(rows)
if only and os.path.exists(out):
    old = pd.read_csv(out); df = pd.concat([old[~old.attack.isin(names)], df])
df.to_csv(out, index=False)
print('wrote', out, df.shape)
