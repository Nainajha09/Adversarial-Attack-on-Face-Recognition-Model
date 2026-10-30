"""Step 3: run one attack on the attack split and save adversarial images.
usage: python run_attack.py <dataset> <attack>"""
import json, sys, time, numpy as np, torch
import attacks as A
from common import FaceNet

ds, name = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
d = np.load(f'data/{ds}.npz')
thr = json.load(open(f'data/{ds}_thresholds.json'))['facenet']['threshold']
X = torch.from_numpy(d['images']).permute(0, 3, 1, 2).float() / 255.
ident, split = d['identity'], d['split']

# probe/reference pairs: first photo of each attack identity is attacked (probe A),
# its second photo is the reference B used to decide success.
att_ids = [i for i in dict.fromkeys(ident[split == 'attack']) if (ident == i).sum() >= 2]
iA = np.array([np.where(ident == i)[0][0] for i in att_ids]); iB = np.array([np.where(ident == i)[0][1] for i in att_ids])
model = FaceNet()
XA = X[iA]; E0 = model.embed(XA)

t = time.time()
if name == 'fgsm':        adv = A.batched(lambda x, e: A.fgsm(model, x, e), XA, E0)
elif name == 'pgd_linf':  adv = A.batched(lambda x, e: A.pgd_linf(model, x, e), XA, E0)
elif name == 'mim':       adv = A.batched(lambda x, e: A.mim(model, x, e), XA, E0)
elif name in ('pgd_l2', 'gauss_matched'):
    ref = torch.load(f'data/{ds}_adv_pgd_linf.pt')['adv']
    l2 = float(np.median((ref - XA).flatten(1).norm(dim=1).numpy()))
    print('matched L2 =', l2)
    adv = A.batched(lambda x, e: A.pgd_l2(model, x, e, eps=l2), XA, E0) if name == 'pgd_l2' else A.gaussian_matched(XA, l2)
elif name == 'cw_l2':     adv = A.batched(lambda x, e: A.cw_l2(model, x, e, target_dist=thr + 0.1), XA, E0)
elif name == 'sparse':    adv = A.batched(lambda x, e: A.sparse_pgd(model, x, e), XA, E0)
elif name == 'square':    adv = A.batched(lambda x, e: A.square_linf(model, x, e, queries=300), XA, E0, bs=128)
elif name == 'nes':       adv = A.batched(lambda x, e: A.nes_linf(model, x, e, iters=15, samples=20), XA, E0, bs=128)
elif name == 'uap':
    cal = np.where(split == 'calib')[0]
    delta = A.train_uap(model, X[cal], model.embed(X[cal]))
    adv = (XA + delta).clamp(0, 1)
elif name == 'spatial':   adv = A.batched(lambda x, e: A.spatial_grid(model, x, e), XA, E0, bs=128)
elif name == 'imperson':
    tgt = np.roll(np.arange(len(iA)), -1)               # target = the next identity in the list
    adv = A.batched(lambda x, e: A.impersonation(model, x, e), XA, E0[tgt])
elif name == 'rand_sign': adv = A.random_sign(XA)
elif name == 'jpeg50':    adv = A.jpeg(XA, 50)
else: raise SystemExit('unknown attack ' + name)
print(f'{name}: {time.time() - t:.0f}s', flush=True)
torch.save(dict(adv=adv.float(), iA=iA, iB=iB), f'data/{ds}_adv_{name}.pt')
