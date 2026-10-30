"""Shared code: models, pixel conventions and all evaluation metrics.

Conventions (state these in the paper):
  * Images are handled in [0, 1]. Every epsilon below is in [0, 1] units (8/255 = 0.0314).
  * FaceNet normalisation is done INSIDE the model wrapper, exactly once:
    fixed_image_standardization, (255*x - 127.5) / 128, as used by facenet-pytorch.
  * FaceNet = facenet-pytorch InceptionResnetV1, 512-d embeddings, trained on VGGFace2.
  * Decision rule: verification. Two faces are "same person" when the cosine distance
    between their L2-normalised embeddings is below a threshold calibrated on a disjoint
    set of LFW identities (best balanced accuracy).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity
from skimage.transform import SimilarityTransform, warp

torch.set_num_threads(max(1, torch.get_num_threads()))
import os
WEIGHTS = os.environ.get('ADVFACE_WEIGHTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'))
DEV = 'cpu'


# ----------------------------------------------------------------------------- models
class FaceNet(nn.Module):
    """InceptionResnetV1 (VGGFace2). forward(x in [0,1]) -> L2-normalised 512-d embedding."""

    def __init__(self):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(classify=True, num_classes=8631)
        self.net.load_state_dict(torch.load(f'{WEIGHTS}/vggface2.pt', map_location='cpu', weights_only=True))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @staticmethod
    def standardise(x):
        return (x * 255.0 - 127.5) / 128.0

    def forward(self, x):
        self.net.classify = False
        return self.net(self.standardise(x))          # already L2-normalised by the model

    def logits(self, x):
        self.net.classify = True
        out = self.net(self.standardise(x))
        self.net.classify = False
        return out

    @torch.no_grad()
    def embed(self, x, bs=32):
        return torch.cat([self(x[i:i + bs]) for i in range(0, len(x), bs)])

    @torch.no_grad()
    def topk(self, x, k=5, bs=32):
        return torch.cat([self.logits(x[i:i + bs]).topk(k, 1).indices for i in range(0, len(x), bs)])


class ArcFace:
    """Transfer model: InsightFace ArcFace (MobileFaceNet backbone, WebFace600K), ONNX.
    Input must be aligned to the standard 112x112 ArcFace 5-point template."""
    TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

    def __init__(self):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(f'{WEIGHTS}/w600k_mbf.onnx', providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name

    def align(self, img, lm):  # img HxWx3 float [0,1]; lm 5x2 in img coordinates
        t = SimilarityTransform(); t.estimate(self.TEMPLATE, lm)  # maps template -> image
        return warp(img, t, output_shape=(112, 112), order=1, mode='edge')

    def embed(self, imgs, lms, bs=64):  # imgs N x 3 x H x W torch [0,1]
        arr = imgs.permute(0, 2, 3, 1).cpu().numpy()
        al = np.stack([self.align(a, l) for a, l in zip(arr, lms)])
        blob = ((al * 255.0 - 127.5) / 127.5).transpose(0, 3, 1, 2).astype(np.float32)
        out = np.concatenate([self.sess.run(None, {self.inp: blob[i:i + bs]})[0] for i in range(0, len(blob), bs)])
        return torch.from_numpy(out / np.linalg.norm(out, axis=1, keepdims=True))


def cos_dist(a, b):
    """Cosine distance = 1 - cosine similarity. Range [0, 2]."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def calibrate_threshold(emb, ident):
    """Best balanced-accuracy threshold from all genuine pairs and all impostor pairs."""
    ident = np.asarray(ident)
    d = cos_dist(emb[:, None, :], emb[None, :, :]).numpy()
    iu = np.triu_indices(len(ident), 1)
    same = (ident[:, None] == ident[None, :])[iu]
    dd = d[iu]
    gen, imp = dd[same], dd[~same]
    cand = np.linspace(0, 2, 2001)
    tpr = np.array([(gen < t).mean() for t in cand])
    tnr = np.array([(imp >= t).mean() for t in cand])
    bal = (tpr + tnr) / 2
    i = int(np.argmax(bal))
    far1 = cand[np.where(tnr >= 0.99)[0].max()]  # largest threshold with FAR <= 1%
    return dict(threshold=float(cand[i]), bal_acc=float(bal[i]), tpr=float(tpr[i]), tnr=float(tnr[i]),
                threshold_far1=float(far1), n_genuine=int(len(gen)), n_impostor=int(len(imp)))


# ----------------------------------------------------------------------------- image metrics
def _gray(x):  # 3xHxW -> HxW
    return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]


def uqi(a, b, win=8):
    """Universal Quality Index (Wang & Bovik 2002): mean of local Q over 8x8 sliding windows,
    averaged over channels. Identical images -> 1 (windows with zero variance count as 1)."""
    qs = []
    for c in range(a.shape[0]):
        x, y = a[c].astype(np.float64), b[c].astype(np.float64)
        mx, my = uniform_filter(x, win), uniform_filter(y, win)
        vx = uniform_filter(x * x, win) - mx ** 2
        vy = uniform_filter(y * y, win) - my ** 2
        cxy = uniform_filter(x * y, win) - mx * my
        num = 4 * cxy * mx * my
        den = (vx + vy) * (mx ** 2 + my ** 2)
        q = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
        h = win // 2
        qs.append(q[h:-h, h:-h].mean())
    return float(np.mean(qs))


def nmi(a, b, bins=64):
    """Normalised mutual information, Eq. (25): I(X;Y)/sqrt(H(X)H(Y)), in [0,1], on grey levels."""
    ga, gb = _gray(a).ravel(), _gray(b).ravel()
    h, _, _ = np.histogram2d(ga, gb, bins=bins, range=[[0, 1], [0, 1]])
    p = h / h.sum(); px, py = p.sum(1), p.sum(0)
    nz = p > 0
    I = (p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum()
    Hx = -(px[px > 0] * np.log(px[px > 0])).sum(); Hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return float(I / np.sqrt(Hx * Hy))


def image_metrics(x, xa):
    """x, xa: 3xHxW numpy float in [0,1]. Returns pixel / perceptual / statistical metrics."""
    d = xa - x
    mse = float((d ** 2).mean())
    return dict(
        L2=float(np.sqrt((d ** 2).sum())),
        Linf=float(np.abs(d).max()),
        L0_frac=float((np.abs(d).max(0) > 1e-6).mean()),           # fraction of pixels changed
        MSE=mse,
        PSNR=float(10 * np.log10(1.0 / mse)) if mse > 0 else float('inf'),
        SNR_dB=float(20 * np.log10(np.linalg.norm(x) / np.linalg.norm(d))) if mse > 0 else float('inf'),
        SSIM=float(structural_similarity(x.transpose(1, 2, 0), xa.transpose(1, 2, 0), channel_axis=-1, data_range=1.0)),
        UQI=uqi(x, xa),
        PCC=float(np.corrcoef(x.ravel(), xa.ravel())[0, 1]),
        NMI=nmi(x, xa),
        # total variation OF THE PERTURBATION, per pixel (how grainy the noise is)
        TV_pert=float((np.abs(d[:, 1:, :] - d[:, :-1, :]).sum() + np.abs(d[:, :, 1:] - d[:, :, :-1]).sum()) / (x.shape[1] * x.shape[2])),
    )


# ----------------------------------------------------------------------------- region analysis
REGIONS = ['eyes', 'nose', 'mouth', 'rest']


def region_masks(lm, size=160):
    """Boxes around MTCNN landmarks, sized by inter-ocular distance (IOD).
    eyes: two squares of side 0.5*IOD; nose: square 0.5*IOD; mouth: (corner dist + 0.25 IOD) x 0.4 IOD.
    'rest' = everything else (cheeks, forehead, hair, background margin)."""
    iod = np.linalg.norm(lm[1] - lm[0])
    yy, xx = np.mgrid[0:size, 0:size]
    def box(cx, cy, w, h):
        return (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= h / 2)
    s = 0.5 * iod
    eyes = box(*lm[0], s, s) | box(*lm[1], s, s)
    nose = box(*lm[2], s, s) & ~eyes
    mc = (lm[3] + lm[4]) / 2
    mouth = box(mc[0], mc[1], np.linalg.norm(lm[4] - lm[3]) + 0.25 * iod, 0.4 * iod) & ~eyes & ~nose
    rest = ~(eyes | nose | mouth)
    return dict(eyes=eyes, nose=nose, mouth=mouth, rest=rest)


def region_concentration(d, lm):
    """Share of perturbation energy |d|^2 in each region, and that share divided by the region's
    area share (1.0 = no concentration; >1 = the attack focuses there)."""
    e = (d ** 2).sum(0)
    tot = e.sum()
    out = {}
    for k, m in region_masks(lm, d.shape[1]).items():
        share = float(e[m].sum() / tot) if tot > 0 else np.nan
        out[f'share_{k}'] = share
        out[f'conc_{k}'] = share / float(m.mean()) if tot > 0 else np.nan
    return out
