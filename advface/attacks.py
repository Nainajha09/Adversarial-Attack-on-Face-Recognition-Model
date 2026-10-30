"""All attacks, written directly against FaceNet's embedding (no dummy labels).

Threat model ("dodging"): the attacker perturbs photo A of a person so that FaceNet no
longer matches it to that person. Loss = cosine similarity between f(x_adv) and the clean
embedding e0 = f(x). Attacks minimise it. Success is judged separately (evaluate.py) against
a DIFFERENT photo B of the same person, using the calibrated verification threshold.

All images in [0,1]; eps are in [0,1] units. Every attack clips to the valid image range.
"""
import io, math, numpy as np, torch, torch.nn.functional as F
from PIL import Image

EPS = 8 / 255  # shared L-inf budget (= 16/255 in [-1,1] units)


def sim(model, x, e0):
    return F.cosine_similarity(model(x), e0, dim=-1)


def grad_of_sim(model, x, e0):
    x = x.clone().requires_grad_(True)
    s = sim(model, x, e0)
    g, = torch.autograd.grad(s.sum(), x)
    return g, s.detach()


def rand_start(x, r=2 / 255, seed=0):
    """The clean image is a stationary point of sim(f(x'), f(x)) (similarity is already maximal,
    so the gradient is ~0). Every gradient-based attack therefore starts from a small random
    perturbation of size r (cf. R+FGSM, Tramer et al. 2018)."""
    g = torch.Generator().manual_seed(seed)
    return (x + r * (torch.rand(x.shape, generator=g) * 2 - 1)).clamp(0, 1)


def batched(fn, X, E, bs=16, **kw):
    out = [fn(X[i:i + bs], E[i:i + bs], **kw) for i in range(0, len(X), bs)]
    return torch.cat(out)


# ------------------------------------------------------------------ white-box, fixed budget
def fgsm(model, x, e0, eps=EPS, r=2 / 255):
    """R+FGSM: random start of size r, then one gradient-sign step; total L-inf <= eps."""
    g = torch.Generator().manual_seed(0)
    x0 = (x + r * (torch.randint(0, 2, x.shape, generator=g).float() * 2 - 1)).clamp(0, 1)
    gr, _ = grad_of_sim(model, x0, e0)
    xa = x0 - (eps - r) * gr.sign()
    return (x + (xa - x).clamp(-eps, eps)).clamp(0, 1)


def pgd_linf(model, x, e0, eps=EPS, steps=20, alpha=2 / 255, seed=0):
    torch.manual_seed(seed)
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        g, _ = grad_of_sim(model, xa, e0)
        xa = xa - alpha * g.sign()
        xa = (x + (xa - x).clamp(-eps, eps)).clamp(0, 1)
    return xa


def mim(model, x, e0, eps=EPS, steps=20, mu=1.0):
    alpha = eps / 10
    xa, g_acc = rand_start(x, 1 / 255), torch.zeros_like(x)
    for _ in range(steps):
        g, _ = grad_of_sim(model, xa, e0)
        g_acc = mu * g_acc + g / g.abs().flatten(1).sum(1).view(-1, 1, 1, 1).clamp_min(1e-12)
        xa = xa - alpha * g_acc.sign()
        xa = (x + (xa - x).clamp(-eps, eps)).clamp(0, 1)        # projection + range clip
    return xa


def _l2_project(d, eps):
    n = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return d * torch.clamp(eps / n.clamp_min(1e-12), max=1.0)


def pgd_l2(model, x, e0, eps=4.0, steps=20, seed=0):
    torch.manual_seed(seed)
    alpha = 2.5 * eps / steps
    d = torch.randn_like(x); d = _l2_project(d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * eps * 0.5, eps)
    xa = (x + d).clamp(0, 1)
    for _ in range(steps):
        g, _ = grad_of_sim(model, xa, e0)
        g = g / g.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        xa = xa - alpha * g
        xa = (x + _l2_project(xa - x, eps)).clamp(0, 1)
    return xa


# ------------------------------------------------------------------ white-box, minimum norm
def cw_l2(model, x, e0, target_dist=0.624, c=5.0, steps=60, lr=0.01):
    """Carlini-Wagner style (L2, tanh box). Minimise ||delta||^2 + c*max(0, target - d(f(x'), e0)).
    target_dist = verification threshold + margin kappa (0.1). Keeps the smallest successful delta.
    No fixed budget: we REPORT the norm it needs."""
    w = torch.atanh((rand_start(x, 1 / 255) * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    best = x.clone(); best_n = torch.full((len(x),), float('inf'))
    for _ in range(steps):
        xa = (torch.tanh(w) + 1) / 2
        dist = 1 - sim(model, xa, e0)
        n2 = ((xa - x) ** 2).flatten(1).sum(1)
        loss = (n2 + c * torch.clamp(target_dist - dist, min=0)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ok = (dist >= target_dist) & (n2.sqrt() < best_n)
            best[ok] = xa[ok].detach(); best_n[ok] = n2.sqrt()[ok]
    # images never reaching the target keep their final iterate (marked as failures later)
    with torch.no_grad():
        xa = (torch.tanh(w) + 1) / 2
        miss = torch.isinf(best_n)
        best[miss] = xa[miss]
    return best.detach()


# ------------------------------------------------------------------ sparse (L0)
def sparse_pgd(model, x, e0, k=256, steps=20, alpha=0.1):
    """Only k pixels (1% of 160x160) may change, but each by any amount (L-inf box = full range).
    Pixels are chosen by gradient saliency at the clean image, then optimised with PGD."""
    g, _ = grad_of_sim(model, rand_start(x), e0)    # saliency at a nearby point (see rand_start)
    sal = g.abs().sum(1).flatten(1)
    idx = sal.topk(k, dim=1).indices
    mask = torch.zeros_like(sal).scatter_(1, idx, 1.0).view(-1, 1, x.shape[2], x.shape[3])
    xa = x + mask * (rand_start(x, 8 / 255) - x)          # random start inside the k selected pixels
    for _ in range(steps):
        g, _ = grad_of_sim(model, xa, e0)
        xa = (xa - alpha * g.sign() * mask).clamp(0, 1)
    return xa


# ------------------------------------------------------------------ black-box, score-based
def _p_schedule(p_init, i, n_iters):
    """Original Square-attack schedule, rescaled to the query budget."""
    it = int(i / n_iters * 10000)
    for lim, div in [(10, 1), (50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32), (4000, 64), (6000, 128), (8000, 256)]:
        if it <= lim: return p_init / div
    return p_init / 512


@torch.no_grad()
def square_linf(model, x, e0, eps=EPS, queries=500, p_init=0.05, seed=0):
    """Square Attack (Andriushchenko et al. 2020), L-inf. Only uses the similarity score."""
    g = torch.Generator().manual_seed(seed)
    n, c, h, w = x.shape
    stripes = (torch.randint(0, 2, (n, c, 1, w), generator=g).float() * 2 - 1) * eps
    xa = (x + stripes).clamp(0, 1)
    best = sim(model, xa, e0)
    for i in range(1, queries):
        frac = _p_schedule(p_init, i, queries)
        s = max(int(round(math.sqrt(frac * h * w))), 1)
        r = torch.randint(0, h - s + 1, (n,), generator=g); col = torch.randint(0, w - s + 1, (n,), generator=g)
        cand = xa.clone()
        for j in range(n):
            delta = (torch.randint(0, 2, (c, 1, 1), generator=g).float() * 2 - 1) * eps
            cand[j, :, r[j]:r[j] + s, col[j]:col[j] + s] = (x[j, :, r[j]:r[j] + s, col[j]:col[j] + s] + delta).clamp(0, 1)
        sc = sim(model, cand, e0)
        better = sc < best
        xa[better] = cand[better]; best[better] = sc[better]
    return xa


@torch.no_grad()
def nes_linf(model, x, e0, eps=EPS, iters=25, samples=20, sigma=0.005, alpha=1 / 255, seed=0):
    """NES gradient estimation (Ilyas et al. 2018) with antithetic Gaussian samples.
    Queries = iters * samples (= 500). Uses the plain search gradient (no Fisher matrix)."""
    g = torch.Generator().manual_seed(seed)
    xa = x.clone()
    half = samples // 2
    xa = rand_start(x)
    for _ in range(iters):
        u = torch.randn((half,) + x.shape, generator=g)
        u = torch.cat([u, -u])                                    # antithetic pairs
        q = (xa.unsqueeze(0) + sigma * u).clamp(0, 1)
        s = torch.stack([sim(model, q[k], e0) for k in range(samples)])  # samples x n
        grad = (s.view(samples, -1, 1, 1, 1) * u).mean(0) / sigma
        xa = xa - alpha * grad.sign()
        xa = (x + (xa - x).clamp(-eps, eps)).clamp(0, 1)
    return xa


# ------------------------------------------------------------------ universal
def train_uap(model, X_train, E_train, eps=EPS, epochs=5, alpha=1 / 255, bs=16, seed=0):
    """One perturbation for ALL faces, trained on the calibration identities only and then
    applied to unseen attack identities."""
    torch.manual_seed(seed)
    delta = (torch.rand(1, *X_train.shape[1:]) * 2 - 1) * (2 / 255)
    for ep in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), bs):
            b = perm[i:i + bs]
            d = delta.clone().requires_grad_(True)
            s = sim(model, (X_train[b] + d).clamp(0, 1), E_train[b])
            gd, = torch.autograd.grad(s.mean(), d)
            delta = (delta - alpha * gd.sign()).clamp(-eps, eps)
        print(f'  uap epoch {ep}: mean train sim {s.mean().item():.3f}', flush=True)
    return delta


# ------------------------------------------------------------------ spatial
@torch.no_grad()
def spatial_grid(model, x, e0, rots=range(-15, 16, 5), shifts=(-6, 0, 6)):
    """Engstrom et al. 2019 worst-of-grid rotation + translation (no pixel noise at all).
    Reflection padding keeps the image natural-looking."""
    n, _, h, w = x.shape
    best, best_s = x.clone(), torch.full((n,), 2.0)
    for r in rots:
        for tx in shifts:
            for ty in shifts:
                if r == 0 and tx == 0 and ty == 0: continue
                a = math.radians(r)
                theta = torch.tensor([[math.cos(a), -math.sin(a), 2 * tx / w], [math.sin(a), math.cos(a), 2 * ty / h]],
                                     dtype=x.dtype).expand(n, 2, 3)
                grid = F.affine_grid(theta, x.shape, align_corners=False)
                xt = F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)
                s = sim(model, xt, e0)
                m = s < best_s
                best[m] = xt[m]; best_s[m] = s[m]
    return best


# ------------------------------------------------------------------ feature-level (targeted)
def impersonation(model, x, e_target, eps=EPS, steps=20, alpha=2 / 255):
    """Feature adversary: push f(x_adv) TOWARDS another person's embedding (maximise similarity)."""
    xa = x.clone()
    for _ in range(steps):
        g, _ = grad_of_sim(model, xa, e_target)
        xa = xa + alpha * g.sign()
        xa = (x + (xa - x).clamp(-eps, eps)).clamp(0, 1)
    return xa


# ------------------------------------------------------------------ non-adversarial controls
def random_sign(x, eps=EPS, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (x + eps * (torch.randint(0, 2, x.shape, generator=g).float() * 2 - 1)).clamp(0, 1)


def gaussian_matched(x, l2, seed=0):
    """Gaussian noise rescaled so each image's perturbation has the given L2 norm."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(x.shape, generator=g)
    d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * l2
    return (x + d).clamp(0, 1)


def jpeg(x, quality=50):
    out = []
    for im in x:
        buf = io.BytesIO()
        Image.fromarray((im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(buf, 'JPEG', quality=quality)
        out.append(torch.from_numpy(np.asarray(Image.open(buf).convert('RGB'))).permute(2, 0, 1).float() / 255)
    return torch.stack(out)
