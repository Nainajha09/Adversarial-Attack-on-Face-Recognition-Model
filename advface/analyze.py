"""Step 6: paper tables (mean ± 95% CI), statistical tests and figures."""
import sys, json, os, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ds = sys.argv[1]
OUT = f'results/{ds}'; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(f'results/{ds}_per_image.csv')
setl = pd.read_csv(f'results/{ds}_set_level.csv') if os.path.exists(f'results/{ds}_set_level.csv') else None
thr = json.load(open(f'data/{ds}_thresholds.json'))

ORDER = [  # (key, display name, family)
    ('fgsm', 'FGSM', 'White-box, L∞ 8/255'), ('pgd_linf', 'PGD-L∞', 'White-box, L∞ 8/255'),
    ('mim', 'MIM', 'White-box, L∞ 8/255'), ('pgd_l2', 'PGD-L2 (matched L2)', 'White-box, L2'),
    ('cw_l2', 'C&W-L2 (min-norm)', 'White-box, minimum-norm'), ('sparse', 'Sparse-PGD (1% px)', 'White-box, L0'),
    ('square', 'Square (300 q)', 'Black-box, L∞ 8/255'), ('nes', 'NES (300 q)', 'Black-box, L∞ 8/255'),
    ('uap', 'Universal (UAP)', 'Universal, L∞ 8/255'), ('spatial', 'Spatial (rot+shift)', 'Spatial'),
    ('imperson', 'Impersonation', 'Feature-level, targeted'),
    ('rand_sign', 'Random ±8/255 noise', 'Control'), ('gauss_matched', 'Gaussian noise (matched L2)', 'Control'),
    ('jpeg50', 'JPEG q=50', 'Control')]
ORDER = [o for o in ORDER if o[0] in set(df.attack)]
NAME = {k: n for k, n, _ in ORDER}; FAM = {k: f for k, _, f in ORDER}
CONTROLS = {'rand_sign', 'gauss_matched', 'jpeg50'}


def ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n_boot, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan, np.nan
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def fmt(m, lo, hi, d=3):
    return f'{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]' if np.isfinite(m) else '–'


def table(cols, digits):
    rows = []
    for k, n, f in ORDER:
        s = df[df.attack == k]; r = {'Attack': n, 'Family': f}
        for c, dg in zip(cols, digits):
            r[c] = fmt(*ci(s[c]), dg)
        rows.append(r)
    return pd.DataFrame(rows)


tables = {}
_t1 = ['LPIPS', 'SSIM', 'PSNR', 'SNR_dB', 'UQI'] if df['LPIPS'].notna().any() else ['SSIM', 'PSNR', 'SNR_dB', 'UQI']
tables['T1_perceptual'] = table(_t1, [4, 3, 1, 1, 3][-len(_t1):])
t2 = table(['MSE', 'TV_pert', 'PCC', 'NMI'], [5, 4, 4, 3])
if setl is not None:
    sl = setl.set_index('attack')
    t2['FID'] = [f"{sl.loc[k, 'FID']:.1f}" for k, _, _ in ORDER]
    t2['KID×1000'] = [f"{sl.loc[k, 'KIDx1000']:.2f} ± {sl.loc[k, 'KIDx1000_sd']:.2f}" for k, _, _ in ORDER]
tables['T2_statistical'] = t2
tables['T5_magnitude'] = table(['L2', 'Linf', 'L0_frac'], [2, 4, 3])

rows = []
for k, n, f in ORDER:
    s = df[df.attack == k]; cm = s[s.clean_match]; ca = s[s.arc_clean_match]
    imp = k == 'imperson'
    succ = cm.imperson_success if imp else cm.dodge
    asucc = s.arc_imperson_success if imp else ca.arc_dodge
    rows.append({'Attack': n,
                 'Cosine drift (self)': fmt(*ci(s.self_dist)),
                 'Cosine dist. to other photo': fmt(*ci(s.ref_dist)),
                 ('Impersonation success' if imp else 'Verification flip rate'): fmt(*wilson(succ.sum(), len(succ)), 2),
                 'Rank-1 ID acc. (clean→adv)': f'{s.rank1_clean.mean():.2f} → {s.rank1_adv.mean():.2f}',
                 'Top-5 consistency': fmt(*ci(s.top5_consist), 2),
                 'ArcFace transfer flip rate': fmt(*wilson(asucc.sum(), len(asucc)), 2)})
t3 = pd.DataFrame(rows).fillna('')
tables['T3_identity'] = t3
tables['T4_regions'] = table(['conc_eyes', 'conc_nose', 'conc_mouth', 'conc_rest'], [2, 2, 2, 2])

for k, t in tables.items():
    t.to_csv(f'{OUT}/{k}.csv', index=False)

# ----------------------------------------------------------------------------- statistics
st = {}
adv = df[~df.attack.isin(CONTROLS)]
for metric in ['self_dist', 'SSIM']:
    groups = [g[metric].values for _, g in df.groupby('attack')]
    h, p = stats.kruskal(*groups); st[f'kruskal_{metric}'] = dict(H=float(h), p=float(p))
r, p = stats.spearmanr(1 - adv.SSIM, adv.self_dist); st['spearman_(1-SSIM)_vs_drift_all_adv'] = dict(rho=float(r), p=float(p))
r, p = stats.spearmanr(adv.L2, adv.self_dist); st['spearman_L2_vs_drift_all_adv'] = dict(rho=float(r), p=float(p))
per = {}
for k, g in adv.groupby('attack'):
    if g.SSIM.std() > 0:
        r, p = stats.spearmanr(1 - g.SSIM, g.self_dist); per[NAME[k]] = dict(rho=float(r), p=float(p))
st['spearman_within_attack'] = per
# matched-budget comparisons (paired, same images)
def paired(a, b, m):
    x = df[df.attack == a].sort_values('idx')[m].values; y = df[df.attack == b].sort_values('idx')[m].values
    return dict(mean_a=float(x.mean()), mean_b=float(y.mean()), wilcoxon_p=float(stats.wilcoxon(x, y).pvalue))
pairs = [('pgd_linf', 'rand_sign'), ('pgd_linf', 'fgsm'), ('pgd_linf', 'pgd_l2'), ('pgd_l2', 'gauss_matched'), ('pgd_linf', 'square')]
st['paired'] = {f'{NAME[a]} vs {NAME[b]} [{m}]': paired(a, b, m) for a, b in pairs if a in NAME and b in NAME for m in ['self_dist', 'SSIM']}
# region concentration: each attack vs Gaussian control (Mann-Whitney), eyes and mouth
if 'gauss_matched' in NAME:
    base = df[df.attack == 'gauss_matched']
    st['regions_vs_gaussian'] = {f'{NAME[k]} [{r_}]': float(stats.mannwhitneyu(df[df.attack == k][f'conc_{r_}'].dropna(), base[f'conc_{r_}'].dropna()).pvalue)
                                 for k, _, _ in ORDER if k not in ('gauss_matched', 'spatial') for r_ in ['eyes', 'mouth']}
st['thresholds'] = thr
st['n_images_per_attack'] = int(df.groupby('attack').size().min())
json.dump(st, open(f'{OUT}/statistics.json', 'w'), indent=1)

# ----------------------------------------------------------------------------- figures
cmap = plt.get_cmap('tab20')
COL = {k: cmap(i) for i, (k, _, _) in enumerate(ORDER)}
plt.rcParams.update({'font.size': 9})

# F1: perceptibility vs identity drift (the key figure)
fig, ax = plt.subplots(figsize=(8, 5.5))
for k, n, _ in ORDER:
    s = df[df.attack == k]
    ax.scatter(np.clip(1 - s.SSIM, 1e-4, None), s.self_dist, s=10, alpha=.55, color=COL[k], marker='x' if k in CONTROLS else 'o', label=n)
ax.axhline(thr['facenet']['threshold'], ls='--', c='k', lw=1); ax.text(0.9, thr['facenet']['threshold'] + .02, 'verification threshold', ha='right')
ax.set_xscale('log'); ax.set_xlim(8e-5, 1); ax.set_xlabel('1 − SSIM (visible change →)'); ax.set_ylabel('FaceNet cosine distance to own clean embedding')
ax.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(1, 1)); ax.set_title(f'{ds.upper()}: visibility vs identity drift')
fig.tight_layout(); fig.savefig(f'{OUT}/F1_visibility_vs_drift.png', dpi=200); plt.close(fig)

def boxes(metric, ylabel, fname, hline=None, log=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [df[df.attack == k][metric].replace([np.inf], np.nan).dropna() for k, _, _ in ORDER]
    b = ax.boxplot(data, patch_artist=True, showfliers=False)
    for p_, (k, _, _) in zip(b['boxes'], ORDER): p_.set_facecolor(COL[k])
    ax.set_xticks(range(1, len(ORDER) + 1)); ax.set_xticklabels([n for _, n, _ in ORDER], rotation=40, ha='right')
    if hline is not None: ax.axhline(hline, ls='--', c='k', lw=1)
    if log: ax.set_yscale('log')
    ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)

boxes('self_dist', 'Cosine distance (identity drift)', 'F2_drift_boxplot.png', thr['facenet']['threshold'])
boxes('SSIM', 'SSIM (higher = less visible)', 'F3_ssim_boxplot.png')
boxes('NMI', 'Normalised mutual information', 'F4_nmi_boxplot.png')
boxes('TV_pert', 'Total variation of perturbation (per pixel)', 'F5_tv_boxplot.png', log=True)

# F6: region concentration
fig, ax = plt.subplots(figsize=(9, 4)); w = .2
ks = [k for k, _, _ in ORDER if k != 'spatial']
for i, r_ in enumerate(['eyes', 'nose', 'mouth', 'rest']):
    ax.bar(np.arange(len(ks)) + (i - 1.5) * w, [df[df.attack == k][f'conc_{r_}'].mean() for k in ks], w, label=r_)
ax.axhline(1, c='k', lw=1, ls='--'); ax.set_xticks(range(len(ks))); ax.set_xticklabels([NAME[k] for k in ks], rotation=40, ha='right')
ax.set_ylabel('Energy share ÷ area share (1 = uniform)'); ax.legend(); fig.tight_layout(); fig.savefig(f'{OUT}/F6_region_concentration.png', dpi=200); plt.close(fig)

# F7: average perturbation-energy heatmaps + F8 qualitative examples
d = np.load(f'data/{ds}.npz'); X = torch.from_numpy(d['images']).permute(0, 3, 1, 2).float() / 255.
advs = {k: torch.load(f'data/{ds}_adv_{k}.pt') for k, _, _ in ORDER}
iA = advs[ORDER[0][0]]['iA']; XA = X[iA]; lm = d['landmarks'][iA].mean(0)
n = len(ORDER); c_ = 7; r_ = int(np.ceil(n / c_))
fig, axs = plt.subplots(r_, c_, figsize=(c_ * 1.7, r_ * 1.9))
for ax, (k, nm, _) in zip(axs.flat, ORDER):
    e = ((advs[k]['adv'] - XA) ** 2).sum(1).mean(0).numpy()
    ax.imshow(e / (e.max() + 1e-12), cmap='inferno'); ax.scatter(lm[:, 0], lm[:, 1], s=6, c='cyan'); ax.set_title(nm, fontsize=6.5); ax.axis('off')
for ax in list(axs.flat)[n:]: ax.axis('off')
fig.suptitle('Mean perturbation energy (each map scaled to its own max; cyan = mean landmarks)', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/F7_mean_perturbation_heatmaps.png', dpi=200); plt.close(fig)

show = [k for k in ['fgsm', 'pgd_linf', 'cw_l2', 'sparse', 'square', 'uap', 'spatial', 'imperson', 'gauss_matched'] if k in advs]
for j in [0, 1]:
    fig, axs = plt.subplots(3, len(show) + 1, figsize=((len(show) + 1) * 1.45, 4.6))
    axs[0, 0].imshow(XA[j].permute(1, 2, 0)); axs[0, 0].set_title('Original', fontsize=7)
    axs[1, 0].axis('off'); axs[2, 0].axis('off')
    for i, k in enumerate(show, 1):
        a = advs[k]['adv'][j]; dlt = (a - XA[j]); s = df[(df.attack == k) & (df.idx == int(iA[j]))].iloc[0]
        axs[0, i].imshow(a.permute(1, 2, 0).clamp(0, 1)); axs[0, i].set_title(NAME[k], fontsize=6.5)
        axs[1, i].imshow((dlt / (2 * dlt.abs().max() + 1e-12) + .5).permute(1, 2, 0)); axs[1, i].set_title('perturbation (amplified)', fontsize=5.5)
        axs[2, i].axis('off'); axs[2, i].text(0, .5, f"SSIM {s.SSIM:.3f}\nPSNR {s.PSNR:.1f} dB\ndrift {s.self_dist:.2f}\n{'FLIPPED' if s.dodge else 'still matched'}", fontsize=6.5, va='center')
    for ax in axs[:2].flat: ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f'{OUT}/F8_qualitative_{j + 1}.png', dpi=200); plt.close(fig)
print('analysis written to', OUT)
