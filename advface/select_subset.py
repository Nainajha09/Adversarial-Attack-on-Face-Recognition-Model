"""Step 0: choose which photos to use (reproducible, seed 0).
Picks identities with >= 2 photos and splits them into DISJOINT groups:
  calib  : N_CAL identities (2 photos each) -> used only to calibrate the verification threshold
  attack : N_ATT identities (2 photos each) -> photo 1 is attacked, photo 2 is the reference
usage:
  python select_subset.py lfw    <lfw_root_with_person_folders>
  python select_subset.py celeba <img_align_celeba_dir> <identity_CelebA.txt>
"""
import sys, os, random, collections
N_CAL, N_ATT = 150, 100
ds = sys.argv[1]
c = collections.defaultdict(list)
if ds == 'lfw':
    root = sys.argv[2]
    for person in sorted(os.listdir(root)):
        for f in sorted(os.listdir(os.path.join(root, person))):
            if f.endswith('.jpg'): c[person].append(f'{person}/{f}')
else:
    for line in open(sys.argv[3]):
        f, pid = line.split()
        if os.path.exists(os.path.join(sys.argv[2], f)): c[pid].append(f)
ids = sorted(k for k, v in c.items() if len(v) >= 2)
random.seed(0); random.shuffle(ids)
cal, att = ids[:N_CAL], ids[N_CAL:N_CAL + N_ATT]
sel = [p for k in cal + att for p in sorted(c[k])[:2]]
os.makedirs('data', exist_ok=True)
open(f'data/{ds}_selection.txt', 'w').write('\n'.join(sel))
open(f'data/{ds}_split.txt', 'w').write('\n'.join([f'calib {k}' for k in cal] + [f'attack {k}' for k in att]))
print(f'{ds}: {len(ids)} identities with >=2 photos; selected {len(sel)} photos ({len(cal)} calib + {len(att)} attack identities)')
