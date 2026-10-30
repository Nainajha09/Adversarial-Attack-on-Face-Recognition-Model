"""Step 1: face detection + cropping with MTCNN (facenet-pytorch).

For every image we keep:
  * a 160x160 RGB crop (uint8) - the exact input FaceNet sees
  * the 5 MTCNN landmarks (eyes, nose, mouth corners) mapped into crop coordinates
  * the identity name and which split it belongs to (calibration / attack)

Note: facenet-pytorch's MTCNN crops a margin box and resizes. It does NOT rotate the
face to a canonical pose. The landmarks are used later for (a) the region analysis
and (b) aligning faces for the ArcFace transfer model.
"""
import argparse, os, numpy as np, torch
from PIL import Image
from facenet_pytorch import MTCNN

IMG_SIZE, MARGIN = 160, 14


def crop_box(box, w, h):
    # identical to facenet_pytorch.models.utils.detect_face.extract_face
    m = [MARGIN * (box[2] - box[0]) / (IMG_SIZE - MARGIN), MARGIN * (box[3] - box[1]) / (IMG_SIZE - MARGIN)]
    return [int(max(box[0] - m[0] / 2, 0)), int(max(box[1] - m[1] / 2, 0)),
            int(min(box[2] + m[0] / 2, w)), int(min(box[3] + m[1] / 2, h))]


def main(a):
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=MARGIN, keep_all=False, device='cpu')
    split = dict(l.split() [::-1] for l in open(a.split))  # identity -> split
    files = [l.strip() for l in open(a.list) if l.strip()]
    file2id = {}
    if a.identity:  # CelebA: filename -> identity id
        for line in open(a.identity):
            fn_, pid = line.split(); file2id[fn_] = pid
    imgs, lms, ids, splits, names, skipped = [], [], [], [], [], []
    for f in files:
        im = Image.open(os.path.join(a.root, f)).convert('RGB')
        boxes, probs, pts = mtcnn.detect(im, landmarks=True)
        if boxes is None:
            skipped.append(f); continue
        k = int(np.argmax(probs))
        x0, y0, x1, y1 = crop_box(boxes[k], *im.size)
        crop = im.crop((x0, y0, x1, y1)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        lm = pts[k].copy()
        lm[:, 0] = (lm[:, 0] - x0) * IMG_SIZE / (x1 - x0)
        lm[:, 1] = (lm[:, 1] - y0) * IMG_SIZE / (y1 - y0)
        ident = f.split('/')[0] if '/' in f else file2id[f]
        imgs.append(np.asarray(crop)); lms.append(lm); ids.append(ident)
        splits.append(split[ident]); names.append(f)
    np.savez_compressed(a.out, images=np.stack(imgs).astype(np.uint8), landmarks=np.stack(lms).astype(np.float32),
                        identity=np.array(ids), split=np.array(splits), fname=np.array(names))
    print(f'kept {len(imgs)}  skipped {len(skipped)} {skipped[:5]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--list', required=True)
    p.add_argument('--split', required=True); p.add_argument('--out', required=True)
    p.add_argument('--identity', default=None, help='identity_CelebA.txt (CelebA only)')
    main(p.parse_args())
