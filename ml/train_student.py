#!/usr/bin/env python3
"""Train the small goal-conditioned driving policy (the "student") on LeRobot episodes.

Inputs per step: front camera (resized), lidar bird's-eye image, proprioceptive state,
and the instruction text. Output: (steer, speed). Loss is behaviour cloning on the
expert's actions, plus an optional distillation term that pulls the student's image
features toward a teacher's (Qwen-Drive features from extract_teacher.py), so the
teacher's scene/goal understanding is transferred while the actions stay exact for this
embodiment. Exports ONNX for the Jetson.

    python3 ml/train_student.py --data ~/lerobot/atlascar_sim --out runs/student \
        [--teacher-feats runs/teacher/feats.npz] [--epochs 30]
"""
import argparse, hashlib, json, os, time
import numpy as np
import cv2
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

FRONT_HW = (96, 128)          # resized front frame
BEV_HW = (96, 96)
TXT_DIM = 256
MAX_TOK = 24


# ── text: hashed bag-of-words (no downloads) with a learned embedding ─────────
def text_ids(s, n=4096, max_tok=MAX_TOK):
    toks = ''.join(c if c.isalnum() else ' ' for c in s.lower()).split()[:max_tok]
    ids = [1 + int(hashlib.md5(t.encode()).hexdigest(), 16) % (n - 1) for t in toks]
    return ids + [0] * (max_tok - len(ids))


class TextEnc(nn.Module):
    def __init__(self, n=4096, d=TXT_DIM):
        super().__init__(); self.emb = nn.EmbeddingBag(n, d, mode='mean', padding_idx=0)
    def forward(self, ids): return self.emb(ids)


# ── dataset: decode videos once into per-episode caches ───────────────────────
def load_episode(root, ei, cache):
    p = os.path.join(cache, f'ep{ei:06d}.npz')
    if os.path.isfile(p): return np.load(p)
    df = pd.read_parquet(os.path.join(root, 'data/chunk-000', f'episode_{ei:06d}.parquet'))
    def frames(key, hw):
        cap = cv2.VideoCapture(os.path.join(root, 'videos/chunk-000', key, f'episode_{ei:06d}.mp4'))
        out = []
        while True:
            ok, f = cap.read()
            if not ok: break
            out.append(cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA))
        cap.release(); return np.asarray(out, np.uint8)
    fr = frames('observation.images.front', FRONT_HW); bv = frames('observation.images.bev', BEV_HW)
    n = min(len(df), len(fr), len(bv))
    state = np.stack(df['observation.state'].values[:n]).astype(np.float32)
    act = np.stack(df['action'].values[:n]).astype(np.float32)
    task = int(df['task_index'].iloc[0])
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(p, front=fr[:n], bev=bv[:n, :, :, 0], state=state, act=act, task=task)
    return np.load(p)


class EpisodeSet(Dataset):
    def __init__(self, root, episodes, tasks, teacher=None, tdim=0, cache=None):
        self.items = []; self.tf = teacher; self.tdim = tdim
        for ei in episodes:
            e = load_episode(root, ei, cache or os.path.join(root, '_cache'))
            ids = np.asarray(text_ids(tasks[int(e['task'])]), np.int64)
            for k in range(len(e['act'])):
                self.items.append((e['front'][k], e['bev'][k], e['state'][k], ids, e['act'][k], int(ei), k))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fr, bv, st, ids, act, ei, k = self.items[i]
        x = {'front': torch.from_numpy(fr).permute(2, 0, 1).float() / 255.0,
             'bev': torch.from_numpy(bv).unsqueeze(0).float() / 255.0,
             'state': torch.from_numpy(st), 'ids': torch.from_numpy(ids),
             'act': torch.from_numpy(act)}
        if self.tf is not None:
            key = f'{ei}:{k}'
            x['tfeat'] = torch.from_numpy(self.tf[key]) if key in self.tf else torch.zeros(self.tdim)
            x['has_t'] = torch.tensor(1.0 if key in self.tf else 0.0)
        return x


# ── model ──────────────────────────────────────────────────────────────────────
def conv(i, o, s=2): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Student(nn.Module):
    def __init__(self, teacher_dim=0):
        super().__init__()
        self.front = nn.Sequential(conv(3, 32), conv(32, 64), conv(64, 96), conv(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.bev = nn.Sequential(conv(1, 16), conv(16, 32), conv(32, 64), conv(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.txt = TextEnc()
        self.state = nn.Sequential(nn.Linear(5, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 64 + TXT_DIM + 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
        self.proj = nn.Linear(128, teacher_dim) if teacher_dim else None
    def forward(self, front, bev, state, ids):
        f = self.front(front); b = self.bev(bev); t = self.txt(ids); s = self.state(state)
        return self.head(torch.cat([f, b, t, s], 1)), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='runs/student')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--teacher-feats', default=None)
    ap.add_argument('--distill-w', type=float, default=0.5); ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    root = os.path.expanduser(a.data); os.makedirs(a.out, exist_ok=True)
    info = json.load(open(os.path.join(root, 'meta/info.json')))
    tasks = {}
    for l in open(os.path.join(root, 'meta/tasks.jsonl')):
        d = json.loads(l); tasks[d['task_index']] = d['task']
    n_ep = info['total_episodes']; rng = np.random.default_rng(0); perm = rng.permutation(n_ep)
    n_val = max(1, int(n_ep * a.val_frac)); val_eps, tr_eps = sorted(perm[:n_val]), sorted(perm[n_val:])
    teacher = None; tdim = 0
    if a.teacher_feats and os.path.isfile(a.teacher_feats):
        z = np.load(a.teacher_feats); teacher = {k: z[k] for k in z.files}
        if teacher: tdim = int(next(iter(teacher.values())).shape[-1])
        print(f'teacher features: {len(teacher)} frames, dim {tdim}', flush=True)
        if not teacher: teacher = None
    print(f'episodes: {len(tr_eps)} train / {len(val_eps)} val; decoding videos (cached after first run)...', flush=True)
    tr = EpisodeSet(root, tr_eps, tasks, teacher, tdim); va = EpisodeSet(root, val_eps, tasks, teacher, tdim)
    A = np.stack([it[4] for it in tr.items]); mu, sd = A.mean(0), A.std(0) + 1e-6
    np.save(os.path.join(a.out, 'action_norm.npy'), np.stack([mu, sd]))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{len(tr)} train steps, {len(va)} val steps | device {dev} | action mean {mu.round(3)} std {sd.round(3)}', flush=True)
    mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
    model = Student(tdim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    dv = DataLoader(va, batch_size=a.bs, num_workers=a.workers)
    best = 1e9; log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = td = 0.0; nb = 0
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            pred, feat = model(b['front'], b['bev'], b['state'], b['ids'])
            loss = F.smooth_l1_loss(pred, (b['act'] - mu_t) / sd_t)
            dloss = torch.tensor(0.0, device=dev)
            if model.proj is not None and 'tfeat' in b:
                p = F.normalize(model.proj(feat), dim=1); t = F.normalize(b['tfeat'], dim=1)
                dloss = ((1 - (p * t).sum(1)) * b['has_t']).sum() / b['has_t'].sum().clamp(min=1)
                loss = loss + a.distill_w * dloss
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); td += dloss.item(); nb += 1
        sched.step()
        model.eval(); se = np.zeros(2); n = 0
        with torch.no_grad():
            for b in dv:
                b = {k: v.to(dev) for k, v in b.items()}
                pred, _ = model(b['front'], b['bev'], b['state'], b['ids'])
                se += (pred * sd_t + mu_t - b['act']).abs().sum(0).cpu().numpy(); n += len(pred)
        mae = se / max(n, 1)
        rec = {'epoch': ep, 'train_loss': tl / max(nb, 1), 'distill': td / max(nb, 1),
               'val_mae_steer_rad': float(mae[0]), 'val_mae_speed_mps': float(mae[1]), 'secs': round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        score = mae[0] / 0.4 + mae[1] / 1.5
        if score < best:
            best = score; torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
    model.load_state_dict(torch.load(os.path.join(a.out, 'best.pt'), map_location=dev)); model.eval()
    class Wrap(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, front, bev, state, ids): return s.m(front, bev, state, ids)[0] * sd_t + mu_t
    dummy = (torch.zeros(1, 3, *FRONT_HW, device=dev), torch.zeros(1, 1, *BEV_HW, device=dev),
             torch.zeros(1, 5, device=dev), torch.zeros(1, MAX_TOK, dtype=torch.long, device=dev))
    torch.onnx.export(Wrap(model), dummy, os.path.join(a.out, 'student.onnx'),
                      input_names=['front', 'bev', 'state', 'ids'], output_names=['action'], opset_version=17)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'done: best score {best:.3f}; {n_params/1e6:.2f} M params -> {a.out}/student.onnx')


if __name__ == '__main__':
    main()
