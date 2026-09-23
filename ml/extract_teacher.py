#!/usr/bin/env python3
"""Extract Qwen-Drive-1.0 teacher features for the student's distillation loss.

For a subsample of frames in each episode, run the teacher's vision tower on the front
frame and pool its output into one vector. Writes feats.npz keyed "episode:frame".

The teacher is a custom architecture (QwenDriveForPlanning); loading it is best-effort.
If it cannot be loaded, this falls back to a base Qwen3-VL model's vision features, and
if that fails too it writes an empty file and exits 0, so the student still trains with
behaviour cloning alone. Run on a GPU node (Turing: fp16).

    python3 ml/extract_teacher.py --data ~/lerobot/atlascar_sim --out runs/teacher --every 5
"""
import argparse, json, os, time, traceback
import numpy as np, cv2, torch

MODEL = 'Qwen/Qwen-Drive-1.0-4B'
FALLBACK = 'Qwen/Qwen3-VL-2B-Instruct'


def load_teacher(name, dtype):
    from transformers import AutoModel, AutoProcessor, AutoConfig
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=dtype, device_map='auto')
    model.eval()
    return model, proc, cfg


def find_vision_tower(model):
    for path in ('visual', 'model.visual', 'vision_tower', 'model.vision_tower', 'vlm.visual', 'vlm.model.visual'):
        obj = model
        try:
            for part in path.split('.'): obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def vision_features(vt, proc, img_rgb, dev, dtype):
    from PIL import Image
    im = Image.fromarray(img_rgb)
    try:
        inputs = proc.image_processor(images=im, return_tensors='pt')
    except Exception:
        inputs = proc(images=im, text='<|vision_start|><|image_pad|><|vision_end|>', return_tensors='pt')
    pv = inputs['pixel_values'].to(dev, dtype)
    grid = inputs.get('image_grid_thw')
    with torch.no_grad():
        out = vt(pv, grid_thw=grid.to(dev)) if grid is not None else vt(pv)
    if hasattr(out, 'last_hidden_state'): out = out.last_hidden_state
    if isinstance(out, (tuple, list)): out = out[0]
    return out.float().reshape(-1, out.shape[-1]).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='runs/teacher')
    ap.add_argument('--every', type=int, default=5); ap.add_argument('--model', default=MODEL)
    ap.add_argument('--max-episodes', type=int, default=0)
    a = ap.parse_args()
    root = os.path.expanduser(a.data); os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if dev == 'cuda' else torch.float32
    model = proc = vt = None; used = None
    for name in (a.model, FALLBACK):
        try:
            t0 = time.time(); model, proc, cfg = load_teacher(name, dtype)
            vt = find_vision_tower(model)
            if vt is None: raise RuntimeError('no vision tower found on ' + name)
            used = name
            print(f'loaded {name} ({getattr(cfg, "architectures", None)}) in {time.time()-t0:.0f}s', flush=True); break
        except Exception as e:
            print(f'could not use {name}: {type(e).__name__}: {str(e)[:300]}', flush=True)
            traceback.print_exc(limit=2); model = None
    if model is None:
        np.savez_compressed(os.path.join(a.out, 'feats.npz'))
        json.dump({'teacher': None, 'n_feats': 0}, open(os.path.join(a.out, 'meta.json'), 'w'))
        print('no teacher available; wrote an empty feats.npz (student trains BC-only)'); return
    info = json.load(open(os.path.join(root, 'meta/info.json')))
    n_ep = info['total_episodes'] if not a.max_episodes else min(a.max_episodes, info['total_episodes'])
    feats = {}; t0 = time.time(); nfail = 0
    for ei in range(n_ep):
        cap = cv2.VideoCapture(os.path.join(root, 'videos/chunk-000/observation.images.front', f'episode_{ei:06d}.mp4'))
        k = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if k % a.every == 0:
                try:
                    feats[f'{ei}:{k}'] = vision_features(vt, proc, f[:, :, ::-1].copy(), dev, dtype).astype(np.float32)
                except Exception as e:
                    nfail += 1
                    if nfail <= 3: print(f'feature failure ep{ei} k{k}: {type(e).__name__}: {str(e)[:200]}', flush=True)
                    if nfail > 50 and not feats: break
            k += 1
        cap.release()
        if ei % 20 == 0: print(f'  ep {ei}/{n_ep}: {len(feats)} feats, {time.time()-t0:.0f}s', flush=True)
        if nfail > 50 and not feats: print('giving up on features'); break
    np.savez_compressed(os.path.join(a.out, 'feats.npz'), **feats)
    json.dump({'teacher': used, 'n_feats': len(feats), 'failures': nfail,
               'dim': int(next(iter(feats.values())).shape[-1]) if feats else 0},
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(f'wrote {len(feats)} features ({nfail} failures) from {used} -> {a.out}/feats.npz')


if __name__ == '__main__':
    main()
