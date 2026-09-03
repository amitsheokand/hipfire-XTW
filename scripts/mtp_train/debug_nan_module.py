"""Localize first-NaN submodule: train 23 steps (seed-42 order), then
forward wiki_0411 with per-module finiteness hooks."""
import glob
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fuse_mtp import build_head_config, chunked_ce_loss  # noqa: E402
from mtp_module import Qwen35MtpBlock  # noqa: E402

random.seed(42)
torch.manual_seed(42)
device = "cuda:0"
allbases = sorted({p[: -len(".hidden.npy")] for p in glob.glob("/tmp/hftest/corpus_full/*.hidden.npy")})
holdout = set("lru_cache_pep8_strict,humaneval_0_has_close_elements,agentic_user_multistep,trains-meet,tool_call_system".split(","))
train_files = [b for b in allbases if os.path.basename(b) not in holdout]
E = torch.from_numpy(np.load("/tmp/hftest/mtp_tables/lm_head.f32.npy", mmap_mode="r")).to(device, torch.float32)
cfg = build_head_config()
mtp = Qwen35MtpBlock(cfg).to(device=device, dtype=torch.float32)
with torch.no_grad():
    mtp.fc.weight.zero_()
opt = torch.optim.AdamW(mtp.parameters(), lr=3e-6, weight_decay=0.01, foreach=False)
for step in range(23):
    base = random.choice(train_files)
    hid = torch.from_numpy(np.load(base + ".hidden.npy")).to(device, torch.float32)
    pe = torch.from_numpy(np.load(base + ".prevemb.npy")).to(device, torch.float32)
    lab = torch.from_numpy(np.load(base + ".labels.npy")).to(device, torch.long)
    mh = mtp(pe.unsqueeze(0), hid.unsqueeze(0))[0]
    loss = chunked_ce_loss(mh, E, lab)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mtp.parameters(), max_norm=50.0, foreach=False)
    opt.step()
    del hid, pe, lab, mh, loss
print("trained 23 steps, now probing wiki_0411 per-module", flush=True)

bad = []


def hook(name):
    def fn(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if isinstance(o, torch.Tensor) and not torch.isfinite(o).all().item():
            bad.append(name)

    return fn


for n, m in mtp.named_modules():
    m.register_forward_hook(hook(n or "root"))
hid = torch.from_numpy(np.load("/tmp/hftest/corpus_full/wiki_0411.hidden.npy")).to(device, torch.float32)
pe = torch.from_numpy(np.load("/tmp/hftest/corpus_full/wiki_0411.prevemb.npy")).to(device, torch.float32)
with torch.no_grad():
    mtp(pe.unsqueeze(0), hid.unsqueeze(0))
print("non-finite modules:", bad if bad else "NONE (all finite?!)", flush=True)
