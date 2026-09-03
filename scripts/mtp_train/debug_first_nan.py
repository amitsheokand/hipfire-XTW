"""Bisect first-NaN: replicate trainer loop with anomaly detection."""
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
E = torch.from_numpy(
    np.load("/tmp/hftest/mtp_tables/lm_head.f32.npy", mmap_mode="r")
).to(device, torch.float32)
cfg = build_head_config()
mtp = Qwen35MtpBlock(cfg).to(device=device, dtype=torch.float32)
opt = torch.optim.AdamW(mtp.parameters(), lr=1e-5, weight_decay=0.01, foreach=False)
for step in range(80):
    base = random.choice(train_files)
    hid = torch.from_numpy(np.load(base + ".hidden.npy")).to(device, torch.float32)
    pe = torch.from_numpy(np.load(base + ".prevemb.npy")).to(device, torch.float32)
    lab = torch.from_numpy(np.load(base + ".labels.npy")).to(device, torch.long)
    with torch.autograd.detect_anomaly():
        mh = mtp(pe.unsqueeze(0), hid.unsqueeze(0))[0]
        loss = chunked_ce_loss(mh, E, lab)
        opt.zero_grad()
        loss.backward()
    gbad = [n for n, p in mtp.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    opt.step()
    wbad = [n for n, p in mtp.named_parameters() if not torch.isfinite(p).all()]
    status = "OK" if loss.isfinite() and not gbad and not wbad else "BAD"
    print(f"{step} {os.path.basename(base)} loss={loss.item():.4f} {status} gbad={gbad[:2]} wbad={wbad[:2]}", flush=True)
    if status == "BAD":
        break
    del hid, pe, lab, mh, loss
