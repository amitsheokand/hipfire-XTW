"""Offline agreement of the PACKED .mtp sidecar (Q8 dequant) vs trunk labels."""
import struct
import sys

import numpy as np
import torch

# --- parse HFQM index ---
path = sys.argv[1]
f = open(path, "rb")
hdr = f.read(32)
moff, doff = struct.unpack("<QQ", hdr[16:32])
f.seek(moff)
meta = f.read(doff - moff)
depth = 0
instr = False
esc = False
end = 0
for i, b in enumerate(meta):
    if esc:
        esc = False
        continue
    if b == 92 and instr:
        esc = True
        continue
    if b == 34:
        instr = not instr
        continue
    if instr:
        continue
    if b == 123:
        depth += 1
    elif b == 125:
        depth -= 1
        if depth == 0:
            end = i + 1
            break
f.seek(moff + end)
(n,) = struct.unpack("<I", f.read(4))
tensors = []
off = doff
for _ in range(n):
    (nl,) = struct.unpack("<H", f.read(2))
    name = f.read(nl).decode()
    qt = struct.unpack("<B", f.read(1))[0]
    nd = struct.unpack("<B", f.read(1))[0]
    shape = struct.unpack("<" + "I" * nd, f.read(4 * nd))
    gs, ds = struct.unpack("<IQ", f.read(12))
    tensors.append((name, shape, qt, off, ds))
    off += ds
f.close()

import mmap as mmap_mod

fh = open(path, "rb")
mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
sd = {}
for name, shape, qt, off, ds in tensors:
    raw = np.frombuffer(mm, dtype=np.uint8, count=ds, offset=off)
    if len(shape) == 1:
        # norms stay F32
        sd[name] = torch.from_numpy(raw.view("<f4").reshape(shape).copy())
    else:
        rows, cols = shape
        ng = cols // 32
        r = raw.reshape(rows, ng, 34)
        sc = r[:, :, 0:2].view("<f2").astype(np.float32)
        q = r[:, :, 2:].view(np.int8).astype(np.float32)
        sd[name] = torch.from_numpy((q.reshape(rows, cols) * np.repeat(sc, 32, axis=1).reshape(rows, cols)).copy())
print("dequant ok:", {k: tuple(v.shape) for k, v in sd.items()})

sys.path.insert(0, "/home/amitsheokand/dev/hipfire/scripts/mtp_train")
from train_fuse_mtp import build_head_config
from mtp_module import Qwen35MtpBlock

cfg = build_head_config()
mtp = Qwen35MtpBlock(cfg).to(device="cuda:0", dtype=torch.float32)
# Sidecar names are already hipfire-canonical bare (shared_head_norm, wq,
# ...); the module uses the same bare names. Load directly.
# Sidecar names are hipfire-canonical (eh_proj, wq, ...); map to module keys.
HIPFIRE_TO_MODULE = {
    "shared_head_norm": "norm.weight",
    "enorm": "pre_fc_norm_embedding.weight",
    "hnorm": "pre_fc_norm_hidden.weight",
    "attn_norm": "layers.0.input_layernorm.weight",
    "attn_post_norm": "layers.0.post_attention_layernorm.weight",
    "attn_q_norm": "layers.0.self_attn.q_norm.weight",
    "attn_k_norm": "layers.0.self_attn.k_norm.weight",
    "eh_proj": "fc.weight",
    "wq": "layers.0.self_attn.q_proj.weight",
    "wk": "layers.0.self_attn.k_proj.weight",
    "wv": "layers.0.self_attn.v_proj.weight",
    "wo": "layers.0.self_attn.o_proj.weight",
    "ffn_gate": "layers.0.mlp.gate_proj.weight",
    "ffn_up": "layers.0.mlp.up_proj.weight",
    "ffn_down": "layers.0.mlp.down_proj.weight",
}
mapped = {HIPFIRE_TO_MODULE[k]: v.to("cuda:0", torch.float32) for k, v in sd.items()}
missing, unexpected = mtp.load_state_dict(mapped, strict=False)
print("load: missing=", missing, "unexpected=", unexpected)
E = torch.from_numpy(
    np.load("/tmp/hftest/mtp_tables/lm_head.f32.npy", mmap_mode="r")
).to("cuda:0", torch.float32)

import glob

tot_agree = tot = 0
for base in sorted(glob.glob("/tmp/hftest/corpus_full/*.hidden.npy"))[:0]:
    pass
# eval files (same holdout as trainer)
for name in ["lru_cache_pep8_strict", "humaneval_0_has_close_elements", "agentic_user_multistep", "tool_call_system"]:
    b = f"/tmp/hftest/corpus_full/{name}"
    import os

    if not os.path.exists(b + ".hidden.npy"):
        print("skip", name)
        continue
    hid = torch.from_numpy(np.load(b + ".hidden.npy")).to("cuda:0", torch.float32)
    pe = torch.from_numpy(np.load(b + ".prevemb.npy")).to("cuda:0", torch.float32)
    lab = torch.from_numpy(np.load(b + ".labels.npy")).to("cuda:0", torch.long)
    E = torch.from_numpy(np.load("/tmp/hftest/mtp_tables/lm_head.f32.npy", mmap_mode="r")).to("cuda:0", torch.float32)
    with torch.no_grad():
        mh = mtp(pe.unsqueeze(0), hid.unsqueeze(0))[0]
        pred = (mh @ E.T).float().argmax(-1)[:-1]
        a = int((pred == lab[:-1]).sum())
        tot_agree += a
        tot += lab.numel() - 1
        print(name, f"agree={100 * a / (lab.numel() - 1):.1f}%")
print(f"PACKED agree={100 * tot_agree / max(1, tot):.2f}%")

# --- Autoregressive (serve-like) agreement: k>=1 steps consume the head's
# own previous output as hidden (see mtp_head_forward_block_only callers),
# not trunk hiddens. K=3 chain per position; measure d1/d2/d3 agreement.
ar_tot = [0, 0, 0]
ar_n = [0, 0, 0]
for name in ["lru_cache_pep8_strict", "humaneval_0_has_close_elements", "tool_call_system"]:
    b = f"/tmp/hftest/corpus_full/{name}"
    import os

    if not os.path.exists(b + ".hidden.npy"):
        continue
    hid = torch.from_numpy(np.load(b + ".hidden.npy")).to("cuda:0", torch.float32)
    pe = torch.from_numpy(np.load(b + ".prevemb.npy")).to("cuda:0", torch.float32)
    lab = torch.from_numpy(np.load(b + ".labels.npy")).to("cuda:0", torch.long)
    T = hid.shape[0]
    with torch.no_grad():
        for t in range(0, T - 4):
            h_in = hid[t]
            e_in = pe[t]
            for k in range(3):
                mh = mtp(e_in.unsqueeze(0).unsqueeze(0), h_in.unsqueeze(0).unsqueeze(0))[0, 0]
                pred = (mh @ E.T).float().argmax(-1).item()
                if pred == lab[t + 1 + k].item():
                    ar_tot[k] += 1
                ar_n[k] += 1
                if k < 2:
                    # next step consumes own output as hidden + pred emb.
                    # emb(pred) via E rows (tied table).
                    h_in = mh
                    e_in = E[pred]
print("AR d1/d2/d3 agree:", [f"{100 * a / max(1, n):.1f}%" for a, n in zip(ar_tot, ar_n)])
