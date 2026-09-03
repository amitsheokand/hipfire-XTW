"""Train a Fuse-2 MTP head from exported trunk corpus (no HF trunk needed).

Data: <corpus>/*.hidden.npy + *.prevemb.npy + *.labels.npy + *.ids.npy
  (see saddle-lab fuse_export_mtp_corpus), plus <tables>/lm_head.f32.npy
  (tied embed/head, see export_fuse_embed.py).
Head: Qwen35MtpBlock (scripts/mtp_train/mtp_module.py) with Fuse dims
  (H=2560, 16 heads / 4 kv / head_dim 256, mlp 8192, rope_theta 1e7 to match
  the hipfire trunk override, partial-rotary 0.25 -> n_rot 64).
Loss: CE(mtp_logits[t], labels[t]) — self-distill vs the MQ4 trunk argmax.
Output: <out_dir>/ with model.safetensors (mtp.* keys) + config.json,
  ready for `mtp_extract --hf-dir <out_dir> --output fuse-2-moe.mtp`.

Usage: mtp-torch train_fuse_mtp.py --corpus <dir> --tables <dir>
         --out <dir> [--steps N] [--eval-every N] [--lr F]
"""
import argparse
import glob
import json
import os

# Fragmentation guard for the 34 GB R9700 (production daemon resident):
# cap cached splits. NOTE: expandable_segments is DELIBERATELY off — on
# ROCm/gfx1201 it grows the pool unboundedly across varying sequence
# lengths (~350 MB/step to OOM); length-bucketing below keeps shapes few
# so the classic allocator reuses. And NEVER torch.cuda.empty_cache():
# on gfx1201 it deterministically corrupts subsequent math (NaN losses).
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_module import Qwen35MtpBlock  # noqa: E402

H = 2560
N_HEADS = 16
N_KV = 4
HEAD_DIM = 256
N_FF = 8192
VOCAB = 248320
ROPE_THETA = 10_000_000.0
RMS_EPS = 1e-6


def build_head_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        hidden_size=H,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        intermediate_size=N_FF,
        num_hidden_layers=1,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        partial_rotary_factor=0.25,
        vocab_size=VOCAB,
        tie_word_embeddings=True,
    )
    return cfg


# mtp_extract naming map (must match exactly).
EXPECTED_KEYS = [
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.fc.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]


def check_keys(mtp):
    # Module keys omit the mtp. prefix (load_pretrained_ strips it on load,
    # we add it back on save); compare prefixed.
    have = {"mtp." + k for k in mtp.state_dict().keys()}
    missing = [k for k in EXPECTED_KEYS if k not in have]
    extra = sorted(k for k in have if k not in EXPECTED_KEYS and "rotary" not in k)
    print(f"  key check: missing={missing} extra_nonrotary={extra}")
    return not missing


@torch.no_grad()
def eval_agreement(mtp, E, eval_files, device, chunk=128):
    tot_agree = tot = 0
    for base in eval_files:
        hid = torch.from_numpy(np.load(base + ".hidden.npy")).to(device, torch.float32)
        pe = torch.from_numpy(np.load(base + ".prevemb.npy")).to(device, torch.float32)
        lab = torch.from_numpy(np.load(base + ".labels.npy")).to(device, torch.long)
        mh = mtp(pe.unsqueeze(0), hid.unsqueeze(0))[0]
        for s in range(0, mh.shape[0] - 1, chunk):
            e = min(s + chunk, mh.shape[0] - 1)
            pred = (mh[s:e] @ E.T).float().argmax(-1)
            tot_agree += int((pred == lab[s:e]).sum())
            tot += e - s
        del hid, pe, lab, mh
    # NOTE: no torch.cuda.empty_cache() here (or in the train loop): on ROCm
    # gfx1201 it deterministically corrupts subsequent math (NaN losses a
    # few steps later). expandable_segments + chunking keeps memory flat.
    return tot_agree / max(1, tot)


def check_finite(tag, step, base, **tensors):
    for name, t in tensors.items():
        if t is not None and isinstance(t, torch.Tensor) and not torch.isfinite(t).all().item():
            print(f"NAN-HUNT step={step} prompt={os.path.basename(base)} stage={tag} tensor={name}", flush=True)
            return False
    return True


def chunked_ce_loss(mh, E, lab, chunk=64, z_weight=1e-4):
    # CE over positions in chunks: avoids materializing [T, V] float.
    # mh: [T, H] bf16 (requires grad); E: [V, H] bf16.
    # From-scratch heads emit large-magnitude logits (BF16 inf); z-loss
    # keeps them bounded (standard from-scratch LM practice) + nan_to_num
    # safety for the first steps.
    total, ztot, count = 0.0, 0.0, 0
    T = mh.shape[0] - 1
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        # E.T is a view (no copy); matmul stays BF16, convert only the
        # small chunk result — never materialize E as F32 (2.5 GB).
        logits = (mh[s:e] @ E.T).float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        total = total + F.cross_entropy(logits, lab[s:e]).float() * (e - s)
        lse = torch.logsumexp(logits, dim=-1)
        ztot = ztot + (lse * lse).sum()
        count += e - s
    return total / max(1, count) + z_weight * ztot / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--spike-skip", type=float, default=3.0)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--clip-norm", type=float, default=50.0)
    ap.add_argument("--constant-lr", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", default="lru_cache_pep8_strict,humaneval_0_has_close_elements,agentic_user_multistep,trains-meet,tool_call_system")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0"
    t0 = time.time()

    # File split: hipfire-prompt bases with holdout names -> eval, rest train.
    # Length buckets: varying T fragments the caching allocator on ROCm
    # (~350 MB/step to OOM); a few shapes reuse cleanly. Bucket by capped T.
    holdout = set(args.holdout.split(","))
    allbases = sorted({p[: -len(".hidden.npy")] for p in glob.glob(f"{args.corpus}/*.hidden.npy")})
    eval_files = [b for b in allbases if os.path.basename(b) in holdout]
    buckets: dict = {}
    for b in allbases:
        if os.path.basename(b) in holdout:
            continue
        t = int(np.load(b + ".ids.npy").shape[0])
        key = 0 if t <= 256 else (1 if t <= 512 else 2)
        buckets.setdefault(key, []).append(b)
    buckets = {k: v for k, v in buckets.items() if v}
    print(f"train buckets: { {k: len(v) for k, v in buckets.items()} }, eval files: {len(eval_files)}")
    assert buckets and eval_files, "empty split"

    print("Load lm_head table (mmap, FP32: matches head precision)...")
    E = torch.from_numpy(
        np.load(f"{args.tables}/lm_head.f32.npy", mmap_mode="r")
    ).to(device, torch.float32)
    print(f"  E: {tuple(E.shape)} {E.dtype}")

    print("Build Fuse MTP head (FP32: BF16 attention softmax overflows at")
    print("random init -> NaN factory; 450 MB head fits fine)...")
    cfg = build_head_config()
    mtp = Qwen35MtpBlock(cfg).to(device=device, dtype=torch.float32)
    # EAGLE-standard init: fc to ZERO so early activations (and grads) stay
    # small; the head grows out of uniform gradually instead of exploding.
    # PLUS residual-scale init (GPT-2 style): attention-out + mlp-down at
    # 0.1x so trunk outlier channels (hidden max ~60) cannot overflow the
    # FP32 softmax through random weights in the first steps.
    with torch.no_grad():
        mtp.fc.weight.zero_()
        for name, mod in mtp.named_modules():
            if isinstance(mod, torch.nn.Linear) and any(
                s in name for s in ("self_attn.o_proj", "mlp.down_proj")
            ):
                mod.weight.mul_(0.1)
    n_train = sum(p.numel() for p in mtp.parameters())
    print(f"  trainable: {n_train:,} params")
    if not check_keys(mtp):
        print("KEY MISMATCH vs mtp_extract map — abort before training")
        sys.exit(2)

    # foreach=False: ROCm foreach Adam kernels are suspect on gfx1201
    # (NaN updates + ILLEGAL_INSTRUCTION crashes observed with foreach).
    opt = torch.optim.AdamW(mtp.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    os.makedirs(args.out, exist_ok=True)

    def lr_lambda(step):
        if args.constant_lr:
            return 1.0
        warmup = args.warmup
        if step < warmup:
            return step / warmup
        import math

        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print("Eval BEFORE training...")
    a0 = eval_agreement(mtp, E, eval_files, device)
    print(f"  baseline agree={100 * a0:.2f}%")
    hist = [(0, a0)]
    losses = []
    for step in range(args.steps):
        base = random.choice(random.choice(list(buckets.values())))
        hid = torch.from_numpy(np.load(base + ".hidden.npy")).to(device, torch.float32)
        pe = torch.from_numpy(np.load(base + ".prevemb.npy")).to(device, torch.float32)
        lab = torch.from_numpy(np.load(base + ".labels.npy")).to(device, torch.long)
        ts = time.time()
        mh = mtp(pe.unsqueeze(0), hid.unsqueeze(0))[0]
        if not check_finite("fwd", step, base, mh=mh):
            break
        loss = chunked_ce_loss(mh, E, lab)
        if not check_finite("loss", step, base, loss=loss):
            break
        opt.zero_grad()
        loss.backward()
        if not args.no_clip:
            # foreach=False everywhere on gfx1201 (foreach collective kernels
            # are implicated in NaN updates + ILLEGAL_INSTRUCTION crashes).
            # Norm must match the natural scale (~36 early): 1.0 freezes
            # learning 36x; 50 clips only true spikes.
            gn = torch.nn.utils.clip_grad_norm_(mtp.parameters(), max_norm=args.clip_norm, foreach=False)
        else:
            gn = torch.tensor(float("nan"))
        # Loss-spike skip: outlier prompts (trunk hidden max ~60) can blow
        # past the update; skip the step but keep the sample for later.
        spike = False
        if losses:
            import statistics

            med = statistics.median(losses[-100:])
            if loss.item() > args.spike_skip * med:
                spike = True
                print(f"SPIKE-SKIP step={step} loss={loss.item():.2f} med={med:.2f}", flush=True)
        if not spike:
            opt.step()
        sched.step()
        torch.cuda.synchronize()
        dt = (time.time() - ts) * 1000
        losses.append(loss.item())
        del mh
        if step % args.eval_every == 0 or step == args.steps - 1:
            ae = eval_agreement(mtp, E, eval_files, device)
            hist.append((step + 1, ae))
            print(
                f"{step:<5} lr={sched.get_last_lr()[0]:.2e} loss={losses[-1]:.4f} "
                f"T={hid.shape[0]:<4} {dt:<7.1f}ms agree={100 * ae:.2f}% ({100 * (ae - a0):+.2f})",
                flush=True,
            )
            # Periodic checkpoint: crash-safe, keep best by agreement.
            ck = {f"mtp.{k}": v.detach().cpu().float() for k, v in mtp.state_dict().items()}
            save_file(ck, f"{args.out}/model_step{step + 1}.safetensors")
            best = max(hist, key=lambda h: h[1])
            if (step + 1, ae) == best:
                save_file(ck, f"{args.out}/model_best.safetensors")
        elif step % 10 == 0:
            print(
                f"{step:<5} lr={sched.get_last_lr()[0]:.2e} loss={losses[-1]:.4f} "
                f"gn={float(gn):.2e} T={hid.shape[0]:<4} {dt:<7.1f}ms",
                flush=True,
            )
    af = eval_agreement(mtp, E, eval_files, device)
    print(f"final agree={100 * af:.2f}% (delta {100 * (af - a0):+.2f}%)")
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")

    os.makedirs(args.out, exist_ok=True)
    sd = {f"mtp.{k}": v.detach().cpu().float() for k, v in mtp.state_dict().items()}
    save_file(sd, f"{args.out}/model.safetensors")
    with open(f"{args.out}/config.json", "w") as fh:
        json.dump(
            {
                "architectures": ["Qwen3_5ForCausalLM"],
                "hidden_size": H,
                "num_attention_heads": N_HEADS,
                "num_key_value_heads": N_KV,
                "head_dim": HEAD_DIM,
                "intermediate_size": N_FF,
                "num_hidden_layers": 1,
                "rms_norm_eps": RMS_EPS,
                "rope_theta": ROPE_THETA,
                "partial_rotary_factor": 0.25,
                "vocab_size": VOCAB,
                "tie_word_embeddings": True,
                "_name_or_path": "hipfire-fuse-2-moe-mtp",
            },
            fh,
            indent=2,
        )
    json.dump(
        {"baseline_agree": a0, "final_agree": af, "losses": losses, "eval_history": hist},
        open(f"{args.out}/train_metrics.json", "w"),
        indent=1,
    )
    print(f"wrote {args.out}/model.safetensors + config.json")


if __name__ == "__main__":
    main()
