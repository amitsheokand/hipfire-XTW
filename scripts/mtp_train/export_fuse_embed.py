"""Export Fuse-2 embed/lm_head tables (Q8F16) from .mq4 to f32 .npy.

Usage: mtp-torch export_fuse_embed.py <model.mq4> <out_dir>
Writes embed_tokens.f32.npy + lm_head.f32.npy [V, H] float32, and reports
whether the two tables are bit-identical (tied).
"""
import struct
import sys
import numpy as np


def parse_index(path):
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
        tensors.append((name, shape, qt, gs, off, ds))
        off += ds
    return f, tensors


def dequant_q8f16(path, off, rows, cols):
    # 34 B per 32 weights: f16 scale + 32 int8.
    import mmap as mmap_mod

    n_groups_row = cols // 32
    out = np.empty((rows, cols), dtype=np.float32)
    with open(path, "rb") as fh:
        mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
        raw = np.frombuffer(mm, dtype=np.uint8, count=rows * n_groups_row * 34, offset=off)
        raw = raw.reshape(rows, n_groups_row, 34)
        scales = raw[:, :, 0:2].view(np.float16).astype(np.float32)
        quants = raw[:, :, 2:].view(np.int8).astype(np.float32)
        out = quants.reshape(rows, cols) * np.repeat(scales, 32, axis=1).reshape(rows, cols)
        del raw, scales, quants
        mm.close()
    return out


def main():
    model, out_dir = sys.argv[1], sys.argv[2]
    import os

    os.makedirs(out_dir, exist_ok=True)
    f, tensors = parse_index(model)
    f.close()
    got = {}
    for name, shape, qt, gs, off, ds in tensors:
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            assert qt == 3, f"unexpected qt {qt} for {name}"
            rows, cols = shape
            print(f"dequant {name} {shape} ...", flush=True)
            got[name] = dequant_q8f16(model, off, rows, cols)
    e = got["model.embed_tokens.weight"]
    l = got["lm_head.weight"]
    print("tied (exact equal):", bool((e == l).all()))
    np.save(f"{out_dir}/embed_tokens.f32.npy", e)
    np.save(f"{out_dir}/lm_head.f32.npy", l)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
