#!/usr/bin/env python3
"""Stream PEFT LoRA + modules_to_save onto Whittle v2 BF16 shards.

Does not load the 53 GB trunk into RAM. Adapter (~2.8 GB) stays mapped;
each v2 shard is rewritten independently to v2.1-merged/.

W <- W + (lora_alpha / r) * (B @ A)  with fan_in_fan_out=false.
modules_to_save tensors replace the base tensor outright.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

SCALE = 256.0 / 128.0  # lora_alpha / r
ADAPTER_PREFIX = "base_model.model."


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        return header, 8 + n


def write_safetensors(path: Path, tensors: list[tuple[str, dict, bytes]]) -> None:
    header: dict = {}
    payload = bytearray()
    for name, meta, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [start, start + len(data)],
        }
        if "data_offsets" in meta:
            pass
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(raw) % 8)) % 8
    raw = raw + (b" " * pad)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def tensor_bytes(mm: memoryview, header: dict, data_start: int, name: str) -> memoryview:
    meta = header[name]
    s, e = meta["data_offsets"]
    return mm[data_start + s : data_start + e]


def base_name(adapter_key: str) -> str:
    s = adapter_key
    if s.startswith(ADAPTER_PREFIX):
        s = s[len(ADAPTER_PREFIX) :]
    return s.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")


def load_adapter(path: Path):
    header, data_start = read_header(path)
    header.pop("__metadata__", None)
    mm = memoryview(path.read_bytes())
    replace: dict[str, tuple[str, list[int], bytes]] = {}
    lora: dict[str, dict[str, np.ndarray]] = {}
    for name, meta in header.items():
        raw = bytes(tensor_bytes(mm, header, data_start, name))
        bname = base_name(name)
        if "lora_A" in name:
            arr = np.frombuffer(raw, dtype="<f4").reshape(meta["shape"]).copy()
            lora.setdefault(bname, {})["A"] = arr
        elif "lora_B" in name:
            arr = np.frombuffer(raw, dtype="<f4").reshape(meta["shape"]).copy()
            lora.setdefault(bname, {})["B"] = arr
        else:
            replace[bname] = (meta["dtype"], list(meta["shape"]), raw)
    missing = [k for k, v in lora.items() if "A" not in v or "B" not in v]
    if missing:
        raise SystemExit(f"incomplete LoRA pairs: {missing[:8]}")
    del mm
    print(
        f"adapter: {len(lora)} LoRA pairs, {len(replace)} replacements, "
        f"scale={SCALE}",
        flush=True,
    )
    return replace, lora


def merge_shard(
    src: Path,
    dest: Path,
    replace: dict,
    lora: dict,
) -> tuple[int, int]:
    header, data_start = read_header(src)
    meta_blob = header.pop("__metadata__", None)
    names = [n for n in header if n != "__metadata__"]
    mm = memoryview(src.read_bytes())
    out: list[tuple[str, dict, bytes]] = []
    n_lora = 0
    n_replace = 0
    for name in names:
        meta = header[name]
        raw = bytes(tensor_bytes(mm, header, data_start, name))
        if name in replace:
            dtype, shape, repl = replace[name]
            if list(shape) != list(meta["shape"]):
                raise SystemExit(
                    f"replacement shape mismatch {name}: {shape} vs {meta['shape']}"
                )
            out.append((name, {**meta, "dtype": dtype, "shape": shape}, repl))
            n_replace += 1
            continue
        if name in lora:
            if meta["dtype"] != "BF16":
                raise SystemExit(f"LoRA target {name} is {meta['dtype']}, expected BF16")
            pair = lora[name]
            a, b = pair["A"], pair["B"]
            m, k = meta["shape"]
            if a.shape != (a.shape[0], k) or b.shape != (m, a.shape[0]):
                raise SystemExit(
                    f"LoRA shape mismatch {name}: W={meta['shape']} A={a.shape} B={b.shape}"
                )
            w = bf16_to_f32(np.frombuffer(raw, dtype="<u2").reshape(m, k))
            w = w + (SCALE * (b @ a))
            merged = f32_to_bf16(w).tobytes()
            out.append((name, meta, merged))
            n_lora += 1
            continue
        out.append((name, meta, raw))
    del mm
    if meta_blob is not None:
        # metadata is not a tensor; HuggingFace writers omit it on rewrite.
        pass
    write_safetensors(dest, out)
    return n_lora, n_replace


def copy_sidecars(src_dir: Path, dest_dir: Path) -> None:
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ):
        s = src_dir / name
        if s.exists():
            shutil.copy2(s, dest_dir / name)
            print(f"copied {name}", flush=True)


def main() -> int:
    root = Path.home() / ".hipfire/hf-cache/Qwen3.8-Whittle-MoE-27B-A17.8B"
    src_dir = root / "v2"
    adapter = root / "v2.1/adapter/adapter_model.safetensors"
    dest_dir = root / "v2.1-merged"
    dest_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    copy_sidecars(src_dir, dest_dir)

    t0 = time.time()
    replace, lora = load_adapter(adapter)
    total_lora = 0
    total_repl = 0
    for i, shard in enumerate(shards, 1):
        src = src_dir / shard
        dest = dest_dir / shard
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            print(f"[{i}/{len(shards)}] skip existing {shard}", flush=True)
            continue
        print(f"[{i}/{len(shards)}] {shard} ({src.stat().st_size / 1e9:.2f} GB)", flush=True)
        n_l, n_r = merge_shard(src, dest, replace, lora)
        total_lora += n_l
        total_repl += n_r
        print(
            f"  wrote {dest.stat().st_size / 1e9:.2f} GB  "
            f"lora={n_l} replace={n_r}  elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    print(
        f"done: {dest_dir}  tensors with LoRA={total_lora} replacements={total_repl}  "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    applied = set()
    for shard in shards:
        header, _ = read_header(src_dir / shard)
        applied.update(n for n in header if n in lora or n in replace)
    missing_lora = sorted(set(lora) - applied)
    missing_repl = sorted(set(replace) - applied)
    if missing_lora or missing_repl:
        print(f"ERROR unused LoRA {len(missing_lora)} replace {len(missing_repl)}", flush=True)
        print(missing_lora[:8], missing_repl[:8], flush=True)
        return 1
    print("all adapter tensors applied", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
