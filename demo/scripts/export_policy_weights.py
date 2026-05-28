#!/usr/bin/env python3
"""Export a loco_mjx URMA single-robot policy to a JS-loadable bundle.

For a given (robot, model_path) pair this writes:
  <out>/<robot>.weights.bin   — concatenated Float32 weight arrays
  <out>/<robot>.weights.json  — manifest: layer name, shape, byte offset; hyperparams

WeightNorm-wrapped Dense layers are pre-applied: kernel = scale[None,:] * kernel
/ column_norm[None,:].  After export the JS side just does plain matmul.

Run inside the rlx conda env, from the loco_mjx repo root:
  python shape_your_body_website/demo/scripts/export_policy_weights.py \
      --robot unitree_go2 \
      --model analysis/.viz_cache/esgjr7ec/latest.model \
      --out shape_your_body_website/demo/weights
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np


def restore_policy(model_path: Path):
    """Unzip the .model file and restore the raw policy params without a target,
    so we don't need to recreate the exact env shapes used during training.
    Returns (params_dict, stored_algo_config_dict).
    """
    import orbax.checkpoint as ocp

    ckpt_dir = os.path.abspath(model_path.parent)
    tmp_dir = f"{ckpt_dir}/tmp_export"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.unpack_archive(str(model_path), tmp_dir, "zip")
    try:
        with open(f"{tmp_dir}/config_algorithm.json") as f:
            stored_algo_cfg = json.load(f)

        ckpter = ocp.PyTreeCheckpointer()
        restored = ckpter.restore(tmp_dir)
        return restored["policy"]["params"]["params"], stored_algo_cfg
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def flatten_params(params, prefix=""):
    """Recursively flatten {dict-of-dicts} into [(path, np.ndarray)]."""
    out = []
    if isinstance(params, dict):
        for k in sorted(params.keys()):
            child = f"{prefix}.{k}" if prefix else k
            out.extend(flatten_params(params[k], child))
    else:
        out.append((prefix, np.asarray(params)))
    return out


def apply_weight_norm(flat):
    """Pre-apply Flax WeightNorm: replace kernel with scale[None,:]*kernel/||kernel||_axis.

    flax stores the scale as a single dict key with literal slashes, e.g.
        Dense_3.kernel                              (the raw kernel)
        WeightNorm_0.Dense_3/kernel/scale           (the per-output scale)
    """
    import re
    by_path = dict(flat)
    handled_scales = set()

    pat = re.compile(r"(Dense_\d+)/kernel/scale$")
    for path in list(by_path.keys()):
        m = pat.search(path)
        if not m:
            continue
        dense_name = m.group(1)
        kernel_path = f"{dense_name}.kernel"
        if kernel_path not in by_path:
            continue
        kernel = by_path[kernel_path]
        scale = by_path[path]
        if kernel.ndim != 2 or scale.shape != (kernel.shape[1],):
            continue
        col_norm = np.linalg.norm(kernel, axis=0)
        by_path[kernel_path] = (scale[None, :] * kernel) / (col_norm[None, :] + 1e-12)
        handled_scales.add(path)

    final = []
    for path, arr in by_path.items():
        if path in handled_scales:
            continue
        final.append((path, np.asarray(arr, dtype=np.float32)))
    return final


def write_bundle(flat, out_base: Path, hyper: dict):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_base.with_suffix(".weights.bin")
    json_path = out_base.with_suffix(".weights.json")

    blob = bytearray()
    layers = []
    for path, arr in flat:
        offset = len(blob)
        flat_arr = np.ascontiguousarray(arr.astype(np.float32))
        blob.extend(flat_arr.tobytes())
        layers.append({
            "name": path,
            "shape": list(flat_arr.shape),
            "offset": offset,
            "size_bytes": flat_arr.nbytes,
        })

    bin_path.write_bytes(bytes(blob))
    json_path.write_text(json.dumps({"layers": layers, "hyperparams": hyper}, indent=2))
    print(f"[export] wrote {bin_path.name} ({len(blob)/1024:.0f} KB) + {json_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output directory; bundle written as <out>/<robot>.weights.{bin,json}")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_base = args.out / args.robot

    params, stored_cfg = restore_policy(args.model)
    flat = flatten_params(params)
    flat = apply_weight_norm(flat)

    hyper = {
        "softmax_temperature": float(stored_cfg.get("softmax_temperature", 1.0)),
        "softmax_temperature_min": float(stored_cfg.get("softmax_temperature_min", 0.025)),
        "stability_epsilon": float(stored_cfg.get("stability_epsilon", 1e-8)),
        "policy_mean_abs_clip": float(stored_cfg.get("policy_mean_abs_clip", 10.0)),
    }
    write_bundle(flat, out_base, hyper)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
