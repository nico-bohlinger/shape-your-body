#!/usr/bin/env python3
"""Export per-design MJCFs for the Shape Your Body interactive demo.

For a (setting, robot, seed) triple this script writes:
  <out>/reference.xml      — nominal URDF (factor = 0)
  <out>/initial.xml        — perturbed start (the f_init the cluster co-design started from)
  <out>/optimized.xml      — final f* the VGDS search converged to
  <out>/iter_00.xml ... iter_50.xml — intermediate VGDS particles

The XMLs are MuJoCo's compiled-and-saved form, so they're self-contained
(no <include> chains). Asset references (meshes, hfields) stay as plain
filenames pointing at the existing demo/scenes/<robot>/assets/ folder, so
copy this script's output into demo/scenes/<robot>/ and the assets resolve
automatically.

Run from the loco_mjx repo root, inside the rlx conda env:
  python shape_your_body_website/demo/scripts/dump_design_xmls.py \
      --setting e4 --robot unitree_go2 --seed 0 \
      --out shape_your_body_website/demo/scenes/unitree_go2

Prereqs (already populated for Bohlinger's local cache):
  analysis/.viz_cache/eval/<group>.json    — optimised particle
  analysis/.viz_cache/trajectory/<group>.npz — 51-step trajectory (--include-iters)
  analysis/.viz_cache/<run_id>/curriculum_coeff.txt — for f_init reconstruction

If a cached artifact is missing, run analysis/viz_policy.py once with the same
(setting, robot, seed) first; that populates everything we need.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SHORT = {"unitree_go2": "go2", "mit_humanoid": "mit", "golem": "golem"}

BASE_RUN_BY_SETTING = {
    "e1": {
        "unitree_go2":  "esgjr7ec",
        "mit_humanoid": "k1r2zj4b",
        "golem":        "z5o4aztg",
    },
    "e2": {
        "unitree_go2":  "irrigc68",
        "mit_humanoid": "07v9gtba",
        "golem":        "j5ikzoxl",
    },
}
ALL50_BASE_RUN = "9huz0fww"


def opt_group_for(setting: str, robot: str, seed: int) -> str:
    short = SHORT.get(robot)
    if setting == "e1":
        return f"E50__e1_{short}__lam100_in0__s{seed}"
    if setting == "e2":
        lam = 100 if short == "mit" else 20
        return f"E66_e2_pert__{short}__particle__lam{lam}__s{seed}"
    if setting == "e4":
        if robot == "unitree_go2":
            return f"E69_e4_pert__go2__particle__s{seed}"
        return f"E75_e4_all50__{robot}__particle__lam100__s{seed}"
    raise ValueError(f"unknown setting {setting!r}")


def base_run_id(setting: str, robot: str) -> str:
    if setting == "e4":
        return ALL50_BASE_RUN
    return BASE_RUN_BY_SETTING[setting][robot]


def make_init_particle(dim: int, seed: int, c: float) -> np.ndarray:
    """Bit-identical to loco_mjx/algorithms/optimization_baselines/surrogate.py."""
    if seed < 0:
        return np.zeros(int(dim), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(low=-float(c), high=float(c), size=int(dim)).astype(np.float32)


def load_cached(cache: Path, setting: str, robot: str, seed: int):
    group = opt_group_for(setting, robot, seed)
    eval_path = cache / "eval" / f"{group}.json"
    if not eval_path.exists():
        sys.exit(
            f"[dump_design_xmls] missing {eval_path}\n"
            f"  run analysis/viz_policy.py --setting {setting} --robot {robot} --seed {seed}\n"
            f"  first to populate the cache."
        )
    f_opt = np.asarray(json.loads(eval_path.read_text())["particle_normalized"], dtype=np.float32)

    coeff_path = cache / base_run_id(setting, robot) / "curriculum_coeff.txt"
    coeff = float(coeff_path.read_text().strip()) if coeff_path.exists() else 1.0

    traj_path = cache / "trajectory" / f"{group}.npz"
    traj = None
    if traj_path.exists():
        with np.load(traj_path) as data:
            for key in ("particles", "particle_trajectory", "factors", "trajectory"):
                if key in data.files:
                    traj = np.asarray(data[key], dtype=np.float32)
                    break
        if traj is None:
            print(f"[dump_design_xmls] {traj_path.name}: no recognized key. Got {data.files}.",
                  file=sys.stderr)
    return f_opt, coeff, traj


def build_env(robot: str):
    """Instantiate the single_robot_urma mujoco env exactly the way viz_policy does."""
    import loco_mjx.environments.locomotion.single_robot_urma.mujoco  # noqa: F401
    from ml_collections import config_dict
    from rl_x.environments.environment_manager import (
        get_environment_config, get_environment_create_train_and_eval_env,
    )
    from rl_x.runner.default_config import get_config as get_runner_config

    env_name = "locomotion.single_robot_urma.mujoco"
    cfg = config_dict.ConfigDict()
    cfg.runner = get_runner_config("test")
    cfg.runner.mode = "test"
    cfg.environment = get_environment_config(env_name)
    cfg.environment.train_robots = (robot,)
    cfg.environment.nr_envs = 1
    cfg.environment.render = False
    cfg.environment.add_goal_arrow = False
    cfg.environment.domain_randomization.initial_state.type = "default"
    cfg.environment.domain_randomization.sampling_type = "none"
    cfg.environment.domain_randomization.perturbation.sampling_type = "none"
    cfg.environment.domain_randomization.observation_noise.type = "none"

    create_envs = get_environment_create_train_and_eval_env(env_name)
    train_env, _eval_env = create_envs(cfg)
    return train_env


HASH_SUFFIX_RE = None


def _strip_mesh_hashes(xml_text: str) -> str:
    """mj_saveLastXML appends a content hash to mesh/texture file refs, e.g.
       base_0-045367d…6cc.obj.  Strip the hash so refs match our flat assets/ dir.
       Also re-introduce assetdir="assets" on <compiler> so refs resolve.
    """
    import re
    global HASH_SUFFIX_RE
    if HASH_SUFFIX_RE is None:
        HASH_SUFFIX_RE = re.compile(r'(file=")([^"/]+?)-[0-9a-f]{40}(\.[A-Za-z0-9]+)"')
    out = HASH_SUFFIX_RE.sub(r'\1\2\3"', xml_text)
    def patch_compiler(m):
        head, tail = m.group(1), m.group(2)
        if 'assetdir' not in head:
            head += ' assetdir="assets"'
        if 'balanceinertia' not in head:
            head += ' balanceinertia="true"'
        if 'inertiafromgeom' not in head:
            head += ' inertiafromgeom="false"'
        return head + tail
    out = re.sub(r'(<compiler\b[^>]*?)(/?>)', patch_compiler, out, count=1)
    return out


def save_xml(env, out_path: Path):
    import mujoco
    import tempfile
    target = env.envs[0] if hasattr(env, "envs") else env
    model = target.internal_state["mj_model"]
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        mujoco.mj_saveLastXML(str(tmp_path), model)
        xml = tmp_path.read_text()
    finally:
        tmp_path.unlink(missing_ok=True)
    xml = _strip_mesh_hashes(xml)
    out_path.write_text(xml)


def apply_factor(env, factor):
    target = env.envs[0] if hasattr(env, "envs") else env
    target.set_factor_params(np.asarray(factor, dtype=np.float64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["e1", "e2", "e4"], required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="seeds to dump (space-separated, e.g. --seeds 0 1 2 ... 9)")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory (e.g. shape_your_body_website/demo/scenes/unitree_go2)")
    p.add_argument("--cache-dir", type=Path, default=Path("analysis/.viz_cache"))
    p.add_argument("--with-reference", action="store_true",
                   help="also write reference.xml (factor=0).  Independent of seed.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = build_env(args.robot)
    target = env.envs[0] if hasattr(env, "envs") else env
    fd = target.factor_dim
    dim = int(fd() if callable(fd) else fd)

    if args.with_reference:
        apply_factor(env, np.zeros(dim, dtype=np.float32))
        save_xml(env, args.out / "reference.xml")
        print(f"[dump_design_xmls] wrote {args.out / 'reference.xml'}")

    for seed in args.seeds:
        f_opt, coeff, traj = load_cached(args.cache_dir, args.setting, args.robot, seed)
        if traj is None:
            print(f"[dump_design_xmls] skipping seed {seed}: no trajectory NPZ", file=sys.stderr)
            continue
        if traj.shape[1] != dim:
            sys.exit(f"trajectory dim mismatch: traj={traj.shape[1]} env={dim}")
        prefix = f"s{seed}_iter_"
        for i, factor in enumerate(traj):
            apply_factor(env, factor)
            save_xml(env, args.out / f"{prefix}{i:02d}.xml")
        print(f"[dump_design_xmls] wrote {traj.shape[0]} {prefix}*.xml files")


if __name__ == "__main__":
    main()
