#!/usr/bin/env python3
"""Dump per-design metadata used by the in-browser URMA policy.

For each (robot, design) tuple we instantiate the env, apply the design factor,
and write a JSON file containing everything the JS observation builder needs:

  joint_descriptions  : [n_joints][30]  static per-design
  general_static      : { gains_and_scale_normalized: [3],
                           mass_normalized: [1],
                           robot_dimensions_normalized: [3],
                           trunk_attributes_normalized: [7] }
  nominal_qpos        : [n_joints]  per-actuator nominal joint angle
  actuator_qpos_idx   : [n_joints]  qpos index for each actuator
  actuator_qvel_idx   : [n_joints]  qvel index for each actuator
  imu_angular_velocity: { adr, dim }
  trunk_body_id       : int
  scaling_factor      : float   (sigma_robot for action -> joint-target conversion)
  control_dt          : float
  policy_input_dims   : { joint_state: 4, joint_description: 30, general_state: 27 }
  observation_normalization : { joint_pos: 6.5, joint_vel: 180.0, prev_action: 10.0, imu_ang_vel: 50.0, clip: [-10, 10] }
  gait                : { period: 0.5, phase_offset: [0.0, -3.14159] }

The JSON is written to <out>/<design>.json. Per-design files are tiny (~5 KB).

Run from the loco_mjx repo root:
  python shape_your_body_website/demo/scripts/dump_design_metadata.py \
      --setting e1 --robot unitree_go2 --seed 0 \
      --out shape_your_body_website/demo/scenes/unitree_go2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SHORT = {"unitree_go2": "go2", "mit_humanoid": "mit", "golem": "golem"}


def opt_group_for(setting, robot, seed):
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
    raise ValueError(setting)


def make_init_particle(dim, seed, c):
    if seed < 0:
        return np.zeros(int(dim), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(low=-float(c), high=float(c), size=int(dim)).astype(np.float32)


def load_cached(cache, setting, robot, seed):
    group = opt_group_for(setting, robot, seed)
    eval_path = cache / "eval" / f"{group}.json"
    f_opt = np.asarray(json.loads(eval_path.read_text())["particle_normalized"], dtype=np.float32)

    base = {
        ("e1", "unitree_go2"):  "esgjr7ec",
        ("e1", "mit_humanoid"): "k1r2zj4b",
        ("e1", "golem"):        "z5o4aztg",
    }
    run_id = base.get((setting, robot), "9huz0fww")
    coeff = float((cache / run_id / "curriculum_coeff.txt").read_text().strip())

    traj = None
    traj_path = cache / "trajectory" / f"{group}.npz"
    if traj_path.exists():
        with np.load(traj_path) as d:
            for k in ("particles", "particle_trajectory", "factors", "trajectory"):
                if k in d.files:
                    traj = np.asarray(d[k], dtype=np.float32)
                    break
    return f_opt, coeff, traj


def build_env(robot):
    import loco_mjx.environments.locomotion.single_robot_urma.mujoco  # noqa
    from ml_collections import config_dict
    from rl_x.environments.environment_manager import (
        get_environment_config, get_environment_create_train_and_eval_env,
    )
    from rl_x.runner.default_config import get_config as get_runner_config

    name = "locomotion.single_robot_urma.mujoco"
    cfg = config_dict.ConfigDict()
    cfg.runner = get_runner_config("test")
    cfg.runner.mode = "test"
    cfg.environment = get_environment_config(name)
    cfg.environment.train_robots = (robot,)
    cfg.environment.nr_envs = 1
    cfg.environment.render = False
    cfg.environment.add_goal_arrow = False
    cfg.environment.domain_randomization.initial_state.type = "default"
    cfg.environment.domain_randomization.sampling_type = "none"
    cfg.environment.domain_randomization.perturbation.sampling_type = "none"
    cfg.environment.domain_randomization.observation_noise.type = "none"
    train_env, _ = get_environment_create_train_and_eval_env(name)(cfg)
    return train_env.envs[0]


def collect(env_inner):
    import mujoco
    s = env_inner.internal_state
    mj_model = s["mj_model"]
    joint_desc = np.asarray(s["policy_joint_descriptions"], dtype=np.float32)
    n_joints, dim_desc = joint_desc.shape

    actuator_qpos_idx = list(map(int, env_inner.actuator_joint_mask_qpos))
    actuator_qvel_idx = list(map(int, env_inner.actuator_joint_mask_qvel))
    nominal_qpos = list(map(float, s["actuator_joint_nominal_positions"]))

    # actuator_joint_keep_nominal: per-actuator binary flag (1 = joint should hold nominal).
    keep_nominal = list(map(int, np.asarray(s["actuator_joint_keep_nominal"]).reshape(-1)))

    imu_ang_adr = int(env_inner.imu_angular_velocity_sensor_adr)
    imu_ang_dim = int(env_inner.imu_angular_velocity_sensor_dim)

    # Per-design corrected trunk z so the lowest foot rests on the floor.
    # Mirrors loco_mjx/.../seen_robot_functions/default.py L343-345 with plane terrain (center_height=0).
    qpos = env_inner.initial_qpos.copy()
    qpos[env_inner.actuator_joint_mask_qpos] = s["actuator_joint_nominal_positions"]
    data = mujoco.MjData(mj_model)
    data.qpos = qpos
    mujoco.mj_forward(mj_model, data)
    min_feet_z = float(np.min(data.geom_xpos[env_inner.foot_geom_indices, 2]))
    home_qpos_z = float(qpos[2]) - min_feet_z   # plane floor at z=0

    return {
        "n_joints": int(n_joints),
        "joint_descriptions": joint_desc.tolist(),
        "general_static": {
            "gains_and_scale_normalized": list(map(float, s["seen_gains_and_action_scaling_factor_normalized"])),
            "mass_normalized": list(map(float, s["seen_mass_normalized"])),
            "robot_dimensions_normalized": list(map(float, s["robot_dimensions_normalized"])),
            "trunk_attributes_normalized": list(map(float, s["seen_trunk_attributes_normalized"])),
        },
        "nominal_qpos": nominal_qpos,
        "actuator_qpos_idx": actuator_qpos_idx,
        "actuator_qvel_idx": actuator_qvel_idx,
        "keep_nominal": keep_nominal,
        "imu_angular_velocity": {"adr": imu_ang_adr, "dim": imu_ang_dim},
        "trunk_body_id": int(env_inner.trunk_body_id),
        "scaling_factor": float(s["scaling_factor"]),
        "control_dt": float(env_inner.dt),
        "home_qpos_z": home_qpos_z,
        "input_dims": {"joint_state": 4, "joint_description": int(dim_desc), "general_state": 27},
        "observation_normalization": {
            "joint_pos": 6.5,
            "joint_vel": 180.0,
            "prev_action": 10.0,
            "imu_ang_vel": 50.0,
            "clip": [-10.0, 10.0],
        },
        "gait": {"period": 0.5, "phase_offset": [0.0, -float(np.pi)]},
    }


def write_design(env_inner, factor, out_path):
    env_inner.set_factor_params(np.asarray(factor, dtype=np.float64))
    out_path.write_text(json.dumps(collect(env_inner)))
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["e1", "e2", "e4"], required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("analysis/.viz_cache"))
    p.add_argument("--with-reference", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env_inner = build_env(args.robot)
    fd = env_inner.factor_dim
    dim = int(fd() if callable(fd) else fd)

    if args.with_reference:
        write_design(env_inner, np.zeros(dim, dtype=np.float32), args.out / "reference.json")
        print(f"[meta] wrote {args.out / 'reference.json'}")

    for seed in args.seeds:
        f_opt, coeff, traj = load_cached(args.cache_dir, args.setting, args.robot, seed)
        if traj is None:
            print(f"[meta] skipping seed {seed}: no trajectory NPZ", file=sys.stderr)
            continue
        prefix = f"s{seed}_iter_"
        for i, fac in enumerate(traj):
            write_design(env_inner, fac, args.out / f"{prefix}{i:02d}.json")
        print(f"[meta] wrote {len(traj)} {prefix}*.json files")


if __name__ == "__main__":
    main()
