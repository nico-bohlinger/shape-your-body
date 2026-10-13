#!/bin/bash

#SBATCH --job-name=shape_your_body
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6000
#SBATCH --time=2-23:59:59

python experiment.py \
    --algorithm.name=urmav2_for_codesign.mjx \
    --algorithm.total_timesteps=3942400000 \
    --algorithm.minibatch_size=64000 \
    --algorithm.evaluation_and_save_frequency=-1 \
    --environment.name=locomotion.urmav2_for_codesign.mjx \
    --environment.train_robots="('unitree_a1', 'unitree_go1', 'unitree_go2', 'unitree_b2', 'anymal_b', 'anymal_c', 'barkour_v0', 'barkour_vb', 'silver_badger', 'honey_badger', 'bittle', 'aibo', 'solo12', 'spot', 'spot_mini', 'bolt', 'duck_mini', 'mini_pi', 'qmini', 'tron1', 'unitree_h1', 'unitree_g1', 'talos', 'robotis_op3', 'nao_v5', 'adam_lite', 'agibot_x1', 'apptronik_apollo', 'atlas', 'berkeley_humanoid', 'berkeley_humanoid_lite', 'booster_t1', 'elf2', 'engineai_sa01', 'fourier_gr1t2', 'fourier_gr2v3', 'fourier_n1', 'kbot_v1', 'kbot_v2', 'mit_humanoid', 'northwestern_humanoid', 'poppy_humanoid', 'sigmaban', 'star1', 'valkyrie', 'zbot', 'crab', 'custom_hexapod', 'tom_hexapod', 'golem',)" \
    --environment.nr_envs=16000 \
    --environment.reward.log_info=True \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=False \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="placeholder" \
    --runner.project_name="shape_your_body" \
    --runner.exp_name="all_robots" \
    --runner.run_name="seed0" \
    --runner.notes=""
