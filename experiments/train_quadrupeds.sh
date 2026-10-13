#!/bin/bash

#SBATCH --job-name=shape_your_body
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6000
#SBATCH --time=2-23:59:59

python experiment.py \
    --algorithm.name=urmav2_for_codesign.mjx \
    --algorithm.total_timesteps=441548800 \
    --algorithm.nr_epochs=8 \
    --algorithm.minibatch_size=28672 \
    --algorithm.evaluation_and_save_frequency=-1 \
    --environment.name=locomotion.urmav2_for_codesign.mjx \
    --environment.train_robots="('unitree_a1', 'unitree_go1', 'unitree_b2', 'anymal_b', 'anymal_c', 'barkour_v0', 'barkour_vb', 'silver_badger', 'honey_badger', 'bittle', 'aibo', 'solo12', 'spot', 'spot_mini',)" \
    --environment.nr_envs=1792 \
    --environment.reward.log_info=True \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=False \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="placeholder" \
    --runner.project_name="shape_your_body" \
    --runner.exp_name="quadrupeds" \
    --runner.run_name="seed0" \
    --runner.notes=""
