# Shape Your Body: Value Gradients for Multi-Embodiment Robot Design

│ [Website](https://nico-bohlinger.github.io/shape-your-body) │ [Paper](https://arxiv.org/pdf/2606.00702) │ [Video](https://youtu.be/_SHSWUSQZvg) │

<img src="image.png"/>

Code for Value-Gradient Design Search (VGDS) and the surrogate optimizer baselines on the 50 legged robots of the paper.


## Installation
1. Install RL-X

Default installation for a Linux system with a NVIDIA GPU.
For other configurations, see the RL-X [documentation](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide).
```
conda create -n shape_your_body python=3.11.4
conda activate shape_your_body
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
git checkout 14379ffc84b8e2936941edaa69f1e70863f64f51
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -E '\-cu12|\-cu13' | cut -d '=' -f 1) -y
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install "jax[cuda12]==0.7.2"
```

2. Install the project
```
git clone git@github.com:nico-bohlinger/shape-your-body.git
cd shape-your-body
pip install -e .
```


## Experiments
All experiments are started from the `experiments` folder, either with `bash` or with `sbatch` on a Slurm cluster.
The defaults are the settings of the paper, the scripts set what differs between the runs.

1. Train the policy and critic on randomized designs
```
cd shape-your-body/experiments
bash train_go2.sh
```
- Single robots: `train_go2.sh`, `train_mit_humanoid.sh` or `train_golem.sh`
- Robot classes: `train_quadrupeds.sh` (uses 4 GPUs), `train_bipeds.sh` or `train_hexapods.sh`
- All 50 robots: `train_all_robots.sh`

The model is saved to `runs/shape_your_body/<exp_name>/<run_name>/models/latest.model`.
To log to Weights & Biases, set `--runner.wandb_entity` in the script, otherwise set `--runner.track_wandb=False`.

2. Search a design with the trained critic
```
bash search_design.sh
```
- `--algorithm.surrogate_search_method`: `vgds` for VGDS, `random`, `bo`, `cma_es`, `turbo`, `pso`, `cem`, `de`, `ars` or `gcpfo` for the surrogate optimizer baselines and `nominal` to only save the start design
- `--algorithm.surrogate_init_particle_uniform_seed`: seed of the random start design, -1 starts at the nominal design
- `--algorithm.surrogate_particle_l2_prior_weight`: coefficient of the trust region around the nominal design, 100 by default
- `--algorithm.surrogate_search_seed`, `--algorithm.surrogate_init_particle_uniform_seed` and `--environment.seed`: 0 to 9 in the paper

3. Evaluate the design with the trained policy
```
bash evaluate_design.sh
```
The episode return and the velocity tracking error are written to `--algorithm.eval_design_output_path`.

4. Render the trained policy on the design with MuJoCo
```
bash render.sh
```
An empty `--algorithm.eval_design_particle_path` renders the nominal design.
The arrow above the robot shows the target velocity.
Either modify the `commands/trunk_velocity.txt` file, where the values are target x, y and yaw velocities, or connect an **Xbox 360** controller and control the target x, y velocity with the left joystick and the yaw velocity with the right joystick.
With more robots in `--environment.train_robots` and `--environment.nr_envs` set to the number of robots, switch the robot by changing the robot id in the `multi_render.txt` file or by pressing the LB and RB button on the controller.

The search, evaluation and rendering scripts use the Unitree Go2 model from `train_go2.sh`.
For other models, robots, methods or seeds, change the settings and paths in the scripts.
The search needs a state bank of the robot.
The training of a single robot saves it to `models/surrogate.npz`.
In every other case, collect it first with `collect_state_bank.sh`, which is set up for the Unitree Go2 and the model from `train_quadrupeds.sh`.


## License
The robot assets and XML files do not belong to the authors of this repository.
They are used for research purposes only and are not covered by the MIT license. The MIT license only covers the code in this repository.
