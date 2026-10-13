python experiment.py \
    --algorithm.name=urmav2_for_codesign.mjx \
    --algorithm.export_state_bank_path="runs/shape_your_body/quadrupeds/seed0/models/surrogate_go2.npz" \
    --environment.name=locomotion.urmav2_for_codesign.mjx \
    --environment.train_robots="('unitree_go2',)" \
    --environment.seed=0 \
    --runner.mode="test" \
    --runner.track_console=True \
    --runner.load_model="runs/shape_your_body/quadrupeds/seed0/models/latest.model"
