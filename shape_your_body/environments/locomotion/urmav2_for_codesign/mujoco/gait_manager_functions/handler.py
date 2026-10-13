from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.gait_manager_functions.default import DefaultGaitManager


def get_gait_manager_function(name, env, **kwargs):
    if name == "default":
        return DefaultGaitManager(env, **kwargs)
    else:
        raise NotImplementedError
