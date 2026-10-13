from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.control_functions.pd import PDControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    else:
        raise NotImplementedError
