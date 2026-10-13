from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.action_delay_functions.default import DefaultActionDelay
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.action_delay_functions.none import NoneActionDelay


def get_domain_randomization_action_delay_function(name, env, **kwargs):
    if name == "none":
        return NoneActionDelay(env, **kwargs)
    elif name == "default":
        return DefaultActionDelay(env, **kwargs)
    else:
        raise NotImplementedError
