from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.seen_robot_functions.default import DefaultDRSeenRobotFunction
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.seen_robot_functions.none import NoneDRSeenRobotFunction


def get_domain_randomization_seen_robot_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRSeenRobotFunction(env, **kwargs)
    elif name == "none":
        return NoneDRSeenRobotFunction(env, **kwargs)
    else:
        raise NotImplementedError
