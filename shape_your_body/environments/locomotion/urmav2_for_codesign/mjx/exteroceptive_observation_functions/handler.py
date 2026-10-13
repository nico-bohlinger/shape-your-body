from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.exteroceptive_observation_functions.height_over_ground import HeightOverGroundExteroceptiveObservation
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.exteroceptive_observation_functions.none import NoneExteroceptiveObservation


def get_exteroceptive_observation_function(name, env, **kwargs):
    if name == "height_over_ground":
        return HeightOverGroundExteroceptiveObservation(env, **kwargs)
    elif name == "none":
        return NoneExteroceptiveObservation(env, **kwargs)
    else:
        raise NotImplementedError
