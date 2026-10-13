from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.termination_functions.below_height import BelowHeightTermination


def get_termination_function(name, env, **kwargs):
    if name == "below_height":
        return BelowHeightTermination(env, **kwargs)
    else:
        raise NotImplementedError
