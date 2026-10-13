from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.sampling_functions.step_probability import StepProbabilitySampling
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.sampling_functions.step_probability_and_reset import StepProbabilityAndResetSampling
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.sampling_functions.every_step import EveryStepSampling
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.sampling_functions.none import NoneSampling


def get_sampling_function(name, env, **kwargs):
    if name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    if name == "step_probability_and_reset":
        return StepProbabilityAndResetSampling(env, **kwargs)
    elif name == "every_step":
        return EveryStepSampling(env, **kwargs)
    elif name == "none":
        return NoneSampling(env, **kwargs)
    else:
        raise NotImplementedError
