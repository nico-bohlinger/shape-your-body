from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.perturbation_functions.default import DefaultDRPerturbation
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.perturbation_functions.none import NoneDRPerturbation


def get_domain_randomization_perturbation_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRPerturbation(env, **kwargs)
    elif name == "none":
        return NoneDRPerturbation(env, **kwargs)
    else:
        raise NotImplementedError
