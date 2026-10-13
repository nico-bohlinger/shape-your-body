from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.command_functions.random import RandomCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    else:
        raise NotImplementedError
