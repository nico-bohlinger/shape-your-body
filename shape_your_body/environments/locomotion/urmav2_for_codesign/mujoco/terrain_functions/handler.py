from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.terrain_functions.plane import PlaneTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
