robot_config = {
    "short_name": "apo",

    # Best guess taken from: https://github.com/gkjohnson/nasa-urdf-robots/blob/master/val_description/model/robots/valkyrie_A.urdf
    "actuator_joint_max_velocities": [
        5.89, 9.0, 9.0,
        5.0, 5.0, 5.0,
        5.89, 11.5, 5.89, 11.5, 1.0, 1.0,
        5.89, 11.5, 5.89, 11.5, 1.0, 1.0,
        5.89, 7.0, 6.11, 6.11, 11.0, 11.0,
        5.89, 7.0, 6.11, 6.11, 11.0, 11.0
    ],

    "scaling_factor": 0.75,

    "actuator_joints_to_stay_near_nominal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}
