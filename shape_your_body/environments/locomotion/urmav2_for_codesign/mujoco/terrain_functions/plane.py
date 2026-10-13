import numpy as np


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.ground_height = self.env.c_data.geom_xpos[self.env.floor_geom_id, 2]


    def init(self):
        self.env.internal_state["center_height"] = self.ground_height
        self.env.internal_state["robot_imu_height_over_ground"] = self.env.initial_imu_height - self.env.internal_state["center_height"]


    def check_feet_floor_contact(self):
        contact_geom_pairs = self.env.internal_state["data"].contact.geom
        possible_contact_pairs = np.stack([np.full_like(self.env.foot_geom_indices, self.env.floor_geom_id), self.env.foot_geom_indices], axis=1)
        in_contact = np.any(np.all(contact_geom_pairs == possible_contact_pairs[:, None, :], axis=2), axis=1)

        return in_contact


    def ground_height_at(self, x_in_m, y_in_m):
        return np.ones_like(x_in_m) * self.env.internal_state["center_height"]


    def pre_step(self):
        self.env.internal_state["robot_imu_height_over_ground"] = self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 2] - self.env.internal_state["center_height"]


    def post_step(self):
        return


    def sample(self):
        return
