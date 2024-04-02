import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py


class PusherHEREnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def reset(self):
        self.sim.reset()
        self.reset_model()
        ob = self._get_obs()
        return ob

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info={}):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        self.reward_type = 'sparse'
        self.distance_threshold = 0.05
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'])

        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        arm_qpos = self.sim.data.qpos.flat[:7]
        arm_qvel = self.sim.data.qvel.flat[:7]
        grip_goal = self.get_body_com("tips_arm")
        object_pos = self.get_body_com("object")
        object_rel_pos = object_pos - grip_goal

        obs = np.concatenate([
            arm_qpos, arm_qvel, grip_goal,
            object_pos, object_rel_pos,
        ])

        achieved_goal = object_pos
        dg = self.get_body_com("goal")

        return {
            'observation': obs.copy(),
            'grip_goal': grip_goal.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': dg,
        }
