import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class ReacherHEREnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

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
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'])
        done = False
        info = {}
        if reward == 0:
            info['is_success'] = 1
        else:
            info['is_success'] = 0
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        arm_cos = np.cos(theta)
        arm_sin = np.sin(theta)
        arm_qpos = self.sim.data.qpos.flat[2:]
        arm_qvel = self.sim.data.qvel.flat[:2]
        grip_goal = self.get_body_com("fingertip")
        achieved_goal = self.get_body_com("fingertip")

        obs = np.concatenate([
            theta, arm_cos, arm_sin, arm_qpos, arm_qvel, achieved_goal
        ])

        dg = self.get_body_com("target")

        return {
            'observation': obs.copy(),
            'grip_goal': grip_goal.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': dg,
        }

        # return np.concatenate([
        #     np.cos(theta),
        #     np.sin(theta),
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat[:2],
        #     self.get_body_com("fingertip") - self.get_body_com("target")
        # ])
