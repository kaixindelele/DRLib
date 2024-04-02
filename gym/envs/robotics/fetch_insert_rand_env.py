import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.target_goal = np.array([1.35, 0.8, 0.37])

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_pos[2] -= 0.02
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # check the gg == ag:
            gg2dg_xy = np.linalg.norm(grip_pos[:2]-self.target_goal[:2])
            gg2dg_z = np.linalg.norm(grip_pos[2]-self.target_goal[2])
            if gg2dg_xy < 0.01 and gg2dg_z < 0.09:
                object_pos = grip_pos
            else:
                object_pos = np.array([self.target_goal[0],
                                       self.target_goal[1],
                                       0.44])
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'grip_goal': grip_pos.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def reset_hole(self, hole_center, half_hole_size=0.02, hsw1y=0.1):
        cx, cy = hole_center
        hsw1x = (hsw1y - half_hole_size) / 2.0
        w1x = cx - hsw1x - half_hole_size
        w1y = cy

        hsw2x = (2*hsw1y-2*hsw1x)/2.0
        hsw2y = (hsw1y-half_hole_size)/2.0
        w2x = w1x + hsw1x + (2*hsw1y-2*hsw1x)/2.0
        w2y = w1y - hsw1y + (hsw1y-half_hole_size)/2.0

        hsw3x = hsw2x
        hsw3y = hsw2y
        w3x = w2x
        w3y = w1y + hsw1y - (hsw1y-half_hole_size)/2.0

        hsw4x = (hsw1y-half_hole_size)/2
        hsw4y = half_hole_size
        w4x = cx + half_hole_size + (hsw1y - half_hole_size) / 2.0
        w4y = cy
        # print("((w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y)):",
        #       ((w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y)))
        # print("((hsw1x, hsw1y), (hsw2x, hsw2y), (hsw3x, hsw3y), (hsw4x, hsw4y)):",
        #       ((hsw1x, hsw1y), (hsw2x, hsw2y), (hsw3x, hsw3y), (hsw4x, hsw4y)))
        return (w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y)

    def sample_dg(self):
        target_x = np.random.random() * 0.35 + 1.1
        target_y = np.random.random() * 0.35 + 0.5
        self.target_goal = self.sim.data.get_site_xpos('target0')
        target_goal = np.array([target_x, target_y, self.target_goal[2]])
        self._render_callback()
        return target_goal

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.target_goal = self.sample_dg()

        (w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y) = self.reset_hole([self.target_goal[0],
                                                                          self.target_goal[1]])

        self.sim.model.body_pos[self.sim.model.body_name2id('w1')] = np.array(
            [w1x, w1y, self.sim.model.body_pos[self.sim.model.body_name2id('w1')][2]])

        self.sim.model.body_pos[self.sim.model.body_name2id('w2')] = np.array(
            [w2x, w2y, self.sim.model.body_pos[self.sim.model.body_name2id('w2')][2]])
        
        self.sim.model.body_pos[self.sim.model.body_name2id('w3')] = np.array(
            [w3x, w3y, self.sim.model.body_pos[self.sim.model.body_name2id('w3')][2]])

        self.sim.model.body_pos[self.sim.model.body_name2id('w4')] = np.array(
            [w4x, w4y, self.sim.model.body_pos[self.sim.model.body_name2id('w4')][2]])

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.target_goal
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
