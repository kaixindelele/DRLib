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
        self.target_goal = np.array([1.38, 0.63, 0.53])

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

        # check grip z below the drawer init z.
        gg_z = self.sim.data.get_site_xpos('robot0:grip')[2]
        # print('gg_z:', gg_z)
        if gg_z < 0.64:
            pos_ctrl[2] = abs(pos_ctrl[2])

        pos_ctrl *= 0.04  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # rot_ctrl = [0.0, 0.5, 0., 0.]
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
        # grip_pos[2] -= 0.02
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # window pos
        object_pos = self.sim.data.get_site_xpos('objGeom')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('objGeom'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('objGeom') * dt
        object_velr = self.sim.data.get_site_xvelr('objGeom') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

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

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # drawer_x = 0.0
        # drawer_y = 0.1
        # drawer_x = np.random.random() * 0.24 - 0.12
        # drawer_y = np.random.random() * 0.2 - 0.1
        # valid = False
        # while not valid:
        #     drawer_x = np.random.random() * 0.24 - 0.12
        #     drawer_y = np.random.random() * 0.2 - 0.1
        #     if abs(drawer_x) < 0.08 or drawer_y < 0.03:
        #         valid = False
        #     else:
        #         valid = True

        # self.sim.data.set_joint_qpos('drawer_slidex', drawer_x)
        # self.sim.data.set_joint_qpos('drawer_slidey', drawer_y)

        print("self.sim.model.body_pos[self.sim.model.body_name2id('drawercase_link')]:", self.sim.model.body_pos[self.sim.model.body_name2id('drawercase_link')])
        print("init_goal_slidey:", self.sim.data.get_joint_qpos('goal_slidey'))
        rand_x = np.random.random() * 0.2 - 0.3
        rand_y = np.random.random() * 0.2 + 0.1
        self.sim.model.body_pos[self.sim.model.body_name2id('drawercase_link')] = np.array([rand_x, 0, rand_y])
        print("init_goal_slidey:", self.sim.data.get_joint_qpos('goal_slidey'))
        self.sim.data.set_joint_qpos('goal_slidey', 0.0)
        print("init_goal_slidey:", self.sim.data.get_joint_qpos('goal_slidey'))
        self.sim.forward()
        return True

    def _sample_goal(self):
        object_pos = self.sim.data.get_site_xpos('objGeom')
        goal_x = object_pos[0]
        goal_y = object_pos[1]
        goal_z = np.random.random() * 0.12 + object_pos[2] + 0.03
        goal = np.array([goal_x, goal_y, goal_z])

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
        # gripper_rotation = np.array([0.0, 0.5, 0., 0.])

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
