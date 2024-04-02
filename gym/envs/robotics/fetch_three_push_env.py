import numpy as np
from copy import deepcopy
from gym.envs.robotics import rotations, robot_three_push_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_three_push_env.RobotEnv):
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
        # 確保夾爪不會擡起來
        # action[2] = 0.0
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
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        np.set_printoptions(precision=4, suppress=True)
        # print("grip_velp:", grip_velp)
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # print("robot_qvel:", robot_qvel)
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
            
            # object 1
            object1_pos = self.sim.data.get_site_xpos('object1')
            # rotations
            object1_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
            # velocities
            object1_velp = self.sim.data.get_site_xvelp('object1') * dt
            object1_velr = self.sim.data.get_site_xvelr('object1') * dt
            # gripper state
            object1_rel_pos = object1_pos - grip_pos
            object1_velp -= grip_velp

            # object 1
            object2_pos = self.sim.data.get_site_xpos('object2')
            # rotations
            object2_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object2'))
            # velocities
            object2_velp = self.sim.data.get_site_xvelp('object2') * dt
            object2_velr = self.sim.data.get_site_xvelr('object2') * dt
            # gripper state
            object2_rel_pos = object2_pos - grip_pos
            object2_velp -= grip_velp
        
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # 需要重新设计ag:
        # 如果对象1没有达到目标，ag而是object,dg是dg, 如果达到了则ag 是object1，dg是dg的正上方+0.05
        sub_goal = self.check_object0(object_pos, self.goal)
        if not sub_goal:
            achieved_goal = np.squeeze(object_pos.copy())
            desired_goal = deepcopy(self.goal)
        else:
            achieved_goal = np.squeeze(object1_pos.copy())
            desired_goal = deepcopy(self.goal)
            # desired_goal[1] += 0.05
        dg0 = deepcopy(self.goal)
        dg1 = deepcopy(self.dg1)
        dg2 = deepcopy(self.dg2)
        obj01_rel_pos = object_pos - object1_pos
        obj02_rel_pos = object_pos - object2_pos
        obj12_rel_pos = object1_pos - object2_pos

        obj_dist01 = object_pos - object1_pos
        obj_dist02 = object_pos - object2_pos
        obj_dist12 = object1_pos - object2_pos
        # dg1[1] += 0.1
        # dg1[0] += 0.1
        obs = np.concatenate([
            grip_pos, gripper_state,
            object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(),

            object1_pos.ravel(), object1_rel_pos.ravel(), object1_rot.ravel(),
            object1_velp.ravel(), object1_velr.ravel(),
            grip_velp, gripper_vel,

            object2_pos.ravel(), object2_rel_pos.ravel(), object2_rot.ravel(),
            object2_velp.ravel(), object2_velr.ravel(),
            obj01_rel_pos, obj12_rel_pos, obj02_rel_pos,
            # obj_dist
            # obj_dist01.ravel(), obj_dist02.ravel(), obj_dist12.ravel(),
            grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'grip_goal': grip_pos.copy(),
            'ag0': object_pos.copy(),
            'ag1': object1_pos.copy(),
            'ag2': object2_pos.copy(),
            'dg0': dg0.copy(),
            'dg1': dg1.copy(),
            'dg2': dg2.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
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
        site_id = self.sim.model.site_name2id('target1')
        self.sim.model.site_pos[site_id] = self.dg1 - sites_offset[0]
        site_id = self.sim.model.site_name2id('target2')
        self.sim.model.site_pos[site_id] = self.dg2 - sites_offset[0]
        self.sim.forward()

    def sample_obj_pos(self):
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                 size=2)
        return object_xpos

    def sample_all_obj_pos(self, success_list=[0., 0.]):
        dist01 = 0.0
        dist12 = 0.0
        dist02 = 0.0
        obj_dist = 0.05
        stack_flag = False

        while (dist01 < obj_dist or dist12 < obj_dist or dist02 < obj_dist) and not stack_flag:
            obj0_xpos = self.sample_obj_pos()
            obj1_xpos = self.sample_obj_pos()
            obj2_xpos = self.sample_obj_pos()
            dist01 = np.linalg.norm(obj0_xpos - obj1_xpos)
            dist12 = np.linalg.norm(obj1_xpos - obj2_xpos)
            dist02 = np.linalg.norm(obj0_xpos - obj2_xpos)
        return obj0_xpos, obj1_xpos, obj2_xpos

    def _reset_sim(self, success_list=[]):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # 确保第0个物体距离末端最近!
        if self.has_object:
            # import time
            # st = time.time()
            obj0_xpos, obj1_xpos, obj2_xpos = self.sample_all_obj_pos(success_list=success_list)
            # obj0
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = obj0_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            # obj1
            object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert object1_qpos.shape == (7,)
            object1_qpos[:2] = obj1_xpos
            object1_qpos[2] = object_qpos[2]
            self.sim.data.set_joint_qpos('object1:joint', object1_qpos)
            # obj2
            object2_qpos = self.sim.data.get_joint_qpos('object2:joint')
            assert object2_qpos.shape == (7,)
            object2_qpos[:2] = obj2_xpos
            object2_qpos[2] = object_qpos[2]
            self.sim.data.set_joint_qpos('object2:joint', object2_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()

    def check_object0(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _is_success(self, obs):
        d0 = goal_distance(obs['ag0'], self.goal)
        d1 = goal_distance(obs['ag1'], obs['dg1'])
        return (d0 < self.distance_threshold and d1 < self.distance_threshold).astype(np.float32)

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
