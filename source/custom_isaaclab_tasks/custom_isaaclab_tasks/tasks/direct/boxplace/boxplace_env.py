# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    CollisionPropertiesCfg,
)
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, spawn_preview_surface
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab.sim.utils import bind_visual_material
from isaaclab.sensors import CameraCfg, Camera, TiledCamera, TiledCameraCfg, save_images_to_file

from . import factory_control, factory_utils
from .boxplace_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, BoxPlaceEnvCfg

import gymnasium as gym

class BoxPlaceEnv(DirectRLEnv):
    cfg: BoxPlaceEnvCfg

    def __init__(self, cfg: BoxPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Update number of obs/states
        camera_resolution = (cfg.tiled_camera.height, cfg.tiled_camera.width)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-float("inf"), float("inf"), shape=(sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])+cfg.action_space,)), 
            "rgb": gym.spaces.Box(-float("inf"), float("inf"), shape=(3, *camera_resolution))
        })
        self.state_dim = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        self.max_episode_steps = (cfg.episode_length_s // cfg.sim.dt) // cfg.decimation
        self.cfg_task = cfg.task


        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(
            self.cfg.ctrl.default_task_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))

        self.pos_threshold = torch.tensor(
            self.cfg.ctrl.pos_action_threshold, device=self.device
        ).repeat((self.num_envs, 1))
        self.rot_threshold = torch.tensor(
            self.cfg.ctrl.rot_action_threshold, device=self.device
        ).repeat((self.num_envs, 1))
        self.gripper_threshold = torch.tensor(
            self.cfg.ctrl.gripper_action_threshold, device=self.device
        ).repeat((self.num_envs, 1))
        factory_utils.set_friction(
            self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs
        )

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frames = [
            torch.zeros((self.num_envs, 3), device=self.device)
            for box in self._box_assets
        ]
        self.init_fixed_pos_obs_noises = [
            torch.zeros((self.num_envs, 3), device=self.device)
            for box in self._box_assets
        ]

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index(
            "panda_fingertip_centered"
        )

        # Tensors for finite-differencing.
        self.last_update_timestamp = (
            0.0  # Note: This is for finite differencing body velocities.
        )
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self.ep_success_times = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self.target_box = torch.zeros(size=(1,1), device=self.device)
        self.target_boxes = torch.full(
            size=(self.num_envs,1), fill_value=int(self.target_box), dtype=torch.int, device=self.device
        )

        # The position at which each box gets placed
        # i.e. box_permutations[env_id, i] is the position of box i
        self.box_permutations = torch.zeros(
            (self.num_envs, 3), dtype=torch.int, device=self.device
        )

        self.fixed_positions = [
            torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            for box in self._box_assets
        ]
        self.fixed_quats = [
            torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
            for box in self._box_assets
        ]

    def _setup_scene(self):
        """Initialize simulation scene."""

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -1.05),
        )

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        cfg.func(
            "/World/envs/env_.*/Table",
            cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Add the boxes
        self._box_assets = []
        for box_idx, box_name in enumerate(["red_box", "green_box", "blue_box"]):
            box_properties = RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            )
            colour = [0.0, 0.0, 0.0]
            colour[box_idx] = 1.0
            box_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{box_name}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0.3 * (box_idx - 1), 0.15)
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd",
                    rigid_props=box_properties,
                    collision_props=CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=tuple(colour),  # RGB in [0,1]
                    ),
                ),
            )
            box = RigidObject(cfg=box_cfg)
            self._box_assets.append(box)
            self.scene.rigid_objects[box_name] = box

        # Rigid body properties of toy_truck and box
        toy_truck_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        toy_mass_properties = MassPropertiesCfg(
            mass=0.05,
        )
        toy_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/ToyTruck",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/ToyTruck/toy_truck.usd",
                rigid_props=toy_truck_properties,
                mass_props=toy_mass_properties,
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
            ),
        )
        self._toy = RigidObject(cfg=toy_cfg)
        self.scene.rigid_objects["toy"] = self._toy

        # Create the materials for the toy
        spawn_preview_surface(
            prim_path="/World/red_material",
            cfg=PreviewSurfaceCfg(
                diffuse_color=(1.0,0.0,0.0),  # RGB in [0,1]
            ),
        )
        spawn_preview_surface(
            prim_path="/World/green_material",
            cfg=PreviewSurfaceCfg(
                diffuse_color=(0.0,1.0,0.0),  # RGB in [0,1]
            ),
        )
        spawn_preview_surface(
            prim_path="/World/blue_material",
            cfg=PreviewSurfaceCfg(
                diffuse_color=(0.0,0.0,1.0),  # RGB in [0,1]
            ),
        )

        # Cameras
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?

        # Find the position/orientation of the left/center/right boxes,
        # store it in self.fixed_positions and self.fixed_quats
        for env_id in range(self.num_envs):
            for box_position in range(3):  # left, center, right
                # Find the asset which is at the left/center/right
                asset_idx_for_position = (
                    (self.box_permutations[env_id, :] == box_position).float().argmax()
                )
                # Store the position for that asset in the self.fixed_positions
                self.fixed_positions[box_position][env_id] = (
                    self._box_assets[asset_idx_for_position].data.root_pos_w[env_id, :]
                    - self.scene.env_origins[env_id, :]
                )
                self.fixed_quats[box_position][env_id, :] = self._box_assets[
                    asset_idx_for_position
                ].data.root_quat_w[env_id, :]
                # Store the obs frame in the self.fixed_pos_obs_frame
                self.fixed_pos_obs_frames[box_position][env_id, :] = (
                    self.fixed_positions[box_position][env_id, :]
                )
        self.held_pos = self._toy.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._toy.data.root_quat_w

        self.fingertip_midpoint_pos = (
            self._robot.data.body_pos_w[:, self.fingertip_body_idx]
            - self.scene.env_origins
        )
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[
            :, self.fingertip_body_idx
        ]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[
            :, self.fingertip_body_idx
        ]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[
            :, self.fingertip_body_idx
        ]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[
            :, self.left_finger_body_idx - 1, 0:6, 0:7
        ]
        self.right_finger_jacobian = jacobians[
            :, self.right_finger_body_idx - 1, 0:6, 0:7
        ]
        self.fingertip_midpoint_jacobian = (
            self.left_finger_jacobian + self.right_finger_jacobian
        ) * 0.5
        self.arm_mass_matrix = (
            self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        )
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat,
            torch_utils.quat_conjugate(self.prev_fingertip_quat),
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_positions = [
            self.fixed_pos_obs_frames[i] + self.init_fixed_pos_obs_noises[i]
            for i in range(len(self._box_assets))
        ]

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
            "target_box": self.target_boxes,
        }
        for i in range(len(self._box_assets)):
            obs_dict[f"fingertip_pos_rel_fixed_{i}"] = (
                self.fingertip_midpoint_pos - noisy_fixed_positions[i]
            )

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
            "target_box": self.target_boxes, 
            "box_permutation": self.box_permutations,
            "gripper_state": self.ctrl_target_joint_pos[:, 7:9]
        }
        for i in range(len(self._box_assets)):
            state_dict[f"fingertip_pos_rel_fixed_{i}"] = (
                self.fingertip_midpoint_pos - self.fixed_pos_obs_frames[i]
            )
            state_dict[f"held_pos_rel_fixed_{i}"] = (
                self.held_pos - self.fixed_pos_obs_frames[i]
            )
            state_dict[f"fixed_pos_{i}"] = self.fixed_positions[i]
            state_dict[f"fixed_quat_{i}"] = self.fixed_quats[i]
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(
            obs_dict, self.cfg.obs_order + ["prev_actions"]
        )
        state_tensors = factory_utils.collapse_obs_dict(
            state_dict, self.cfg.state_order + ["prev_actions"]
        )
        raw_visual_obs = self._tiled_camera.data.output["rgb"]
        visual_obs = torch.permute(raw_visual_obs, dims=(0,3,1,2))

        if self.cfg.write_image_to_file:
            save_images_to_file(raw_visual_obs[0:1]/255.0, f"/data/AlexPleava/cartpole_current_obs.png")
        _, state_dict = self._get_factory_obs_state_dict()
        state_tensors = factory_utils.collapse_obs_dict(
            state_dict, self.cfg.state_order
        )
        self.extras["state"] = state_tensors

        parallel_obs = {"state": obs_tensors, "rgb": visual_obs}
        # parallel_obs = torch.load("/home/AlexPleava/projects/distill-plan/baselines/evaluation/scripts/dstl/observation.pt")

        # with open("observation.txt", "w") as obs_file:
        #     obs_file.write(str(parallel_obs))
        # assert False

        return parallel_obs#{"state": obs_tensors, "rgb": visual_obs}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = (
            self.ema_factor * action.clone().to(self.device)
            + (1 - self.ema_factor) * self.actions
        )

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                self.num_envs, 1
            ),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        target_euler_xyz = torch.stack(
            torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1
        )
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2],
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold
        gripper_action = self.gripper_threshold * self.actions[:, 6:7]
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                self.num_envs, 1
            ),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        target_euler_xyz = torch.stack(
            torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1
        )
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2],
        )
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=gripper_action,
        )

    def generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos,
    ):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        curr_successes = self.object_a_is_into_b(
            xy_threshold=0.10, height_diff=0.15, height_threshold=0.10
        )
        curr_successes &= self.gripper_is_open()
        return curr_successes

    def _log_metrics(self, rew_dict, rew_buf, curr_successes):
        """Keep track of episode statistics and log rewards."""

        # Log which environments are currently successful
        self.extras["successes"] = curr_successes

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(
            curr_successes, torch.logical_not(self.ep_succeeded)
        )
        self.ep_succeeded[curr_successes] = 1
        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[
            first_success_ids
        ]
        self.extras["success_times"] = self.ep_success_times

        # The current length of the episode so far
        self.extras["episode_lengths"] = self.episode_length_buf

        # The state vector of the scene
        _, state_dict = self._get_factory_obs_state_dict()
        state_tensors = factory_utils.collapse_obs_dict(
            state_dict, self.cfg.state_order
        )
        self.extras["state"] = state_tensors

        # The reward terms
        for rew_name, rew in rew_dict.items():
            self.extras[f"reward_terms_{rew_name}"] = rew.mean()
        # The total reward
        self.extras["rewards"] = rew_buf

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["xy_dist_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_metrics(rew_dict, rew_buf, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        rew_dict["xy_dist_coarse"] = factory_utils.squashing_fn(
            self.get_xy_dist(), self.cfg.coarse_alignment_sharpness, 0
        )
        rew_scales["xy_dist_coarse"] = self.cfg.coarse_alignment_coefficient
        rew_dict["xy_dist_fine"] = factory_utils.squashing_fn(
            self.get_xy_dist(), self.cfg.fine_alignment_sharpness, 0
        )
        rew_scales["xy_dist_fine"] = self.cfg.fine_alignment_coefficient

        rew_dict["inside_and_released"] = self.object_a_is_into_b(
            xy_threshold=0.10, height_diff=0.0, height_threshold=0.10
        )
        rew_dict["inside_and_released"] &= self.gripper_is_open()
        rew_dict["inside_and_released"] = rew_dict["inside_and_released"].float()
        rew_scales["inside_and_released"] = self.cfg.success_coefficient

        return rew_dict, rew_scales

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)
        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(
            joints=self.cfg.ctrl.reset_joints, env_ids=env_ids
        )
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._toy.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._toy.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._toy.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._toy.reset()

        for box in self._box_assets:
            fixed_state = box.data.default_root_state.clone()[env_ids]
            fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
            fixed_state[:, 7:] = 0.0
            box.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
            box.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
            box.reset()

    def set_pos_inverse_kinematics(
        self,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        env_ids,
    ):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[
                    env_ids
                ],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[
                    env_ids
                ],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])
            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(
            self.ctrl_target_joint_pos[env_ids], env_ids=env_ids
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        self.randomize_boxes(env_ids=env_ids)
        self.step_sim_no_action()

        # TODO: With hand position randomization, this causes the toy to not initialize in the hand
        # self.randomize_pose(env_ids=env_ids)

        # self.step_sim_no_action()

        self.randomize_toy(env_ids=env_ids)

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(
            self.cfg.ctrl.reset_task_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def randomize_boxes(self, env_ids):
        # Get a random permutation of the boxes for each environment
        # box_permutations[env_id, box_id] = position of box # box_id
        # 0 is -Y, 1 is middle, 2 is +Y
        self.box_permutations[env_ids, :] = torch.stack(
            [torch.randperm(3, dtype=torch.int, device=self.device) for _ in env_ids]
        )

        # For each box asset, write its state in each environment
        for box_asset_idx, box_asset in enumerate(self._box_assets):
            box_asset_state = box_asset.data.default_root_state.clone()
            for env_id in env_ids:
                random_x_offset = self.cfg.box_position_randomization * (
                    torch.rand(1, dtype=torch.float32, device=self.device) - 0.5
                )
                random_y_position = 0.3 * (
                    self.box_permutations[env_id, box_asset_idx] - 1
                )
                box_asset_state[env_id, 0:1] += random_x_offset
                box_asset_state[env_id, 1:2] = random_y_position
                box_asset_state[env_id, 0:3] += self.scene.env_origins[env_id, :]
                box_asset_state[env_id, 7:] = 0.0
            box_asset.write_root_pose_to_sim(box_asset_state[:, 0:7], env_ids=env_ids)
            box_asset.write_root_velocity_to_sim(
                box_asset_state[:, 7:], env_ids=env_ids
            )
            box_asset.reset()

            fixed_asset_pos_noise = torch.randn(
                (len(env_ids), 3), dtype=torch.float32, device=self.device
            )
            fixed_asset_pos_rand = torch.tensor(
                self.cfg.obs_rand.fixed_asset_pos,
                dtype=torch.float32,
                device=self.device,
            )
            fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(
                fixed_asset_pos_rand
            )
            self.init_fixed_pos_obs_noises[box_asset_idx] = fixed_asset_pos_noise
        self.step_sim_no_action()

    def randomize_toy(self, env_ids):
        
        if self.cfg.task_index is None:
            self.target_box = torch.randint(low=0,high=3,size=(1,1,), device=self.device)
        elif self.cfg.task_index in (1, 2, 3):
            self.target_box = self.cfg.task_index
        else: 
            raise ValueError(f"`task_index = {self.cfg.task_index}` is not valid. Choose one of (None, 1, 2, 3) for this env.")
        self.target_boxes = torch.full(
            size=(self.num_envs,1), fill_value=int(self.target_box), dtype=torch.int, device=self.device
        )
        colour_name = ("red", "green", "blue")[int(self.target_box)]
        bind_visual_material(
            prim_path="/World/envs/env_0/ToyTruck", 
            material_path=f"/World/{colour_name}_material"
        )

        toy_state = self._toy.data.default_root_state.clone()[env_ids]
        random = self.cfg.toy_position_randomization * (
            torch.rand((len(env_ids), 1), dtype=torch.float32, device=self.device) - 0.5
        )
        toy_state[env_ids, 0:1] += random
        toy_state[:, 0:3] += (
            self.scene.env_origins[env_ids] + self.fingertip_midpoint_pos
        )
        toy_state[:, 3:7] = self.fingertip_midpoint_quat
        toy_state[:, 7:] = 0.0
        self._toy.write_root_pose_to_sim(toy_state[:, 0:7], env_ids=env_ids)
        self._toy.write_root_velocity_to_sim(toy_state[:, 7:], env_ids=env_ids)
        self._toy.reset()

    def randomize_pose(self, env_ids):
        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = self.fingertip_midpoint_pos

            rand_sample = torch.rand(
                (n_bad, 3), dtype=torch.float32, device=self.device
            )
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(
                self.cfg.initial_pos_noise, device=self.device
            )
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device)
                .unsqueeze(0)
                .repeat(n_bad, 1)
            )

            rand_sample = torch.rand(
                (n_bad, 3), dtype=torch.float32, device=self.device
            )
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(
                self.cfg_task.hand_init_orn_noise, device=self.device
            )
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(
                hand_init_orn_rand
            )
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0],
                pitch=hand_down_euler[:, 1],
                yaw=hand_down_euler[:, 2],
            )

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=self.cfg.ctrl.reset_joints,  # [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
                env_ids=bad_envs,
            )

            ik_attempt += 1

    def object_a_is_into_b(
        self,
        xy_threshold: float = 0.03,  # xy_distance_threshold
        height_threshold: float = 0.04,  # height_distance_threshold
        height_diff: float = 0.0,  # expected height_diff
    ) -> torch.Tensor:
        """Check if an object a is put into another object b by the specified robot."""
        successes = []
        # check object a is into object b
        for env_idx in range(self.num_envs):
            pos_diff = self._toy.data.root_pos_w - self._box_assets[int(self.target_box)].data.root_pos_w
            height_dist = torch.linalg.vector_norm(pos_diff[env_idx, 2:])
            xy_dist = torch.linalg.vector_norm(pos_diff[env_idx, :2])
            success = torch.logical_and(
                xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold
            )
            successes.append(success)
        return torch.tensor(successes, device=self.device)

    def get_xy_dist(self):
        xy_dists = []
        for env_idx in range(self.num_envs):
            pos_diff = self._toy.data.root_pos_w - self._box_assets[int(self.target_box)].data.root_pos_w
            xy_dist = torch.linalg.vector_norm(pos_diff[env_idx, :2])
            xy_dists.append(xy_dist)
        return torch.tensor(xy_dists, device=self.device)

    def gripper_is_open(self):
        return (self.ctrl_target_joint_pos[:, 7:9] >= 0.03).all(axis=1)
