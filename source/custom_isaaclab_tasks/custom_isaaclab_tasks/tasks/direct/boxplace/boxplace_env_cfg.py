# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed_0": 3,
    "fingertip_pos_rel_fixed_1": 3,
    "fingertip_pos_rel_fixed_2": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "target_box": 1,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed_0": 3,
    "fingertip_pos_rel_fixed_1": 3,
    "fingertip_pos_rel_fixed_2": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed_0": 3,
    "held_pos_rel_fixed_1": 3,
    "held_pos_rel_fixed_2": 3,
    "held_quat": 4,
    "fixed_pos_0": 3,
    "fixed_quat_0": 4,
    "fixed_pos_1": 3,
    "fixed_quat_1": 4,
    "fixed_pos_2": 3,
    "fixed_quat_2": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "target_box": 1,
    "box_permutation": 3,
    "gripper_state": 2,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]
    gripper_action_threshold = [0.05]

    reset_joints = [
        1.5178e-03,
        -1.9651e-01,
        -1.4364e-03,
        -1.7761,
        -2.7717e-04,
        1.7796,
        7.8556e-01,
    ]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [
        1.5178e-03,
        -1.9651e-01,
        -1.4364e-03,
        -1.7761,
        -2.7717e-04,
        1.7796,
        7.8556e-01,
    ]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class BoxPlaceEnvCfg(DirectRLEnvCfg):
    decimation = 20
    action_space = 7
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 58
    obs_order: list = [
        "fingertip_pos_rel_fixed_0",
        "fingertip_pos_rel_fixed_1",
        "fingertip_pos_rel_fixed_2",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "target_box",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed_0",
        "held_pos_rel_fixed_1",
        "held_pos_rel_fixed_2",
        "held_quat",
        "fixed_pos_0",
        "fixed_quat_0",
        "fixed_pos_1",
        "fixed_quat_1",
        "fixed_pos_2",
        "fixed_quat_2",
        "target_box",
        "box_permutation",
        "gripper_state",
    ]

    box_position_randomization = 0.25
    toy_position_randomization = 0.05
    initial_pos_noise = [0.2, 0.2, 0.2]

    coarse_alignment_sharpness = 5
    coarse_alignment_coefficient = 1 / 6
    fine_alignment_sharpness = 100
    fine_alignment_coefficient = 2 / 6
    success_coefficient = 3 / 6

    task_name: str = "box_place"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()
    write_image_to_file = False
    target_box = None
    episode_length_s = 30.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:1",
        dt=1 / 100,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # If clone_in_fabric is True, camera initialization fails. No idea why.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=2.0, clone_in_fabric=False, replicate_physics=True
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 0, 0.5), 
        lookat=(0.5, 0, 0.5), 
        origin_type="asset_root", 
        asset_name="robot", 
        env_index=0, 
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_fingertip_centered/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0, -0.05), rot=(0.3826834, 0, -0.9238795, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        width=64,
        height=64,
    )



@configclass
class FactoryTaskPegInsertCfg(BoxPlaceEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshCfg(BoxPlaceEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadCfg(BoxPlaceEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0
