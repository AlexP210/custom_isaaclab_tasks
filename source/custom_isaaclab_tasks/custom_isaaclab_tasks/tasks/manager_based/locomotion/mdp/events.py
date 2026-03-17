# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def illegal_contact_event(env: ManagerBasedEnv, env_ids: torch.Tensor, threshold: float, sensor_cfg: SceneEntityCfg):
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    env.extras["termination/illegal_contact"] = mdp.illegal_contact(env=env, threshold=threshold, sensor_cfg=sensor_cfg)

def time_out_event(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    env.extras["truncation/time_out"] = mdp.time_out(env)

def success_event(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    env.extras["successes"] = torch.full(size=(env.num_envs,), fill_value=False)

def episode_length_event(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    env.extras["episode_lengths"] = env.episode_length_buf