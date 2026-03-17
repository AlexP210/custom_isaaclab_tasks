# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

import os
filenames = os.listdir(os.path.join(os.path.dirname(__file__), "agents"))
agent_cfg_filenames = [filename for filename in filenames if "cfg" in filename]
agent_names = [agent_cfg_filename.split("_cfg")[0] for agent_cfg_filename in agent_cfg_filenames]
entry_points = {
	f"{agent_name}_cfg_entry_point": f"{agents.__name__}:{agent_name}_cfg.yaml" 
	for agent_name in agent_names
}
entry_points["env_cfg_entry_point"] = f"{__name__}.boxplace_env_cfg:FactoryTaskPegInsertCfg"
gym.register(
    id="BoxPlace-Direct-v0",
    entry_point=f"{__name__}.boxplace_env:BoxPlaceEnv",
    disable_env_checker=True,
    kwargs=entry_points,
)
