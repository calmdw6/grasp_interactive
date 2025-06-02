# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
import numpy as np
import time

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        # init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")

def state_machine(count, episode=399 , move_state=180 , open_state=200 , close_state=250 , lift_state=399):
    if count % episode == 0:
        flag = "reset"

    elif count % episode <= move_state:
        flag = "move_to_target"
    
    elif count % episode <= open_state and move_state < count % episode:
        flag = "open_gripper"
    
    elif open_state < count % episode and count % episode <= close_state:
        flag = "close_gripper"
    
    elif close_state < count % episode and count % episode <= episode:
        flag = "lift_to_target"
    
    return flag

def generate_target_pose(object_pose_w, x, y, z, roll, pitch, yaw):
    """
    object_pose_w: The object position and orientation in world frame.

    """
    target_pose_w = torch.zeros_like(object_pose_w[:, :7])
    target_pose_b = torch.zeros_like(object_pose_w[:, :7])
    euler_angles = torch.zeros_like(object_pose_w[:, :3])
    r = torch.empty(object_pose_w.shape[0], device = object_pose_w.device)

    target_pose_b[:, 0] = r.uniform_(*x)
    target_pose_b[:, 1] = r.uniform_(*y)
    target_pose_b[:, 2] = r.uniform_(*z)

    euler_angles[:, 0].uniform_(*roll)
    euler_angles[:, 1].uniform_(*pitch)
    euler_angles[:, 2].uniform_(*yaw)
    quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    target_pose_b[:, 3:7] = quat_unique(quat)

    target_pose_w[:, :3], target_pose_w[:, 3:7] = combine_frame_transforms(
        object_pose_w[:, :3],
        object_pose_w[:, 3:7],
        target_pose_b[:, :3],
        target_pose_b[:, 3:7],
    )

    return target_pose_w

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    object = scene["object"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    start_time = time.time()
    while simulation_app.is_running():
        flag = state_machine(count=count)
        joint_default_pos = robot.data.default_joint_pos.clone()
        joint_default_vel = robot.data.default_joint_vel.clone()
        object_default_state = object.data.default_root_state.clone()
        object_default_state[:, :3] += scene.env_origins
        # reset
        if flag  == "reset":
            # reset time
            count = 0
            robot.set_joint_position_target(joint_default_pos)

            robot.write_joint_state_to_sim(joint_default_pos, joint_default_vel)
            object.write_root_state_to_sim(object_default_state)

            object.reset()
            robot.reset()
            # reset actions
            joint_pos_des = joint_default_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            #np.deg2rad()
            target_pose_w = generate_target_pose(object_pose_w=object.data.root_state_w, x=(0.0, 0.0), y=(0.0, 0.0), z=(0.0, 0.1), roll=(np.deg2rad(180), np.deg2rad(180)), pitch=(np.deg2rad(-50), np.deg2rad(50)), yaw=(np.deg2rad(0), np.deg2rad(0)))


        elif flag == "close_gripper":
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(torch.tensor([-0.025, -0.025], device="cuda"), joint_ids=torch.tensor([7,8], device="cuda"))
        
        elif flag == "open_gripper":
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(torch.tensor([0.025, 0.025], device="cuda"), joint_ids=torch.tensor([7,8], device="cuda"))
        
        elif flag == "move_to_target":
            ik_commands[:] = target_pose_w[:, :7]
            ik_commands[:, :3] -= scene.env_origins
            diff_ik_controller.set_command(ik_commands)
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        elif flag == "lift_to_target":
            ik_commands[:] = target_pose_w[:, :7]
            ik_commands[:, 2] += 0.16
            ik_commands[:, :3] -= scene.env_origins
            diff_ik_controller.set_command(ik_commands)
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            end_time = time.time()
            # print("[time]: ", end_time-start_time)

        else:
            robot.set_joint_position_target(joint_default_pos)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1

        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=1/180, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = GraspSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()