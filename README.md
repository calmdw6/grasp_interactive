# grasp_interactive

This is the interactive environments for grasping task without any components on DRL.

The motivation of this file is mainly for developers needs parallel environments for non-neural network methods in grasping tasks.

## Funtions:
1. Time-based state machine for different stages: reach, close gripper and move.
2. Random target/grasp pose. The generated target pose is relative to the object's pose frame not the world frame!

## Integration to IsaacLab:

Download the `grasp.py` file and copy it under the Isaac Lab direction: `[path-to-IsaacLab]/scripts/grasp/grasp.py`

`for example`: /Home/user0/IsaacLab/scripts/grasp/grasp.py`
