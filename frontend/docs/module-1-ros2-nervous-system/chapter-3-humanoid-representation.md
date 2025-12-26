---
sidebar_position: 4
---

# Chapter 3: Robot Body Representation with URDF

## Learning Objectives

After completing this chapter, you will be able to:
- Explain the purpose of URDF in humanoid robotics
- Define links, joints, and kinematic chains in URDF
- Model humanoid structures and constraints in URDF
- Describe how URDF connects the physical body to ROS 2 control and simulation

## The Purpose of URDF in Humanoid Robotics

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. In humanoid robotics, URDF serves several critical purposes:

1. **Physical Structure Definition**: Defines the physical structure of the robot including links (rigid parts) and joints (connections between links)

2. **Kinematic Model**: Provides the kinematic model necessary for forward and inverse kinematics calculations

3. **Visual and Collision Models**: Specifies how the robot appears visually and how it interacts with the environment in simulation

4. **Integration with ROS Ecosystem**: Enables integration with ROS tools for visualization, simulation, and control

5. **Standardization**: Provides a standardized way to represent robot models across different platforms and applications

URDF is essential for humanoid robots because of their complex multi-joint structures that require precise modeling for effective control and simulation.

## Defining Links, Joints, and Kinematic Chains

### Links

Links represent rigid bodies in the robot structure. Each link has:

- **Physical Properties**: Mass, center of mass, and inertia matrix
- **Visual Properties**: How the link appears in visualization tools
- **Collision Properties**: How the link interacts with the environment in simulation

```xml
<link name="base_link">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.1" length="0.2"/>
    </geometry>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.1" length="0.2"/>
    </geometry>
  </collision>
</link>
```

### Joints

Joints connect links and define their relative motion. Common joint types include:

- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint with limits
- **Fixed**: No motion between links
- **Floating**: 6-DOF motion
- **Planar**: Motion in a plane

```xml
<joint name="base_to_wheel" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>
```

### Kinematic Chains

Kinematic chains are sequences of links connected by joints. In humanoid robots, these represent:

- **Leg chains**: Hip → Knee → Ankle → Foot
- **Arm chains**: Shoulder → Elbow → Wrist → Hand
- **Spine chain**: Multiple vertebrae connections
- **Neck chain**: Connection from torso to head

## Modeling Humanoid Structures and Constraints

### Humanoid-Specific Considerations

Humanoid robots require special modeling considerations due to their human-like structure:

1. **Degrees of Freedom**: Humanoid robots typically have many DOFs to mimic human motion
2. **Balance**: Center of mass and stability considerations
3. **Workspace**: Reachable space for arms and legs
4. **Anthropomorphic Design**: Joint limits and ranges that match human capabilities

### Example: Simple Humanoid URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Root Link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </visual>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <inertial>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <origin xyz="0.15 0 0.8" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>
</robot>
```

### Constraints in Humanoid Modeling

When modeling humanoid structures, consider:

1. **Joint Limits**: Range of motion constraints based on mechanical design
2. **Collision Avoidance**: Preventing self-collision during motion
3. **Center of Mass**: Maintaining balance during locomotion
4. **Actuator Limits**: Torque and speed constraints of real actuators

## How URDF Connects the Physical Body to ROS 2 Control and Simulation

### URDF in ROS 2 Ecosystem

URDF integrates with ROS 2 through several key components:

1. **Robot State Publisher**: Publishes joint states and transforms for visualization
2. **TF2**: Provides coordinate transforms between robot frames
3. **Gazebo/Other Simulators**: Uses URDF for physics simulation
4. **MoveIt!**: Uses URDF for motion planning

### Robot State Publisher Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.broadcaster = TransformBroadcaster(self, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.t = 0.0

    def timer_callback(self):
        # Create joint state message
        msg = JointState()
        msg.name = ['neck_joint', 'left_shoulder_joint']
        msg.position = [math.sin(self.t), math.cos(self.t)]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.joint_pub.publish(msg)
        self.t += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = StatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()
```

### URDF in Simulation

URDF models can be used with simulation environments like Gazebo to:

1. **Test Control Algorithms**: Validate control strategies in a safe environment
2. **Develop Perception Systems**: Test sensors and perception algorithms
3. **Plan Complex Behaviors**: Test whole-body behaviors before deployment
4. **Debug Issues**: Identify problems before testing on real hardware

## Summary

In this chapter, you've learned about URDF and its critical role in humanoid robotics. You now understand how to define links and joints, model humanoid structures with appropriate constraints, and connect URDF models to ROS 2 control and simulation systems.

## Exercises

1. Create a simple URDF model of a robot with at least 3 links and 2 joints.
2. Explain the difference between visual and collision properties in URDF.
3. Describe how joint limits in URDF affect robot control.

## References

- [URDF Documentation](http://wiki.ros.org/urdf)
- [ROS 2 URDF Tutorials](https://docs.ros.org/en/rolling/Tutorials/URDF/Working-with-URDF/)
- [Robot State Publisher](https://github.com/ros/robot_state_publisher)