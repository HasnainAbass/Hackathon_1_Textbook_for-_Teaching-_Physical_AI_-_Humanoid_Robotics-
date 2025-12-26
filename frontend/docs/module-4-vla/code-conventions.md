# Code Example Formatting and Conventions for VLA Module

## ROS 2 Code Examples

All ROS 2 code examples in this module follow consistent formatting and conventions to ensure clarity and reproducibility.

### Python Code Example Template

```python
#!/usr/bin/env python3
# Example ROS 2 node for VLA system

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class VLAModuleExample(Node):
    def __init__(self):
        super().__init__('vla_example_node')

        # Create subscribers for voice commands
        self.voice_sub = self.create_subscription(
            String,
            'voice_commands',
            self.voice_callback,
            10
        )

        # Create publishers for robot actions
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.get_logger().info('VLA Example Node Initialized')

    def voice_callback(self, msg):
        """Process incoming voice commands"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Process command and execute appropriate action
        self.execute_command(command)

    def execute_command(self, command):
        """Execute the appropriate robot action based on command"""
        if 'forward' in command.lower():
            self.move_forward()
        elif 'backward' in command.lower():
            self.move_backward()
        # Add more command processing as needed

    def move_forward(self):
        """Move robot forward"""
        twist = Twist()
        twist.linear.x = 0.5  # Move forward at 0.5 m/s
        self.cmd_vel_pub.publish(twist)

    def move_backward(self):
        """Move robot backward"""
        twist = Twist()
        twist.linear.x = -0.5  # Move backward at 0.5 m/s
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = VLAModuleExample()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File Example

```xml
<!-- Example launch file for VLA system -->
<launch>
  <!-- Launch the VLA example node -->
  <node pkg="vla_examples"
        exec="vla_example_node"
        name="vla_example"
        output="screen">
  </node>

  <!-- Include other required nodes -->
  <include file="$(find-pkg-share vla_bringup)/launch/robot.launch.py"/>
</launch>
```

## Markdown Code Block Conventions

### Language Specification
- Use specific language identifiers: `python`, `bash`, `yaml`, `xml`, `json`, `cpp`
- Include line numbers when referencing specific lines
- Use appropriate syntax highlighting

### Configuration Examples
- Use YAML for ROS 2 configuration files
- Use JSON for API configurations
- Use INI format for simple key-value configurations

### Command Line Examples
- Prefix terminal commands with `$` for user commands
- Prefix with `#` for administrative commands
- Use `%` for ROS 2 command line tools like `ros2 run`

Example:
```bash
$ ros2 run vla_examples vla_example_node
# ros2 launch vla_examples example.launch.py
% ros2 param set /vla_example param_name value
```

## Simulation Environment Code

For Gazebo/Isaac Sim examples, use the following conventions:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class SimulationVLANode(Node):
    def __init__(self):
        super().__init__('simulation_vla_node')

        # Subscribe to simulated camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            String,
            '/voice_commands',
            self.voice_callback,
            10
        )

        # Publish to simulated robot
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
```

## Error Handling and Safety

All code examples should include appropriate error handling:

```python
def safe_execute_command(self, command):
    """Safely execute command with error handling"""
    try:
        # Validate command
        if not self.validate_command(command):
            self.get_logger().error(f'Invalid command: {command}')
            return False

        # Execute command
        result = self.process_command(command)
        return result

    except Exception as e:
        self.get_logger().error(f'Command execution failed: {e}')
        return False
```

## Testing Code Examples

Include unit test examples where appropriate:

```python
import unittest
from vla_examples.vla_example_node import VLAModuleExample

class TestVLAModule(unittest.TestCase):
    def setUp(self):
        self.node = VLAModuleExample()

    def test_command_processing(self):
        # Test command processing logic
        result = self.node.process_command('move forward')
        self.assertTrue(result)

    def tearDown(self):
        self.node.destroy_node()
```