# Code Examples for Voice Command Processing in Simulation

## Introduction

This section provides practical code examples for implementing voice-to-action interfaces in simulation environments. These examples demonstrate how to integrate speech-to-text, intent extraction, and ROS 2 communication in a complete voice command processing pipeline.

## Complete Voice Command Processing Pipeline

### Main Voice Processing Node

```python
#!/usr/bin/env python3
"""
Complete voice command processing pipeline for simulation environments.
This node integrates speech-to-text, intent extraction, and ROS 2 integration.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vla_interfaces.msg import Intent
import whisper
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import time


class VoiceCommandProcessorNode(Node):
    def __init__(self):
        super().__init__('voice_command_processor')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("small")
        self.get_logger().info('Whisper model loaded successfully')

        # Initialize thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        # QoS profile for real-time processing
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.feedback_pub = self.create_publisher(String, 'voice_feedback', qos_profile)
        self.intent_pub = self.create_publisher(Intent, 'extracted_intent', qos_profile)

        # Subscribers
        self.voice_command_sub = self.create_subscription(
            String,
            'simulated_voice_input',
            self.voice_command_callback,
            qos_profile
        )

        # Processing state
        self.is_processing = False

        self.get_logger().info('Voice Command Processor Node initialized')

    def voice_command_callback(self, msg):
        """Handle incoming voice commands (simulated in this example)"""
        if self.is_processing:
            self.get_logger().warn('Already processing a command, ignoring new command')
            return

        # Process command in separate thread to avoid blocking
        future = self.executor.submit(self.process_voice_command, msg.data)
        future.add_done_callback(self.processing_complete)

    def process_voice_command(self, command_text):
        """Process voice command: STT → Intent Extraction → ROS 2 Action"""
        self.is_processing = True
        start_time = time.time()

        try:
            self.get_logger().info(f'Processing command: "{command_text}"')

            # Simulate speech-to-text (in real implementation, this would process audio)
            transcribed_text = command_text.lower().strip()
            self.get_logger().info(f'Transcribed: {transcribed_text}')

            # Extract intent from transcribed text
            intent_data = self.extract_intent(transcribed_text)

            if intent_data['confidence'] < 0.7:
                self.get_logger().warn(f'Low confidence intent: {intent_data["confidence"]}')
                self.publish_feedback(f'Command unclear, confidence: {intent_data["confidence"]:.2f}')
                return

            # Publish extracted intent
            self.publish_intent(intent_data)

            # Execute action based on intent
            self.execute_action(intent_data)

            processing_time = time.time() - start_time
            self.get_logger().info(f'Command processed in {processing_time:.2f}s')

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')
            self.publish_feedback(f'Error processing command: {e}')

        finally:
            self.is_processing = False

    def extract_intent(self, text):
        """Extract intent from transcribed text using rule-based approach"""
        text = text.lower()

        # Navigation intents
        if any(word in text for word in ['forward', 'move forward', 'go forward']):
            return {
                'intent': 'navigation',
                'action': 'move_forward',
                'parameters': {'distance': '2', 'direction': 'forward'},
                'confidence': 0.9
            }
        elif any(word in text for word in ['backward', 'move backward', 'go backward']):
            return {
                'intent': 'navigation',
                'action': 'move_backward',
                'parameters': {'distance': '2', 'direction': 'backward'},
                'confidence': 0.9
            }
        elif any(word in text for word in ['left', 'turn left', 'rotate left']):
            return {
                'intent': 'navigation',
                'action': 'turn_left',
                'parameters': {'angle': '90', 'direction': 'left'},
                'confidence': 0.9
            }
        elif any(word in text for word in ['right', 'turn right', 'rotate right']):
            return {
                'intent': 'navigation',
                'action': 'turn_right',
                'parameters': {'angle': '90', 'direction': 'right'},
                'confidence': 0.9
            }
        elif any(word in text for word in ['stop', 'halt', 'pause']):
            return {
                'intent': 'system',
                'action': 'stop',
                'parameters': {},
                'confidence': 0.95
            }
        elif any(word in text for word in ['red object', 'red ball', 'red item']):
            return {
                'intent': 'manipulation',
                'action': 'pick_red_object',
                'parameters': {'object': 'red object'},
                'confidence': 0.85
            }

        # Default unknown intent
        return {
            'intent': 'unknown',
            'action': 'unknown',
            'parameters': {},
            'confidence': 0.0
        }

    def execute_action(self, intent_data):
        """Execute ROS 2 action based on extracted intent"""
        action = intent_data['action']

        if action == 'move_forward':
            self.move_forward(intent_data['parameters'])
        elif action == 'move_backward':
            self.move_backward(intent_data['parameters'])
        elif action == 'turn_left':
            self.turn_left(intent_data['parameters'])
        elif action == 'turn_right':
            self.turn_right(intent_data['parameters'])
        elif action == 'stop':
            self.stop_robot()
        elif action == 'pick_red_object':
            self.pick_red_object(intent_data['parameters'])
        else:
            self.get_logger().warn(f'Unknown action: {action}')
            self.publish_feedback(f'Unknown command: {action}')

    def move_forward(self, params):
        """Move robot forward"""
        cmd = Twist()
        distance = float(params.get('distance', '1'))
        cmd.linear.x = 0.5  # m/s

        # Publish command for duration (simulated)
        self.get_logger().info(f'Moving forward {distance} meters')
        self.cmd_vel_pub.publish(cmd)
        self.publish_feedback(f'Moving forward {distance} meters')

        # Stop after movement
        time.sleep(distance / 0.5)
        self.stop_robot()

    def move_backward(self, params):
        """Move robot backward"""
        cmd = Twist()
        distance = float(params.get('distance', '1'))
        cmd.linear.x = -0.5  # m/s

        self.get_logger().info(f'Moving backward {distance} meters')
        self.cmd_vel_pub.publish(cmd)
        self.publish_feedback(f'Moving backward {distance} meters')

        time.sleep(distance / 0.5)
        self.stop_robot()

    def turn_left(self, params):
        """Turn robot left"""
        cmd = Twist()
        angle = float(params.get('angle', '90'))
        cmd.angular.z = 0.5  # rad/s

        self.get_logger().info(f'Turning left {angle} degrees')
        self.cmd_vel_pub.publish(cmd)
        self.publish_feedback(f'Turning left {angle} degrees')

        time.sleep(angle / 180)  # Approximate time for turn
        self.stop_robot()

    def turn_right(self, params):
        """Turn robot right"""
        cmd = Twist()
        angle = float(params.get('angle', '90'))
        cmd.angular.z = -0.5  # rad/s

        self.get_logger().info(f'Turning right {angle} degrees')
        self.cmd_vel_pub.publish(cmd)
        self.publish_feedback(f'Turning right {angle} degrees')

        time.sleep(angle / 180)
        self.stop_robot()

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.publish_feedback('Robot stopped')

    def pick_red_object(self, params):
        """Simulate picking red object"""
        self.get_logger().info('Attempting to pick red object')
        self.publish_feedback('Attempting to pick red object')
        # In simulation, this would trigger object manipulation

    def publish_intent(self, intent_data):
        """Publish extracted intent to ROS 2"""
        intent_msg = Intent()
        intent_msg.action = intent_data['action']
        intent_msg.confidence = intent_data['confidence']

        # Add parameters
        for name, value in intent_data['parameters'].items():
            param = Intent.Parameter()
            param.name = name
            param.value = str(value)
            intent_msg.parameters.append(param)

        self.intent_pub.publish(intent_msg)

    def publish_feedback(self, message):
        """Publish feedback to user"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

    def processing_complete(self, future):
        """Handle completion of voice processing"""
        try:
            result = future.result()
            self.get_logger().info('Voice command processing completed')
        except Exception as e:
            self.get_logger().error(f'Voice processing failed: {e}')


def main(args=None):
    rclpy.init(args=args)

    node = VoiceCommandProcessorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down voice command processor')
    finally:
        node.executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File for Voice Command Processing

```xml
<!-- launch/voice_command_processing.launch.py -->
<launch>
  <!-- Launch the voice command processor -->
  <node pkg="vla_examples"
        exec="voice_command_processor"
        name="voice_command_processor"
        output="screen">
  </node>

  <!-- Launch simulation environment -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gzserver.launch.py">
    <arg name="world" value="$(find-pkg-share vla_examples)/worlds/voice_demo.world"/>
  </include>

  <include file="$(find-pkg-share gazebo_ros)/launch/gzclient.launch.py"/>

  <!-- Launch robot controller -->
  <node pkg="controller_manager"
        exec="spawner"
        args="diff_drive_controller joint_state_broadcaster">
  </node>
</launch>
```

## Voice Command Simulator Node

For testing purposes, create a simulator that sends voice commands:

```python
#!/usr/bin/env python3
"""
Voice command simulator for testing the voice processing pipeline.
This node simulates voice commands in a controlled environment.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import random


class VoiceCommandSimulator(Node):
    def __init__(self):
        super().__init__('voice_command_simulator')

        self.publisher = self.create_publisher(
            String,
            'simulated_voice_input',
            10
        )

        # Define test commands
        self.test_commands = [
            "Move forward 2 meters",
            "Turn left 90 degrees",
            "Go to the kitchen",
            "Pick up the red ball",
            "Stop the robot",
            "Move backward 1 meter",
            "Turn right 45 degrees"
        ]

        # Timer to send commands periodically
        self.timer = self.create_timer(5.0, self.send_random_command)

        self.get_logger().info('Voice Command Simulator initialized')

    def send_random_command(self):
        """Send a random voice command for testing"""
        if self.test_commands:
            command = random.choice(self.test_commands)
            msg = String()
            msg.data = command

            self.publisher.publish(msg)
            self.get_logger().info(f'Sent simulated command: "{command}"')


def main(args=None):
    rclpy.init(args=args)

    node = VoiceCommandSimulator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down voice command simulator')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Integration Test Example

```python
#!/usr/bin/env python3
"""
Integration test for voice command processing pipeline.
Tests the complete flow from voice input to robot action.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vla_interfaces.msg import Intent
import time
import threading


class VoiceIntegrationTest(Node):
    def __init__(self):
        super().__init__('voice_integration_test')

        # Publishers
        self.voice_pub = self.create_publisher(String, 'simulated_voice_input', 10)

        # Subscribers
        self.feedback_sub = self.create_subscription(
            String,
            'voice_feedback',
            self.feedback_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.intent_sub = self.create_subscription(
            Intent,
            'extracted_intent',
            self.intent_callback,
            10
        )

        # Test results
        self.received_feedback = []
        self.received_cmd_vel = []
        self.received_intents = []

        # Test commands
        self.test_commands = [
            ("Move forward 1 meter", "navigation"),
            ("Turn left", "navigation"),
            ("Stop", "system")
        ]

        self.get_logger().info('Voice Integration Test initialized')

    def feedback_callback(self, msg):
        """Handle feedback messages"""
        self.received_feedback.append(msg.data)
        self.get_logger().info(f'Received feedback: {msg.data}')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.received_cmd_vel.append((msg.linear.x, msg.angular.z))
        self.get_logger().info(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')

    def intent_callback(self, msg):
        """Handle extracted intents"""
        self.received_intents.append(msg.action)
        self.get_logger().info(f'Received intent: {msg.action}')

    def run_test(self):
        """Run the integration test"""
        self.get_logger().info('Starting voice integration test...')

        # Send test commands with delays
        for command_text, expected_intent in self.test_commands:
            self.get_logger().info(f'Sending command: "{command_text}"')

            # Clear previous results
            self.received_feedback.clear()
            self.received_cmd_vel.clear()
            self.received_intents.clear()

            # Send command
            cmd_msg = String()
            cmd_msg.data = command_text
            self.voice_pub.publish(cmd_msg)

            # Wait for processing
            time.sleep(3.0)

            # Check results
            success = self.verify_results(expected_intent)
            if success:
                self.get_logger().info(f'✓ Test passed for: "{command_text}"')
            else:
                self.get_logger().error(f'✗ Test failed for: "{command_text}"')

    def verify_results(self, expected_intent):
        """Verify that the correct intent was processed"""
        # Check if intent was extracted
        if not self.received_intents:
            self.get_logger().error('No intents received')
            return False

        # Check if expected intent was received
        if expected_intent in self.received_intents:
            return True

        self.get_logger().error(f'Expected intent {expected_intent}, got {self.received_intents}')
        return False


def main(args=None):
    rclpy.init(args=args)

    test_node = VoiceIntegrationTest()

    # Run test in separate thread to allow ROS spinning
    def run_test():
        time.sleep(2.0)  # Wait for connections
        test_node.run_test()
        rclpy.shutdown()

    test_thread = threading.Thread(target=run_test)
    test_thread.start()

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info('Shutting down test')
    finally:
        test_node.destroy_node()
        test_thread.join()


if __name__ == '__main__':
    main()
```

## Configuration File

Create a configuration file for voice processing parameters:

```yaml
# config/voice_processing.yaml
voice_command_processor:
  ros__parameters:
    # Whisper model configuration
    whisper_model_size: "small"
    whisper_language: "en"

    # Processing parameters
    confidence_threshold: 0.7
    max_processing_time: 5.0  # seconds

    # Safety parameters
    max_linear_velocity: 1.0
    max_angular_velocity: 1.0
    emergency_stop_timeout: 10.0

    # Audio parameters (for real implementation)
    audio_sample_rate: 16000
    audio_chunk_size: 1024
    audio_buffer_size: 4096
```

## CMakeLists.txt for Package

```cmake
cmake_minimum_required(VERSION 3.8)
project(vla_examples)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vla_interfaces REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install worlds for simulation
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/voice_command_processor
  scripts/voice_command_simulator
  scripts/voice_integration_test
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

## Package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>vlc_examples</name>
  <version>1.0.0</version>
  <description>Voice command processing examples for VLA systems</description>
  <maintainer email="robotics@example.com">Robotics Team</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav2_msgs</depend>
  <depend>example_interfaces</depend>
  <depend>sensor_msgs</depend>
  <depend>builtin_interfaces</depend>
  <depend>tf_transformations</depend>

  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>openai-whisper</exec_depend>
  <exec_depend>spacy</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Testing the Implementation

### Running the Voice Command Processing Pipeline

1. **Build the package:**
   ```bash
   colcon build --packages-select vla_examples
   source install/setup.bash
   ```

2. **Run the simulation:**
   ```bash
   ros2 launch vla_examples voice_command_processing.launch.py
   ```

3. **Send test commands:**
   ```bash
   ros2 topic pub /simulated_voice_input std_msgs/String "data: 'Move forward 2 meters'"
   ```

4. **Monitor feedback:**
   ```bash
   ros2 topic echo /voice_feedback
   ```

## Performance Considerations

### Real-time Processing Requirements

For real-time voice processing in simulation:

- **Processing Latency**: Keep processing time under 200ms for responsive interaction
- **Thread Management**: Use thread pools to handle concurrent processing
- **Memory Management**: Monitor memory usage, especially with Whisper models
- **CPU Utilization**: Optimize for available computational resources

### Resource Optimization

```python
# Example of resource-optimized processing
class OptimizedVoiceProcessor:
    def __init__(self):
        # Use smaller model for resource-constrained environments
        self.model = whisper.load_model("tiny")

        # Implement caching for common phrases
        self.intent_cache = {}
        self.cache_size = 50
```

These examples provide a complete implementation of voice command processing in simulation environments, following the requirements for the VLA module. The code demonstrates the complete pipeline from voice input to robot action execution, with proper error handling, safety considerations, and testing capabilities.