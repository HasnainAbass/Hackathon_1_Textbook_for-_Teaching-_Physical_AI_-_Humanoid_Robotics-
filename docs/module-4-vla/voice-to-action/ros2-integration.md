# ROS 2 Integration for Voice Commands

## Introduction

ROS 2 integration is the final step in the voice-to-action pipeline, where extracted intents are converted into ROS 2 messages, services, or actions that control the robot. This section covers how to publish structured commands to appropriate ROS 2 topics and execute robot actions based on voice commands.

## ROS 2 Communication Patterns

### Publishers and Subscribers

Voice commands typically result in commands being published to specific topics:

```python
#!/usr/bin/env python3
# ROS 2 node for voice command to ROS 2 action mapping

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from vla_interfaces.msg import Intent  # Custom message type

class VoiceToROS2Mapper(Node):
    def __init__(self):
        super().__init__('voice_to_ros2_mapper')

        # Subscribe to extracted intents
        self.intent_sub = self.create_subscription(
            Intent,
            'extracted_intent',
            self.intent_callback,
            10
        )

        # Publishers for different robot actions
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.voice_feedback_pub = self.create_publisher(
            String,
            'voice_feedback',
            10
        )

        self.get_logger().info('Voice to ROS 2 Mapper Initialized')

    def intent_callback(self, msg):
        """Process extracted intent and publish ROS 2 commands"""
        try:
            intent = msg.action
            confidence = msg.confidence
            parameters = msg.parameters

            # Validate intent confidence
            if confidence < 0.7:
                self.get_logger().warn(f'Low confidence intent: {confidence}')
                self.publish_feedback(f'Command unclear, confidence: {confidence:.2f}')
                return

            # Map intent to ROS 2 command
            if intent == 'navigation':
                self.handle_navigation_intent(parameters)
            elif intent == 'manipulation':
                self.handle_manipulation_intent(parameters)
            elif intent == 'information':
                self.handle_information_intent(parameters)
            else:
                self.get_logger().warn(f'Unknown intent: {intent}')
                self.publish_feedback(f'Unknown command: {intent}')

        except Exception as e:
            self.get_logger().error(f'Error in intent processing: {e}')

    def handle_navigation_intent(self, parameters):
        """Handle navigation intents"""
        cmd = Twist()

        # Parse parameters for navigation
        for param in parameters:
            if param.name == 'direction':
                direction = param.value.lower()
                if direction == 'forward':
                    cmd.linear.x = 0.5  # m/s
                elif direction == 'backward':
                    cmd.linear.x = -0.5
                elif direction == 'left':
                    cmd.angular.z = 0.5  # rad/s
                elif direction == 'right':
                    cmd.angular.z = -0.5
            elif param.name == 'distance':
                # For more complex navigation, use navigation2
                pass

        # Publish the command
        self.cmd_vel_pub.publish(cmd)
        self.publish_feedback('Moving as requested')

    def handle_manipulation_intent(self, parameters):
        """Handle manipulation intents"""
        # For manipulation, we might need to call services or use actions
        # This is a simplified example
        for param in parameters:
            if param.name == 'object':
                object_name = param.value
                self.publish_feedback(f'Attempting to manipulate {object_name}')

    def publish_feedback(self, message):
        """Publish feedback to user"""
        feedback_msg = String()
        feedback_msg.data = message
        self.voice_feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceToROS2Mapper()

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

## Service Calls

For more complex operations, voice commands can trigger ROS 2 service calls:

```python
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger  # Example service

class VoiceServiceCaller(Node):
    def __init__(self):
        super().__init__('voice_service_caller')

        # Create clients for services
        self.emergency_stop_client = self.create_client(
            Trigger,
            'emergency_stop'
        )

        self.home_position_client = self.create_client(
            Trigger,
            'move_to_home'
        )

        # Wait for services to be available
        while not self.emergency_stop_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Emergency stop service not available, waiting...')

        while not self.home_position_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Home position service not available, waiting...')

    def call_emergency_stop(self):
        """Call emergency stop service"""
        request = Trigger.Request()
        future = self.emergency_stop_client.call_async(request)
        return future

    def call_home_position(self):
        """Call move to home position service"""
        request = Trigger.Request()
        future = self.home_position_client.call_async(request)
        return future
```

## Action Clients

For long-running tasks, use ROS 2 actions:

```python
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class VoiceActionClient(Node):
    def __init__(self):
        super().__init__('voice_action_client')

        # Create action clients
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

    def send_navigation_goal(self, x, y, theta):
        """Send navigation goal based on voice command"""
        goal_msg = NavigateToPose.Goal()

        # Set the goal pose
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        goal_msg.pose.pose.orientation.x = quat[0]
        goal_msg.pose.pose.orientation.y = quat[1]
        goal_msg.pose.pose.orientation.z = quat[2]
        goal_msg.pose.pose.orientation.w = quat[3]

        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return

        # Send goal
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )

        send_goal_future.add_done_callback(self.nav_goal_response_callback)

    def nav_goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return

        self.get_logger().info('Navigation goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.nav_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        self.get_logger().info(f'Navigation feedback: {feedback_msg.feedback}')
```

## Message Types for VLA Systems

### Custom Message Definitions

Create custom message types for VLA-specific data:

```text
# vla_interfaces/msg/VoiceCommand.msg
string command_text
string intent
float32 confidence
Parameter[] parameters
builtin_interfaces/Time timestamp

# vla_interfaces/msg/Parameter.msg
string name
string value

# vla_interfaces/msg/VoiceFeedback.msg
string message
string status  # 'success', 'error', 'in_progress', 'cancelled'
builtin_interfaces/Time timestamp
```

### Common ROS 2 Message Types for Voice Commands

```python
# Navigation commands
from geometry_msgs.msg import Twist, PoseStamped

# Manipulation commands
from std_msgs.msg import String
from sensor_msgs.msg import JointState

# System commands
from std_srvs.srv import Trigger, SetBool

# Custom messages
from vla_interfaces.msg import VoiceCommand, Intent
```

## Integration Patterns

### Voice Command Router

A central node that routes voice commands to appropriate handlers:

```python
class VoiceCommandRouter(Node):
    def __init__(self):
        super().__init__('voice_command_router')

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            VoiceCommand,
            'voice_commands',
            self.voice_command_callback,
            10
        )

        # Publishers for different subsystems
        self.nav_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.manip_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.system_pub = self.create_publisher(String, 'system_commands', 10)

    def voice_command_callback(self, msg):
        """Route voice command to appropriate subsystem"""
        intent = msg.intent

        if intent in ['navigation', 'move', 'go', 'drive']:
            self.route_to_navigation(msg)
        elif intent in ['manipulate', 'pick', 'place', 'grasp']:
            self.route_to_manipulation(msg)
        elif intent in ['system', 'status', 'stop', 'start']:
            self.route_to_system(msg)
        else:
            self.handle_unknown_intent(msg)

    def route_to_navigation(self, voice_cmd):
        """Route navigation commands"""
        # Convert voice command to navigation command
        nav_cmd = self.convert_to_navigation(voice_cmd)
        self.nav_pub.publish(nav_cmd)

    def route_to_manipulation(self, voice_cmd):
        """Route manipulation commands"""
        # Convert voice command to manipulation command
        manip_cmd = self.convert_to_manipulation(voice_cmd)
        self.manip_pub.publish(manip_cmd)

    def route_to_system(self, voice_cmd):
        """Route system commands"""
        # Convert voice command to system command
        system_cmd = self.convert_to_system(voice_cmd)
        self.system_pub.publish(system_cmd)
```

## Safety Considerations

### Command Validation

Always validate voice commands before execution:

```python
class SafeVoiceCommandHandler:
    def __init__(self):
        self.safety_limits = {
            'max_linear_velocity': 1.0,  # m/s
            'max_angular_velocity': 1.0,  # rad/s
            'max_distance': 10.0,  # meters
            'max_manipulation_force': 50.0  # Newtons
        }

    def validate_command(self, intent_msg):
        """Validate command against safety limits"""
        if intent_msg.action == 'navigation':
            for param in intent_msg.parameters:
                if param.name == 'distance':
                    distance = float(param.value)
                    if distance > self.safety_limits['max_distance']:
                        return False, f'Distance {distance} exceeds limit {self.safety_limits["max_distance"]}'

                elif param.name == 'velocity':
                    velocity = float(param.value)
                    if abs(velocity) > self.safety_limits['max_linear_velocity']:
                        return False, f'Velocity {velocity} exceeds limit {self.safety_limits["max_linear_velocity"]}'

        return True, "Command is safe"
```

### Emergency Handling

Implement emergency stop capabilities:

```python
class EmergencyHandler(Node):
    def __init__(self):
        super().__init__('emergency_handler')

        # Emergency stop publisher
        self.emergency_pub = self.create_publisher(
            String,
            'emergency_stop',
            10
        )

        # Emergency stop service
        self.emergency_service = self.create_service(
            Trigger,
            'voice_emergency_stop',
            self.emergency_stop_callback
        )

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop request"""
        emergency_msg = String()
        emergency_msg.data = 'EMERGENCY_STOP'
        self.emergency_pub.publish(emergency_msg)

        response.success = True
        response.message = 'Emergency stop activated'
        return response
```

## Performance Optimization

### Message Queues

Configure appropriate QoS settings for voice command processing:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class OptimizedVoiceNode(Node):
    def __init__(self):
        super().__init__('optimized_voice_node')

        # Define QoS for voice commands (real-time priority)
        voice_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribe with optimized QoS
        self.voice_sub = self.create_subscription(
            VoiceCommand,
            'voice_commands',
            self.voice_callback,
            voice_qos
        )
```

### Threading Considerations

Handle voice processing in separate threads when needed:

```python
import threading
from concurrent.futures import ThreadPoolExecutor

class ThreadedVoiceHandler(Node):
    def __init__(self):
        super().__init__('threaded_voice_handler')

        # Create thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            VoiceCommand,
            'voice_commands',
            self.voice_callback,
            10
        )

    def voice_callback(self, msg):
        """Process voice command in separate thread"""
        future = self.executor.submit(self.process_voice_command, msg)
        future.add_done_callback(self.voice_processing_complete)

    def process_voice_command(self, voice_cmd):
        """Process voice command in background thread"""
        # Perform intensive processing (STT, NLP, etc.)
        result = self.extract_intent(voice_cmd.command_text)
        return result

    def voice_processing_complete(self, future):
        """Handle completion of voice processing"""
        try:
            result = future.result()
            # Publish result to ROS 2
            self.publish_result(result)
        except Exception as e:
            self.get_logger().error(f'Voice processing failed: {e}')
```

## Error Handling and Feedback

### Command Execution Feedback

Provide feedback on command execution:

```python
class FeedbackPublisher(Node):
    def __init__(self):
        super().__init__('feedback_publisher')

        self.feedback_pub = self.create_publisher(
            VoiceFeedback,
            'voice_feedback',
            10
        )

    def publish_command_status(self, command_id, status, message):
        """Publish command execution status"""
        feedback_msg = VoiceFeedback()
        feedback_msg.command_id = command_id
        feedback_msg.status = status
        feedback_msg.message = message
        feedback_msg.timestamp = self.get_clock().now().to_msg()

        self.feedback_pub.publish(feedback_msg)

    def publish_success(self, command_id, message="Command executed successfully"):
        """Publish success feedback"""
        self.publish_command_status(command_id, 'success', message)

    def publish_error(self, command_id, error_message):
        """Publish error feedback"""
        self.publish_command_status(command_id, 'error', error_message)
```

## Integration with Simulation

### Gazebo Integration

When using simulation, ensure proper integration with Gazebo:

```python
class SimulationVoiceHandler(Node):
    def __init__(self):
        super().__init__('simulation_voice_handler')

        # Publishers for simulated robot
        self.sim_cmd_vel_pub = self.create_publisher(
            Twist,
            '/robot/cmd_vel',
            10
        )

        # Publishers for simulated sensors
        self.sim_feedback_pub = self.create_publisher(
            String,
            '/robot/voice_feedback',
            10
        )

    def handle_simulation_navigation(self, intent_params):
        """Handle navigation in simulation environment"""
        cmd = Twist()

        # Apply simulation-specific parameters
        for param in intent_params:
            if param.name == 'direction':
                direction = param.value.lower()
                if direction == 'forward':
                    cmd.linear.x = 0.3  # Slower in simulation
                elif direction == 'turn':
                    cmd.angular.z = 0.3

        self.sim_cmd_vel_pub.publish(cmd)
```

## Testing Voice-to-ROS 2 Integration

### Unit Testing

```python
import unittest
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

class TestVoiceToROS2Integration(unittest.TestCase):
    def setUp(self):
        self.node = VoiceToROS2Mapper()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def test_navigation_intent_mapping(self):
        """Test that navigation intent maps to Twist message"""
        # Create test intent message
        intent_msg = Intent()
        intent_msg.action = 'navigation'
        intent_msg.confidence = 0.9

        param = Intent.Parameter()
        param.name = 'direction'
        param.value = 'forward'
        intent_msg.parameters.append(param)

        # Test that it publishes a Twist message
        # This would involve creating a mock publisher and checking published messages
        pass

    def tearDown(self):
        self.node.destroy_node()
        self.executor.shutdown()
```

### Integration Testing

```python
def test_end_to_end_voice_pipeline():
    """Test complete voice-to-action pipeline"""
    # 1. Publish voice command
    # 2. Verify intent extraction
    # 3. Verify ROS 2 command publishing
    # 4. Verify robot action execution
    pass
```

## Best Practices

1. **Always validate commands** before execution to ensure safety
2. **Provide clear feedback** to users about command status
3. **Use appropriate QoS settings** for real-time voice processing
4. **Implement error handling** for all integration points
5. **Test thoroughly** in both simulation and real environments
6. **Monitor performance** to ensure timely command processing

The next sections will cover code examples for voice command processing in simulation and acceptance scenarios.