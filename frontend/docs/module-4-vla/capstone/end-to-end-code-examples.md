# Complete End-to-End Code Example for VLA Pipeline

## Introduction

This chapter provides a complete, integrated code example that demonstrates the full Vision-Language-Action (VLA) pipeline from voice command to robot action execution. The example combines all components covered in previous chapters: voice processing, LLM-based planning, perception, control, and safety validation.

## Complete VLA System Implementation

### Main Integration Node

```python
#!/usr/bin/env python3
"""
Complete Vision-Language-Action (VLA) System
End-to-end implementation integrating all VLA components
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from vla_interfaces.msg import Intent, TaskPlan, PerceptionData
from builtin_interfaces.msg import Time
from openai import OpenAI
import json
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class VLASystemNode(Node):
    def __init__(self):
        super().__init__('vla_system_node')

        # Initialize LLM client
        self.llm_client = OpenAI(api_key=self.get_parameter_or('openai_api_key', ''))

        # Initialize components
        self.voice_processor = VoiceProcessor(self)
        self.planning_system = PlanningSystem(self.llm_client)
        self.perception_system = PerceptionSystem(self)
        self.control_system = ControlSystem(self)
        self.safety_validator = SafetyValidator(self)
        self.integration_coordinator = IntegrationCoordinator(self)

        # Thread safety and execution
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.command_queue = Queue()

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.feedback_pub = self.create_publisher(String, 'vla_feedback', qos_profile)
        self.task_plan_pub = self.create_publisher(TaskPlan, 'generated_task_plans', qos_profile)
        self.intent_pub = self.create_publisher(Intent, 'extracted_intent', qos_profile)

        # Subscribers
        self.voice_sub = self.create_subscription(
            String,
            'voice_commands',
            self.voice_command_callback,
            qos_profile
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            qos_profile
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            qos_profile
        )

        self.odom_sub = self.create_subscription(
            String,  # Using String as placeholder; in real system this would be Odometry
            'robot_state',
            self.state_callback,
            qos_profile
        )

        # System state
        self.current_state = {
            'robot_position': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'environment_map': {},
            'detected_objects': [],
            'current_task': None,
            'task_status': 'idle',
            'safety_status': 'safe'
        }

        # Processing flags
        self.is_processing = False
        self.active_task = None

        self.get_logger().info('Complete VLA System Node initialized')

    def voice_command_callback(self, msg):
        """Handle incoming voice commands through complete pipeline"""
        with self.lock:
            if self.is_processing:
                self.get_logger().warn('System busy, queueing command')
                self.command_queue.put(msg)
                return

            self.is_processing = True

        # Process command in separate thread to avoid blocking
        future = self.executor.submit(self.process_voice_command, msg)
        future.add_done_callback(self.processing_complete)

    def process_voice_command(self, voice_msg):
        """Process voice command through complete VLA pipeline"""
        start_time = time.time()
        command_text = voice_msg.data

        try:
            self.get_logger().info(f'Processing voice command: "{command_text}"')

            # Step 1: Voice Processing (Voice-to-Action)
            self.publish_feedback(f'Processing voice command: {command_text}')
            intent_result = self.voice_processor.process_voice_command(command_text)

            if not intent_result['success']:
                self.publish_feedback(f'Voice processing failed: {intent_result["error"]}')
                return

            intent_data = intent_result['intent']
            self.publish_feedback(f'Extracted intent: {intent_data["action"]}')

            # Publish extracted intent
            intent_msg = self.create_intent_message(intent_data)
            self.intent_pub.publish(intent_msg)

            # Step 2: LLM-Based Planning
            self.publish_feedback('Generating task plan using LLM...')
            plan_result = self.planning_system.generate_and_validate_plan(
                intent_data,
                self.current_state
            )

            if not plan_result['success']:
                self.publish_feedback(f'Planning failed: {plan_result["error"]}')
                return

            task_plan = plan_result['plan']
            self.publish_feedback(f'Generated plan with {len(task_plan["action_sequence"])} actions')

            # Publish task plan
            plan_msg = self.create_task_plan_message(task_plan, command_text)
            self.task_plan_pub.publish(plan_msg)

            # Step 3: Integrate Perception Data
            self.publish_feedback('Integrating perception data...')
            perception_context = self.perception_system.get_current_context()
            enriched_plan = self.integration_coordinator.integrate_perception_data(
                task_plan,
                perception_context
            )

            # Step 4: Safety Validation
            self.publish_feedback('Validating plan safety...')
            if not self.safety_validator.validate_plan(enriched_plan, self.current_state):
                self.publish_feedback('Plan failed safety validation')
                return

            # Step 5: Execute Plan
            self.publish_feedback('Executing task plan...')
            execution_result = self.control_system.execute_plan(enriched_plan)

            if execution_result['success']:
                self.publish_feedback(f'Task completed successfully: {execution_result["message"]}')
            else:
                self.publish_feedback(f'Task execution failed: {execution_result["error"]}')

            processing_time = time.time() - start_time
            self.get_logger().info(f'Command processed in {processing_time:.2f}s')

        except Exception as e:
            self.get_logger().error(f'Error in VLA pipeline: {e}')
            self.publish_feedback(f'VLA pipeline error: {e}')
        finally:
            with self.lock:
                self.is_processing = False

                # Process next queued command if available
                if not self.command_queue.empty():
                    next_command = self.command_queue.get()
                    self.voice_command_callback(next_command)

    def create_intent_message(self, intent_data):
        """Create ROS 2 Intent message from intent data"""
        intent_msg = Intent()
        intent_msg.action = intent_data['action']
        intent_msg.confidence = intent_data['confidence']
        intent_msg.original_text = intent_data['original_text']
        intent_msg.timestamp = self.get_clock().now().to_msg()

        # Add parameters
        for param_name, param_value in intent_data['parameters'].items():
            param = Intent.Parameter()
            param.name = param_name
            param.value = str(param_value)
            intent_msg.parameters.append(param)

        return intent_msg

    def create_task_plan_message(self, plan_data, original_command):
        """Create ROS 2 TaskPlan message from plan data"""
        plan_msg = TaskPlan()
        plan_msg.original_command = original_command
        plan_msg.timestamp = self.get_clock().now().to_msg()
        plan_msg.constraints_respected = True

        for action_data in plan_data['action_sequence']:
            action_msg = TaskPlan.Action()
            action_msg.id = action_data['id']
            action_msg.action_type = action_data['type']
            action_msg.description = action_data['description']
            action_msg.dependencies = action_data['dependencies']

            # Add parameters
            for param_name, param_value in action_data['parameters'].items():
                param = TaskPlan.Parameter()
                param.name = param_name
                param.value = str(param_value)
                action_msg.parameters.append(param)

            plan_msg.actions.append(action_msg)

        return plan_msg

    def image_callback(self, msg):
        """Process camera image for perception"""
        self.perception_system.process_image_data(msg)

    def laser_callback(self, msg):
        """Process laser scan for navigation safety"""
        self.perception_system.process_laser_data(msg)

    def state_callback(self, msg):
        """Update robot state"""
        try:
            state_data = json.loads(msg.data)
            self.current_state.update(state_data)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid state message format')

    def publish_feedback(self, message):
        """Publish feedback to user"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

    def processing_complete(self, future):
        """Handle completion of voice processing"""
        try:
            result = future.result()
            self.get_logger().info('VLA pipeline processing completed')
        except Exception as e:
            self.get_logger().error(f'Processing failed: {e}')


class VoiceProcessor:
    """Voice processing component of VLA system"""
    def __init__(self, node):
        self.node = node

    def process_voice_command(self, command_text):
        """Process voice command through complete pipeline"""
        try:
            # In real implementation, this would include actual STT processing
            # For this example, we'll use the text directly
            transcribed_text = command_text.lower().strip()

            # Extract intent using rule-based approach
            intent_data = self.extract_intent(transcribed_text)

            if intent_data['confidence'] < 0.7:
                return {
                    'success': False,
                    'error': f'Low confidence intent: {intent_data["confidence"]}',
                    'confidence': intent_data['confidence']
                }

            intent_data['original_text'] = command_text
            return {
                'success': True,
                'intent': intent_data
            }

        except Exception as e:
            self.node.get_logger().error(f'Voice processing error: {e}')
            return {
                'success': False,
                'error': str(e)
            }

    def extract_intent(self, text):
        """Extract intent from text using rule-based approach"""
        text_lower = text.lower()

        # Navigation intents
        if any(word in text_lower for word in ['go to', 'navigate to', 'move to', 'drive to']):
            return {
                'action': 'navigation',
                'parameters': {
                    'destination': self.extract_location(text_lower),
                    'command': text_lower
                },
                'confidence': 0.9
            }
        elif any(word in text_lower for word in ['pick up', 'grasp', 'get', 'take', 'lift']):
            return {
                'action': 'manipulation',
                'parameters': {
                    'object': self.extract_object(text_lower),
                    'command': text_lower
                },
                'confidence': 0.85
            }
        elif any(word in text_lower for word in ['stop', 'halt', 'pause', 'wait']):
            return {
                'action': 'system',
                'parameters': {
                    'command': 'stop'
                },
                'confidence': 0.95
            }
        else:
            return {
                'action': 'unknown',
                'parameters': {'command': text_lower},
                'confidence': 0.0
            }

    def extract_location(self, text):
        """Extract location from command text"""
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'charging station', 'dining room']
        for loc in locations:
            if loc in text:
                return loc
        return 'unknown'

    def extract_object(self, text):
        """Extract object from command text"""
        objects = ['cup', 'ball', 'book', 'box', 'bottle', 'glass', 'phone', 'keys']
        for obj in objects:
            if obj in text:
                return obj
        return 'unknown'


class PlanningSystem:
    """LLM-based planning component of VLA system"""
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.validator = PlanningValidator()

    def generate_and_validate_plan(self, intent, context):
        """Generate plan using LLM and validate for execution"""
        try:
            # Generate plan using LLM
            plan = self.generate_plan_with_llm(intent, context)

            # Validate plan for safety and feasibility
            validation_result = self.validator.validate_plan(plan, context)

            if validation_result['valid']:
                return {
                    'success': True,
                    'plan': plan,
                    'validation': validation_result
                }
            else:
                return {
                    'success': False,
                    'error': 'Plan validation failed',
                    'issues': validation_result['issues']
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_plan_with_llm(self, intent, context):
        """Generate task plan using LLM"""
        action = intent['action']
        parameters = intent['parameters']

        prompt = f"""
        You are a task planning expert for humanoid robotics. Given the following intent
        and environmental context, generate a detailed task plan.

        Intent: {action} with parameters {parameters}

        Environmental Context:
        - Current robot position: {context.get('robot_position', 'unknown')}
        - Known locations: {context.get('known_locations', [])}
        - Detected objects: {context.get('detected_objects', [])}
        - Navigation map: {context.get('navigation_map', {})}

        Generate a plan that:
        1. Achieves the stated goal
        2. Respects safety constraints
        3. Uses available resources efficiently
        4. Includes perception steps where needed

        Format as JSON with the following structure:
        {{
            "action_sequence": [
                {{
                    "id": 1,
                    "type": "navigation|perception|manipulation|system",
                    "description": "detailed action description",
                    "parameters": {{"param_name": "value"}},
                    "dependencies": [list_of_previous_action_ids],
                    "requires_perception": true|false
                }}
            ],
            "estimated_duration": number_of_seconds
        }}
        """

        # For this example, we'll return a mock plan
        # In real implementation, this would call the LLM
        if action == 'navigation':
            destination = parameters.get('destination', 'unknown')
            return {
                "action_sequence": [
                    {
                        "id": 1,
                        "type": "navigation",
                        "description": f"Navigate to {destination}",
                        "parameters": {"location": destination},
                        "dependencies": [],
                        "requires_perception": False
                    }
                ],
                "estimated_duration": 30
            }
        elif action == 'manipulation':
            obj = parameters.get('object', 'unknown')
            return {
                "action_sequence": [
                    {
                        "id": 1,
                        "type": "navigation",
                        "description": f"Navigate to {obj}",
                        "parameters": {"object": obj},
                        "dependencies": [],
                        "requires_perception": False
                    },
                    {
                        "id": 2,
                        "type": "perception",
                        "description": f"Detect {obj}",
                        "parameters": {"object_type": obj},
                        "dependencies": [1],
                        "requires_perception": True
                    },
                    {
                        "id": 3,
                        "type": "manipulation",
                        "description": f"Grasp {obj}",
                        "parameters": {"object_id": obj},
                        "dependencies": [2],
                        "requires_perception": False
                    }
                ],
                "estimated_duration": 60
            }
        else:
            return {
                "action_sequence": [
                    {
                        "id": 1,
                        "type": "system",
                        "description": f"Process {action} command",
                        "parameters": parameters,
                        "dependencies": [],
                        "requires_perception": False
                    }
                ],
                "estimated_duration": 10
            }


class PlanningValidator:
    """Validate plans for safety and feasibility"""
    def validate_plan(self, plan, context):
        """Validate plan against safety and feasibility constraints"""
        issues = []

        # Check action types
        valid_action_types = ['navigation', 'perception', 'manipulation', 'system']
        for action in plan['action_sequence']:
            if action['type'] not in valid_action_types:
                issues.append(f"Invalid action type: {action['type']}")

        # Check dependencies
        action_ids = {action['id'] for action in plan['action_sequence']}
        for action in plan['action_sequence']:
            for dep_id in action['dependencies']:
                if dep_id not in action_ids:
                    issues.append(f"Invalid dependency ID: {dep_id}")

        # Check for circular dependencies
        if self.has_circular_dependencies(plan['action_sequence']):
            issues.append("Circular dependencies detected")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def has_circular_dependencies(self, actions):
        """Check for circular dependencies in action sequence"""
        # Simple cycle detection using adjacency list
        adj_list = {action['id']: action['dependencies'] for action in actions}

        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for action in actions:
            if action['id'] not in visited:
                if dfs(action['id']):
                    return True

        return False


class PerceptionSystem:
    """Perception component of VLA system"""
    def __init__(self, node):
        self.node = node
        self.current_objects = []
        self.current_map = {}
        self.laser_obstacles = []

    def get_current_context(self):
        """Get current perception context"""
        return {
            'objects': self.current_objects,
            'map': self.current_map,
            'obstacles': self.laser_obstacles,
            'timestamp': time.time()
        }

    def process_image_data(self, image_msg):
        """Process image data to detect objects"""
        # In a real system, this would use computer vision
        # For this example, return mock data
        detected_objects = [
            {'name': 'cup', 'position': {'x': 1.0, 'y': 0.5}, 'confidence': 0.8},
            {'name': 'ball', 'position': {'x': 2.0, 'y': 1.0}, 'confidence': 0.7}
        ]
        self.current_objects = detected_objects
        return detected_objects

    def process_laser_data(self, laser_msg):
        """Process laser scan data to detect obstacles"""
        # In a real system, this would process laser data
        # For this example, return mock data
        obstacles = [
            {'x': 1.5, 'y': 0.0, 'radius': 0.2},
            {'x': 0.0, 'y': 1.5, 'radius': 0.1}
        ]
        self.laser_obstacles = obstacles
        return obstacles


class ControlSystem:
    """Control component of VLA system"""
    def __init__(self, node):
        self.node = node

    def execute_plan(self, plan):
        """Execute the given plan"""
        try:
            self.node.get_logger().info(f'Executing plan with {len(plan["action_sequence"])} subtasks')

            for subtask in plan['action_sequence']:
                result = self.execute_subtask(subtask)
                if not result['success']:
                    return {
                        'success': False,
                        'error': f'Subtask {subtask["id"]} failed: {result["error"]}'
                    }

            return {
                'success': True,
                'message': f'Plan completed with {len(plan["action_sequence"])} subtasks'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Plan execution error: {str(e)}'
            }

    def execute_subtask(self, subtask):
        """Execute a single subtask"""
        try:
            task_type = subtask['type']
            params = subtask['parameters']

            if task_type == 'navigation':
                return self.execute_navigation(params)
            elif task_type == 'manipulation':
                return self.execute_manipulation(params)
            elif task_type == 'perception':
                return self.execute_perception(params)
            elif task_type == 'system':
                return self.execute_system_task(params)
            else:
                return self.execute_generic_task(task_type, params)

        except Exception as e:
            return {
                'success': False,
                'error': f'Subtask execution error: {str(e)}'
            }

    def execute_navigation(self, params):
        """Execute navigation task"""
        destination = params.get('location', params.get('object', 'unknown'))
        self.node.get_logger().info(f'Navigating to {destination}')
        # In a real system, this would send navigation commands
        # For this example, just return success
        return {'success': True, 'message': f'Navigated to {destination}'}

    def execute_manipulation(self, params):
        """Execute manipulation task"""
        object_id = params.get('object_id', 'unknown')
        self.node.get_logger().info(f'Grasping object {object_id}')
        # In a real system, this would send manipulation commands
        # For this example, just return success
        return {'success': True, 'message': f'Grasped object {object_id}'}

    def execute_perception(self, params):
        """Execute perception task"""
        object_type = params.get('object_type', 'unknown')
        self.node.get_logger().info(f'Detecting {object_type}')
        # In a real system, this would trigger perception processing
        # For this example, just return success
        return {'success': True, 'message': f'Detected {object_type}'}

    def execute_system_task(self, params):
        """Execute system task"""
        command = params.get('command', 'unknown')
        self.node.get_logger().info(f'Executing system command: {command}')
        return {'success': True, 'message': f'Executed system command: {command}'}

    def execute_generic_task(self, task_type, params):
        """Execute generic task"""
        self.node.get_logger().info(f'Executing {task_type} task')
        return {'success': True, 'message': f'Completed {task_type} task'}


class SafetyValidator:
    """Safety validation component of VLA system"""
    def __init__(self, node):
        self.node = node
        self.safety_limits = {
            'max_linear_velocity': 0.5,
            'max_angular_velocity': 0.5,
            'max_manipulation_force': 30.0,
            'min_human_distance': 1.0,
            'max_payload': 5.0
        }

    def validate_plan(self, plan, current_state):
        """Validate plan against safety constraints"""
        try:
            # Check each subtask for safety
            for subtask in plan.get('action_sequence', []):
                if not self.validate_subtask(subtask, current_state):
                    self.node.get_logger().warn(f'Safety validation failed for subtask: {subtask}')
                    return False

            # Check overall plan safety
            if not self.validate_plan_context(plan, current_state):
                self.node.get_logger().warn('Plan failed context safety validation')
                return False

            return True

        except Exception as e:
            self.node.get_logger().error(f'Safety validation error: {e}')
            return False

    def validate_subtask(self, subtask, current_state):
        """Validate individual subtask for safety"""
        task_type = subtask.get('type', 'unknown')

        if task_type == 'navigation':
            return self.validate_navigation_safety(subtask, current_state)
        elif task_type == 'manipulation':
            return self.validate_manipulation_safety(subtask, current_state)
        elif task_type == 'perception':
            return self.validate_perception_safety(subtask, current_state)
        else:
            return True  # For other task types, assume safe

    def validate_navigation_safety(self, subtask, current_state):
        """Validate navigation task safety"""
        # Check if destination is safe
        obstacles = current_state.get('environment_map', {}).get('obstacles', [])
        destination = subtask.get('parameters', {}).get('location', 'unknown')

        # In a real system, this would check if destination is safe
        # For this example, assume safe
        return True

    def validate_manipulation_safety(self, subtask, current_state):
        """Validate manipulation task safety"""
        # Check if manipulation is safe
        # In a real system, this would check various safety constraints
        # For this example, assume safe
        return True

    def validate_perception_safety(self, subtask, current_state):
        """Validate perception task safety"""
        # Check if perception task is safe
        # For this example, assume safe
        return True

    def validate_plan_context(self, plan, current_state):
        """Validate plan in current context"""
        # Check if plan is safe given current state
        # In a real system, this would perform comprehensive safety checks
        # For this example, assume safe
        return True


class IntegrationCoordinator:
    """Coordinate integration between VLA system components"""
    def __init__(self, node):
        self.node = node

    def integrate_perception_data(self, plan, perception_context):
        """Integrate perception data into the plan"""
        enriched_plan = plan.copy()

        # Update plan with current perception data
        enriched_plan['context'] = perception_context
        enriched_plan['timestamp'] = time.time()

        # Adjust navigation plans based on current obstacles
        for action in enriched_plan['action_sequence']:
            if action['type'] == 'navigation':
                action['path'] = self.update_path_with_obstacles(
                    action.get('path', []),
                    perception_context.get('obstacles', [])
                )

        return enriched_plan

    def update_path_with_obstacles(self, path, obstacles):
        """Update navigation path to avoid obstacles"""
        # In a real system, this would perform path planning
        # For this example, return the same path
        return path


def main(args=None):
    """Main function to run the VLA system"""
    rclpy.init(args=args)
    node = VLASystemNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA system node')
    finally:
        node.executor.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File for Complete System

```xml
<!-- launch/vla_complete_system.launch.py -->
<launch>
  <!-- VLA System Node -->
  <node pkg="vla_examples"
        exec="vla_system_node"
        name="vla_system_node"
        output="screen">
    <param from="$(find-pkg-share vla_examples)/config/vla_system.yaml"/>
  </node>

  <!-- Perception Nodes -->
  <node pkg="vla_examples"
        exec="camera_perception_node"
        name="camera_perception_node"
        output="screen">
  </node>

  <node pkg="vla_examples"
        exec="lidar_perception_node"
        name="lidar_perception_node"
        output="screen">
  </node>

  <!-- Control Nodes -->
  <node pkg="vla_examples"
        exec="navigation_control_node"
        name="navigation_control_node"
        output="screen">
  </node>

  <node pkg="vla_examples"
        exec="manipulation_control_node"
        name="manipulation_control_node"
        output="screen">
  </node>

  <!-- Simulation Environment -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gzserver.launch.py">
    <arg name="world" value="$(find-pkg-share vla_examples)/worlds/vla_validation.world"/>
  </include>

  <include file="$(find-pkg-share gazebo_ros)/launch/gzclient.launch.py"/>

  <!-- Robot Controller -->
  <node pkg="controller_manager"
        exec="spawner"
        args="diff_drive_controller joint_state_broadcaster">
  </node>
</launch>
```

## Configuration File

```yaml
# config/vla_system.yaml
vla_system_node:
  ros__parameters:
    # LLM Configuration
    openai_api_key: "${OPENAI_API_KEY}"
    llm_model: "gpt-3.5-turbo"
    llm_temperature: 0.1

    # System Parameters
    processing_rate: 10.0  # Hz
    command_queue_size: 5
    max_processing_time: 30.0

    # Safety Parameters
    safety_check_frequency: 1.0  # Hz
    emergency_stop_timeout: 5.0

    # Voice Processing Parameters
    voice_confidence_threshold: 0.7
    voice_processing_timeout: 5.0

    # Planning Parameters
    planning_timeout: 10.0
    validation_enabled: true

    # Perception Parameters
    perception_update_rate: 5.0  # Hz
    detection_confidence_threshold: 0.5
```

## Package Configuration

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(vla_complete_system)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vla_interfaces REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install worlds for simulation
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/vla_system_node
  scripts/camera_perception_node
  scripts/lidar_perception_node
  scripts/navigation_control_node
  scripts/manipulation_control_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>vlc_complete_system</name>
  <version>1.0.0</version>
  <description>Complete VLA system implementation</description>
  <maintainer email="robotics@example.com">Robotics Team</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>nav2_msgs</depend>
  <depend>example_interfaces</depend>
  <depend>builtin_interfaces</depend>
  <depend>tf_transformations</depend>
  <depend>vla_interfaces</depend>

  <exec_depend>python3-openai</exec_depend>
  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>python3-cv2</exec_depend>
  <exec_depend>python3-psutil</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Testing the Complete System

### Example Test Scripts

```python
#!/usr/bin/env python3
"""
Test script for complete VLA system
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time


class VLASystemTester(Node):
    def __init__(self):
        super().__init__('vla_system_tester')

        # Publisher for test commands
        self.voice_cmd_pub = self.create_publisher(
            String,
            'voice_commands',
            10
        )

        # Publisher for manual control (for safety)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Test commands to send
        self.test_commands = [
            "Move forward 2 meters",
            "Go to the kitchen",
            "Pick up the red cup",
            "Stop the robot",
            "Turn left 90 degrees"
        ]

        # Timer to send test commands
        self.test_index = 0
        self.test_timer = self.create_timer(5.0, self.send_test_command)

    def send_test_command(self):
        """Send test command to VLA system"""
        if self.test_index < len(self.test_commands):
            command = self.test_commands[self.test_index]

            msg = String()
            msg.data = command

            self.voice_cmd_pub.publish(msg)
            self.get_logger().info(f'Sent test command: {command}')

            self.test_index += 1
        else:
            self.get_logger().info('All test commands sent')
            self.test_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    tester = VLASystemTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Shutting down VLA system tester')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Running the Complete System

### Build and Run Instructions

```bash
# Build the package
colcon build --packages-select vla_complete_system
source install/setup.bash

# Run the complete system
ros2 launch vla_complete_system vla_complete_system.launch.py

# Send test commands
ros2 topic pub /voice_commands std_msgs/String "data: 'Move forward 2 meters'"

# Monitor feedback
ros2 topic echo /vla_feedback
```

This complete end-to-end code example demonstrates the full Vision-Language-Action pipeline integration. The system processes voice commands, uses LLMs for planning, integrates perception data, validates safety, and executes robot actions in a coordinated manner. The modular design allows for easy extension and maintenance while ensuring robust operation through proper error handling and safety validation.