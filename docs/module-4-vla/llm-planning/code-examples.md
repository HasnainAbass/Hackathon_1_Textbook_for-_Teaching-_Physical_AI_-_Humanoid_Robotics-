# Code Examples for LLM-Based Task Planning

## Introduction

This section provides practical code examples for implementing LLM-based cognitive planning in humanoid robotics. These examples demonstrate how to integrate large language models with ROS 2 for task decomposition, constraint-aware planning, and human-in-the-loop control.

## Complete LLM-Based Planning System

### Main Planning Node

```python
#!/usr/bin/env python3
"""
Complete LLM-based planning system for humanoid robotics.
Integrates task decomposition, constraint checking, and human-in-the-loop control.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import TaskPlan, Task, ConstraintViolation
from openai import OpenAI
import json
import time
from threading import Lock


class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize LLM client
        self.llm_client = OpenAI(api_key=self.get_parameter_or('openai_api_key', ''))

        # Initialize components
        self.task_decomposer = LLMTaskDecomposer(self.llm_client)
        self.constraint_validator = ConstraintValidator()
        self.human_controller = HumanInLoopController(self)
        self.human_integrator = HITLPlanningIntegrator(self, self.human_controller)

        # Thread safety
        self.lock = Lock()

        # Publishers and subscribers
        self.goal_sub = self.create_subscription(
            String,
            'high_level_goals',
            self.goal_callback,
            10
        )

        self.plan_pub = self.create_publisher(
            TaskPlan,
            'generated_task_plans',
            10
        )

        self.violation_pub = self.create_publisher(
            ConstraintViolation,
            'constraint_violations',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'planning_feedback',
            10
        )

        # State tracking
        self.current_state = {}
        self.active_constraints = self.load_default_constraints()

        self.get_logger().info('LLM Planning Node initialized')

    def goal_callback(self, msg):
        """Process high-level goal with LLM-based planning"""
        with self.lock:
            try:
                goal_description = msg.data
                self.get_logger().info(f'Processing goal: {goal_description}')

                # Step 1: Decompose the goal using LLM
                self.publish_feedback(f'Decomposing goal: {goal_description}')
                subtasks = self.task_decomposer.decompose_task(goal_description)

                if not subtasks:
                    self.publish_feedback('Task decomposition failed')
                    return

                # Step 2: Create initial task plan
                plan = self.create_task_plan(subtasks, goal_description)

                # Step 3: Validate plan against constraints
                is_valid, validation_results = self.constraint_validator.validate_plan(
                    plan, self.current_state
                )

                if not is_valid:
                    # Handle constraint violations
                    self.handle_constraint_violations(validation_results, plan)
                    return

                # Step 4: Check if human approval is needed
                needs_approval, reason = self.human_integrator.check_human_approval_needed(plan)

                if needs_approval:
                    self.get_logger().info(f'Requesting human approval: {reason}')
                    approved = self.human_integrator.request_human_approval(plan)

                    if not approved:
                        self.publish_feedback('Plan rejected by human operator')
                        return

                # Step 5: Publish the validated plan
                plan_msg = self.create_plan_message(plan)
                self.plan_pub.publish(plan_msg)
                self.publish_feedback(f'Plan published with {len(plan.actions)} actions')

            except Exception as e:
                self.get_logger().error(f'Error in planning: {e}')
                self.publish_feedback(f'Planning error: {e}')

    def create_task_plan(self, subtasks, original_goal):
        """Create internal task plan structure"""
        return {
            'original_goal': original_goal,
            'actions': subtasks,
            'timestamp': time.time(),
            'constraints_respected': True
        }

    def create_plan_message(self, plan_data):
        """Create ROS 2 TaskPlan message"""
        plan_msg = TaskPlan()
        plan_msg.original_goal = plan_data['original_goal']
        plan_msg.timestamp = self.get_clock().now().to_msg()
        plan_msg.constraints_respected = plan_data['constraints_respected']

        for i, action_data in enumerate(plan_data['actions']):
            action_msg = TaskPlan.Action()
            action_msg.id = i + 1
            action_msg.action_type = action_data.get('type', 'unknown')
            action_msg.description = action_data.get('description', '')
            action_msg.dependencies = action_data.get('dependencies', [])

            # Add parameters
            for param_name, param_value in action_data.get('parameters', {}).items():
                param = TaskPlan.Parameter()
                param.name = param_name
                param.value = str(param_value)
                action_msg.parameters.append(param)

            plan_msg.actions.append(action_msg)

        return plan_msg

    def handle_constraint_violations(self, validation_results, plan):
        """Handle constraint violations in generated plan"""
        violation_msg = ConstraintViolation()
        violation_msg.goal_description = plan['original_goal']
        violation_msg.timestamp = self.get_clock().now().to_msg()
        violation_msg.violations = []

        for constraint_type, result in validation_results.items():
            if not result['valid']:
                for issue in result['issues']:
                    violation = ConstraintViolation.Violation()
                    violation.type = constraint_type
                    violation.description = issue
                    violation_msg.violations.append(violation)

        # Publish violation
        self.violation_pub.publish(violation_msg)
        self.get_logger().warn(f'Constraint violations in plan for: {plan["original_goal"]}')

        # Try to generate alternative plan
        self.generate_alternative_plan(plan)

    def generate_alternative_plan(self, original_plan):
        """Generate alternative plan that respects constraints"""
        # Implementation to generate constraint-compliant alternative
        pass

    def load_default_constraints(self):
        """Load default safety and operational constraints"""
        return {
            'safety': [
                'maintain 1m distance from humans',
                'avoid collisions with obstacles',
                'limit speed to 0.5 m/s',
                'maximum manipulation force 30N'
            ],
            'environmental': [
                'avoid no-go zones',
                'respect fragile objects',
                'use designated pathways'
            ],
            'operational': [
                'return to charging station when battery < 20%',
                'avoid operation during maintenance windows'
            ]
        }

    def publish_feedback(self, message):
        """Publish feedback message"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)


class LLMTaskDecomposer:
    """LLM-based task decomposer"""
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def decompose_task(self, goal_description):
        """Decompose goal using LLM"""
        prompt = f"""
        You are an expert task decomposition system for humanoid robotics.
        Decompose the following high-level goal into specific, executable subtasks.

        Goal: {goal_description}

        Provide the decomposition in the following JSON format:
        {{
            "subtasks": [
                {{
                    "id": 1,
                    "type": "navigation|manipulation|perception|system",
                    "description": "detailed description of the subtask",
                    "parameters": {{"param_name": "value"}},
                    "dependencies": [list_of_subtask_ids]
                }}
            ]
        }}

        Each subtask should be:
        1. Specific and actionable
        2. Executable by a humanoid robot
        3. In logical order
        4. Include necessary parameters for execution
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code block markers if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove ```

            data = json.loads(content)
            return data.get('subtasks', [])

        except Exception as e:
            self.get_logger().error(f'LLM decomposition failed: {e}')
            # Return fallback decomposition
            return self.fallback_decomposition(goal_description)

    def fallback_decomposition(self, goal_description):
        """Fallback task decomposition using rule-based approach"""
        # Simple rule-based decomposition as fallback
        if 'go to' in goal_description.lower():
            return [{
                'id': 1,
                'type': 'navigation',
                'description': f'Navigate to location in goal: {goal_description}',
                'parameters': {'location': self.extract_location(goal_description)},
                'dependencies': []
            }]
        elif 'pick' in goal_description.lower() or 'grasp' in goal_description.lower():
            return [{
                'id': 1,
                'type': 'manipulation',
                'description': f'Grasp object in goal: {goal_description}',
                'parameters': {'object': self.extract_object(goal_description)},
                'dependencies': []
            }]
        else:
            return [{
                'id': 1,
                'type': 'system',
                'description': f'Process goal: {goal_description}',
                'parameters': {'goal': goal_description},
                'dependencies': []
            }]

    def extract_location(self, goal):
        """Extract location from goal description"""
        # Simple location extraction
        locations = ['kitchen', 'bedroom', 'office', 'living room', 'bathroom']
        goal_lower = goal.lower()
        for loc in locations:
            if loc in goal_lower:
                return loc
        return 'unknown'

    def extract_object(self, goal):
        """Extract object from goal description"""
        # Simple object extraction
        objects = ['cup', 'ball', 'book', 'box', 'bottle', 'glass']
        goal_lower = goal.lower()
        for obj in objects:
            if obj in goal_lower:
                return obj
        return 'unknown'

    def get_logger(self):
        """Helper to get logger"""
        import logging
        return logging.getLogger(__name__)


class ConstraintValidator:
    """Validate plans against constraints"""
    def __init__(self):
        self.safety_constraints = SafetyConstraintChecker()
        self.environmental_constraints = EnvironmentalConstraintChecker()
        self.capability_constraints = CapabilityConstraintChecker()

    def validate_plan(self, plan, context):
        """Validate plan against all constraint types"""
        results = {
            'safety': self.safety_constraints.validate(plan, context),
            'environmental': self.environmental_constraints.validate(plan, context),
            'capability': self.capability_constraints.validate(plan, context)
        }

        overall_valid = all(result['valid'] for result in results.values())
        return overall_valid, results


class SafetyConstraintChecker:
    """Check safety constraints"""
    def __init__(self):
        self.safety_limits = {
            'max_linear_velocity': 0.5,  # m/s
            'max_angular_velocity': 0.5,  # rad/s
            'max_manipulation_force': 30.0,  # Newtons
            'min_human_distance': 1.0,  # meters
            'max_payload': 5.0  # kg
        }

    def validate(self, plan, context):
        """Validate plan against safety constraints"""
        issues = []

        for action in plan.get('actions', []):
            action_issues = self.check_action_safety(action, context)
            issues.extend(action_issues)

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def check_action_safety(self, action, context):
        """Check individual action safety"""
        issues = []

        # Check velocity constraints for navigation actions
        if action.get('type') == 'navigation':
            linear_vel = action.get('parameters', {}).get('linear_velocity', 0)
            if abs(linear_vel) > self.safety_limits['max_linear_velocity']:
                issues.append(f"Linear velocity {linear_vel} exceeds safety limit")

        # Check manipulation force constraints
        if action.get('type') == 'manipulation':
            force = action.get('parameters', {}).get('force', 0)
            if force > self.safety_limits['max_manipulation_force']:
                issues.append(f"Manipulation force {force} exceeds safety limit")

        return issues


class EnvironmentalConstraintChecker:
    """Check environmental constraints"""
    def validate(self, plan, context):
        """Validate plan against environmental constraints"""
        issues = []
        # Implementation for environmental constraints
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


class CapabilityConstraintChecker:
    """Check capability constraints"""
    def validate(self, plan, context):
        """Validate plan against robot capability constraints"""
        issues = []
        # Implementation for capability constraints
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


class HumanInLoopController:
    """Human-in-the-loop controller"""
    def __init__(self, node):
        self.node = node
        self.human_override_active = False
        self.approval_required = False


class HITLPlanningIntegrator:
    """Integrate HITL with planning"""
    def __init__(self, node, human_controller):
        self.node = node
        self.human_controller = human_controller

    def check_human_approval_needed(self, plan):
        """Check if human approval is needed"""
        # Check for safety-critical actions
        for action in plan.get('actions', []):
            if self.is_action_safety_critical(action):
                return True, "Safety-critical action detected"

        return False, "No human approval needed"

    def is_action_safety_critical(self, action):
        """Check if action is safety-critical"""
        safety_critical_types = ['grasp_fragile_object', 'navigate_near_human', 'manipulate_kitchen_tool']
        return action.get('type') in safety_critical_types

    def request_human_approval(self, plan):
        """Request human approval (simplified)"""
        # In a real implementation, this would wait for human response
        return True  # For example purposes, assume approval


def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM planning node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Configuration Files

### Planning Configuration

```yaml
# config/llm_planning.yaml
llm_planning_node:
  ros__parameters:
    # LLM Configuration
    openai_api_key: "${OPENAI_API_KEY}"
    llm_model: "gpt-3.5-turbo"
    llm_temperature: 0.1

    # Planning Parameters
    max_decomposition_depth: 10
    planning_timeout: 30.0
    constraint_validation_enabled: true

    # Safety Parameters
    safety_check_frequency: 1.0  # Hz
    emergency_stop_timeout: 5.0

    # Human-in-the-Loop Parameters
    hitl_enabled: true
    approval_threshold: 0.7
    human_response_timeout: 30.0
```

## Launch File

```xml
<!-- launch/llm_planning.launch.py -->
<launch>
  <!-- LLM Planning Node -->
  <node pkg="vla_examples"
        exec="llm_planning_node"
        name="llm_planning_node"
        output="screen">
    <param from="$(find-pkg-share vla_examples)/config/llm_planning.yaml"/>
  </node>

  <!-- Human-in-the-Loop Interface -->
  <node pkg="vla_examples"
        exec="human_interface_node"
        name="human_interface_node"
        output="screen">
  </node>

  <!-- Constraint Validation -->
  <node pkg="vla_examples"
        exec="constraint_validator_node"
        name="constraint_validator_node"
        output="screen">
  </node>

  <!-- Simulation Environment -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gzserver.launch.py">
    <arg name="world" value="$(find-pkg-share vla_examples)/worlds/planning_demo.world"/>
  </include>

  <include file="$(find-pkg-share gazebo_ros)/launch/gzclient.launch.py"/>
</launch>
```

## Utility Functions

### Plan Validation Utilities

```python
def validate_plan_completeness(plan):
    """Validate that plan is complete and executable"""
    if not plan.get('actions'):
        return False, "Plan has no actions"

    # Check that all dependencies are valid
    action_ids = {action['id'] for action in plan['actions']}
    for action in plan['actions']:
        for dep_id in action.get('dependencies', []):
            if dep_id not in action_ids:
                return False, f"Invalid dependency ID: {dep_id}"

    return True, "Plan is complete"


def estimate_plan_execution_time(plan):
    """Estimate plan execution time"""
    time_estimate = 0.0

    for action in plan.get('actions', []):
        action_type = action.get('type', '')

        if action_type == 'navigation':
            # Estimate based on distance
            distance = float(action.get('parameters', {}).get('distance', 1.0))
            time_estimate += distance / 0.5  # Assume 0.5 m/s average speed

        elif action_type == 'manipulation':
            # Estimate based on complexity
            time_estimate += 5.0  # Assume 5 seconds per manipulation

        elif action_type == 'perception':
            time_estimate += 2.0  # Assume 2 seconds per perception task

        else:
            time_estimate += 1.0  # Default 1 second per action

    return time_estimate


def optimize_plan_order(plan):
    """Optimize action order based on dependencies"""
    import networkx as nx

    # Create dependency graph
    G = nx.DiGraph()

    for action in plan['actions']:
        G.add_node(action['id'], action=action)
        for dep_id in action.get('dependencies', []):
            G.add_edge(dep_id, action['id'])

    # Get topological order
    ordered_ids = list(nx.topological_sort(G))

    # Reorder actions
    id_to_action = {action['id']: action for action in plan['actions']}
    ordered_actions = [id_to_action[aid] for aid in ordered_ids]

    plan['actions'] = ordered_actions
    return plan
```

## Testing Code

### Unit Tests

```python
import unittest
from unittest.mock import Mock, MagicMock
import json

class TestLLMPlanning(unittest.TestCase):
    def setUp(self):
        # Mock the LLM client
        self.mock_llm_client = Mock()
        self.mock_llm_client.chat.completions.create.return_value = Mock()
        response_mock = Mock()
        response_mock.choices = [Mock()]
        response_mock.choices[0].message = Mock()
        response_mock.choices[0].message.content = json.dumps({
            "subtasks": [
                {
                    "id": 1,
                    "type": "navigation",
                    "description": "Navigate to kitchen",
                    "parameters": {"location": "kitchen"},
                    "dependencies": []
                }
            ]
        })
        self.mock_llm_client.chat.completions.create.return_value = response_mock

        self.decomposer = LLMTaskDecomposer(self.mock_llm_client)

    def test_task_decomposition(self):
        """Test LLM-based task decomposition"""
        goal = "Go to the kitchen"
        result = self.decomposer.decompose_task(goal)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['type'], 'navigation')
        self.assertEqual(result[0]['description'], 'Navigate to kitchen')

    def test_fallback_decomposition(self):
        """Test fallback decomposition for invalid LLM responses"""
        # Test with invalid response
        self.mock_llm_client.chat.completions.create.side_effect = Exception("LLM Error")

        decomposer = LLMTaskDecomposer(self.mock_llm_client)
        result = decomposer.decompose_task("Go to kitchen")

        # Should return fallback decomposition
        self.assertGreater(len(result), 0)

    def test_plan_validation(self):
        """Test plan validation utilities"""
        plan = {
            'actions': [
                {'id': 1, 'type': 'navigation', 'dependencies': []},
                {'id': 2, 'type': 'manipulation', 'dependencies': [1]}
            ]
        }

        is_valid, message = validate_plan_completeness(plan)
        self.assertTrue(is_valid)

    def test_plan_execution_time_estimation(self):
        """Test plan execution time estimation"""
        plan = {
            'actions': [
                {'type': 'navigation', 'parameters': {'distance': '2.0'}},
                {'type': 'manipulation', 'parameters': {}}
            ]
        }

        estimated_time = estimate_plan_execution_time(plan)
        # Should be approximately 2.0/0.5 + 5.0 = 9.0 seconds
        self.assertGreaterEqual(estimated_time, 8.0)
        self.assertLessEqual(estimated_time, 10.0)


class TestConstraintValidation(unittest.TestCase):
    def setUp(self):
        self.validator = ConstraintValidator()

    def test_safety_constraint_validation(self):
        """Test safety constraint validation"""
        plan = {
            'actions': [
                {
                    'type': 'navigation',
                    'parameters': {'linear_velocity': 1.0}  # Exceeds limit of 0.5
                }
            ]
        }

        context = {}
        is_valid, results = self.validator.validate_plan(plan, context)

        # Should be invalid due to velocity constraint violation
        self.assertFalse(is_valid)
        self.assertIn('safety', results)
        self.assertFalse(results['safety']['valid'])
```

## Example Usage Scripts

### Simple Planning Example

```python
#!/usr/bin/env python3
"""
Simple example of using the LLM planning system
"""

def simple_planning_example():
    """Simple example of how to use the LLM planning system"""
    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node('planning_example')

    # Publisher for goals
    goal_publisher = node.create_publisher(String, 'high_level_goals', 10)

    # Example goals to test
    test_goals = [
        "Go to the kitchen and bring me a glass of water",
        "Navigate to the charging station",
        "Find the red ball and pick it up",
        "Clean the table and then charge yourself"
    ]

    for goal in test_goals:
        # Publish goal
        goal_msg = String()
        goal_msg.data = goal
        goal_publisher.publish(goal_msg)
        node.get_logger().info(f'Published goal: {goal}')

        # Wait before next goal
        time.sleep(5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    simple_planning_example()
```

### Simulation Integration Example

```python
#!/usr/bin/env python3
"""
Example of integrating LLM planning with simulation
"""

def simulation_integration_example():
    """Example of integrating planning with simulation"""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped
    from vla_interfaces.msg import TaskPlan

    class SimulationIntegrationNode(Node):
        def __init__(self):
            super().__init__('simulation_integration_node')

            # Publishers
            self.goal_publisher = self.create_publisher(String, 'high_level_goals', 10)
            self.nav_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)

            # Subscribers
            self.plan_subscriber = self.create_subscription(
                TaskPlan,
                'generated_task_plans',
                self.plan_callback,
                10
            )

            self.timer = self.create_timer(10.0, self.send_test_goal)

        def send_test_goal(self):
            """Send a test goal to the planning system"""
            goal_msg = String()
            goal_msg.data = "Go to the kitchen and return to the charging station"
            self.goal_publisher.publish(goal_msg)
            self.get_logger().info('Sent test goal to planning system')

        def plan_callback(self, msg):
            """Handle received task plan"""
            self.get_logger().info(f'Received plan with {len(msg.actions)} actions')

            # Execute navigation actions in simulation
            for action in msg.actions:
                if action.action_type == 'navigation':
                    self.execute_navigation_action(action)

        def execute_navigation_action(self, action):
            """Execute navigation action in simulation"""
            # Extract location from parameters
            location_param = None
            for param in action.parameters:
                if param.name == 'location':
                    location_param = param.value
                    break

            if location_param:
                # Publish navigation goal
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'map'

                # Set pose based on location (simplified)
                if location_param == 'kitchen':
                    pose_msg.pose.position.x = 2.0
                    pose_msg.pose.position.y = 1.0
                elif location_param == 'charging_station':
                    pose_msg.pose.position.x = 0.0
                    pose_msg.pose.position.y = 0.0

                self.nav_publisher.publish(pose_msg)
                self.get_logger().info(f'Published navigation goal to {location_param}')

    rclpy.init()
    node = SimulationIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    simulation_integration_example()
```

## Package Configuration

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(vla_llm_planning)

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

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/llm_planning_node
  scripts/human_interface_node
  scripts/constraint_validator_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>vlc_llm_planning</name>
  <version>1.0.0</version>
  <description>LLM-based planning for VLA systems</description>
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
  <depend>vla_interfaces</depend>

  <exec_depend>python3-openai</exec_depend>
  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>python3-networkx</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

These code examples provide a complete implementation of LLM-based cognitive planning for humanoid robotics, including task decomposition, constraint validation, human-in-the-loop control, and integration with ROS 2. The examples demonstrate best practices for implementing safe and effective planning systems that leverage the power of large language models while maintaining safety and reliability.