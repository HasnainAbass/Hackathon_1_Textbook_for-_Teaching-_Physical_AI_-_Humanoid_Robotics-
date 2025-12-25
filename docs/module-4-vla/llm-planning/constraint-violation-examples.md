# Examples of Constraint Violation Detection and Safe Alternatives

## Introduction

This document provides practical examples of constraint violations that can occur in LLM-based planning systems and demonstrates how to detect them and propose safe alternatives. These examples illustrate the importance of constraint-aware planning and the implementation of safety mechanisms in Vision-Language-Action (VLA) systems.

## 1. Navigation Constraint Violations

### 1.1 Collision Avoidance Violations

**Scenario**: LLM generates a plan to navigate to a location but doesn't account for obstacles.

```python
# Example of constraint violation detection
def detect_collision_violation(plan, environment_map):
    """Detect navigation plans that violate collision avoidance constraints"""
    violations = []

    for action in plan.actions:
        if action.action_type == 'NavigateToPose':
            # Calculate intended path
            path = calculate_navigation_path(action.parameters, environment_map)

            # Check for obstacles in path
            obstacles_in_path = find_obstacles_along_path(path, environment_map)

            if obstacles_in_path:
                violations.append({
                    'type': 'collision_avoidance',
                    'action_id': action.id,
                    'description': f'Path to {action.parameters.get("location", "unknown")} contains obstacles',
                    'obstacles': obstacles_in_path,
                    'severity': 'high'
                })

    return violations

# Safe alternative generation
def generate_safe_navigation_alternative(violating_action, environment_map):
    """Generate safe navigation alternative that avoids obstacles"""
    original_location = violating_action.parameters.get('location')

    # Find alternative routes that avoid obstacles
    alternative_paths = find_alternative_paths(
        start_position=get_robot_position(),
        target_location=original_location,
        environment_map=environment_map,
        min_safe_distance=0.5  # 50cm safety buffer
    )

    if alternative_paths:
        # Choose safest path
        safest_path = min(alternative_paths, key=lambda p: p.risk_score)

        return {
            'action_type': 'NavigateToPose',
            'parameters': {
                'location': original_location,
                'path': safest_path.waypoints,
                'max_speed': 0.3,  # Slower for safety
                'safety_buffer': 0.5
            },
            'constraints_respected': True
        }
    else:
        # No safe path available - return to safe location
        return {
            'action_type': 'NavigateToPose',
            'parameters': {
                'location': 'safe_zone',
                'path': find_safe_path_to_safe_zone(),
                'reason': 'no_safe_path_to_target'
            },
            'constraints_respected': True
        }
```

### 1.2 Human Safety Zone Violations

**Scenario**: LLM generates a plan that brings the robot too close to humans.

```python
def detect_human_safety_violation(plan, human_positions):
    """Detect plans that violate human safety zone constraints"""
    violations = []

    for action in plan.actions:
        if action.action_type == 'NavigateToPose':
            target_location = action.parameters.get('location')

            # Calculate distance to nearest human
            min_distance_to_human = calculate_min_distance_to_humans(
                target_location, human_positions
            )

            if min_distance_to_human < 1.0:  # 1 meter safety zone
                violations.append({
                    'type': 'human_safety',
                    'action_id': action.id,
                    'description': f'Navigation target too close to human (distance: {min_distance_to_human:.2f}m)',
                    'min_distance_required': 1.0,
                    'actual_distance': min_distance_to_human,
                    'severity': 'critical'
                })

    return violations

def generate_human_safe_alternative(violating_action, human_positions):
    """Generate alternative that maintains safe distance from humans"""
    original_target = violating_action.parameters.get('location')

    # Find safe location that maintains minimum distance
    safe_location = find_safe_location_around_target(
        original_target,
        human_positions,
        min_safe_distance=1.0
    )

    if safe_location:
        return {
            'action_type': 'NavigateToPose',
            'parameters': {
                'location': safe_location,
                'original_intent': original_target,
                'safety_margin': 1.0
            },
            'constraints_respected': True
        }
    else:
        # No safe location available - abort navigation
        return {
            'action_type': 'System',
            'parameters': {
                'command': 'abort_navigation',
                'reason': 'no_safe_location_near_target'
            },
            'constraints_respected': True
        }
```

## 2. Manipulation Constraint Violations

### 2.1 Payload Limit Violations

**Scenario**: LLM generates a plan to grasp an object that exceeds the robot's payload capacity.

```python
def detect_payload_violation(plan):
    """Detect manipulation plans that exceed payload limits"""
    violations = []
    MAX_PAYLOAD = 5.0  # 5 kg limit

    for action in plan.actions:
        if action.action_type == 'GraspAction':
            object_weight = action.parameters.get('object_weight', 0)

            if object_weight > MAX_PAYLOAD:
                violations.append({
                    'type': 'payload_limit',
                    'action_id': action.id,
                    'description': f'Object weight {object_weight}kg exceeds payload limit {MAX_PAYLOAD}kg',
                    'max_allowed': MAX_PAYLOAD,
                    'actual_weight': object_weight,
                    'severity': 'high'
                })

    return violations

def generate_payload_safe_alternative(violating_action):
    """Generate safe alternative for payload violations"""
    object_weight = violating_action.parameters.get('object_weight', 0)
    object_name = violating_action.parameters.get('object_name', 'unknown')

    if object_weight > 10.0:  # Way too heavy
        return {
            'action_type': 'System',
            'parameters': {
                'command': 'request_assistance',
                'object': object_name,
                'weight': object_weight,
                'reason': 'object_too_heavy'
            },
            'constraints_respected': True
        }
    elif object_weight > 5.0:  # Just over limit
        return {
            'action_type': 'System',
            'parameters': {
                'command': 'find_lighter_alternative',
                'object': object_name,
                'weight': object_weight,
                'reason': 'payload_exceeded'
            },
            'constraints_respected': True
        }
    else:
        # Shouldn't happen if we're checking correctly
        return None
```

### 2.2 Force Limit Violations

**Scenario**: LLM generates a plan with manipulation forces that exceed safety limits.

```python
def detect_force_violation(plan):
    """Detect manipulation plans that exceed force limits"""
    violations = []
    MAX_GRASP_FORCE = 30.0  # 30 Newtons limit

    for action in plan.actions:
        if action.action_type == 'GraspAction':
            grasp_force = action.parameters.get('grasp_force', 0)

            if grasp_force > MAX_GRASP_FORCE:
                violations.append({
                    'type': 'force_limit',
                    'action_id': action.id,
                    'description': f'Grasp force {grasp_force}N exceeds safety limit {MAX_GRASP_FORCE}N',
                    'max_allowed': MAX_GRASP_FORCE,
                    'actual_force': grasp_force,
                    'severity': 'critical'
                })

    return violations

def generate_force_safe_alternative(violating_action):
    """Generate safe alternative for force violations"""
    original_force = violating_action.parameters.get('grasp_force', 0)
    object_name = violating_action.parameters.get('object_name', 'unknown')

    # Reduce force to safe level
    safe_force = min(original_force, 25.0)  # Conservative safe limit

    return {
        'action_type': 'GraspAction',
        'parameters': {
            'object_name': object_name,
            'grasp_force': safe_force,
            'original_force': original_force,
            'force_reduced': True,
            'safety_margin': 5.0  # 5N safety buffer
        },
        'constraints_respected': True
    }
```

## 3. Environmental Constraint Violations

### 3.1 No-Go Zone Violations

**Scenario**: LLM generates a plan that navigates to restricted areas.

```python
def detect_no_go_zone_violation(plan, restricted_areas):
    """Detect plans that enter no-go zones"""
    violations = []

    for action in plan.actions:
        if action.action_type == 'NavigateToPose':
            target_location = action.parameters.get('location', 'unknown')

            # Check if target is in restricted area
            if is_location_in_restricted_area(target_location, restricted_areas):
                violations.append({
                    'type': 'no_go_zone',
                    'action_id': action.id,
                    'description': f'Navigation target {target_location} is in restricted area',
                    'restricted_area': get_restricted_area(target_location, restricted_areas),
                    'severity': 'critical'
                })

    return violations

def generate_no_go_zone_alternative(violating_action, restricted_areas):
    """Generate alternative that avoids no-go zones"""
    original_target = violating_action.parameters.get('location')

    # Find closest accessible location
    closest_accessible = find_closest_accessible_location(
        original_target,
        restricted_areas
    )

    if closest_accessible:
        return {
            'action_type': 'NavigateToPose',
            'parameters': {
                'location': closest_accessible,
                'original_intent': original_target,
                'accessibility_verified': True
            },
            'constraints_respected': True
        }
    else:
        # No alternative available
        return {
            'action_type': 'System',
            'parameters': {
                'command': 'abort_task',
                'reason': 'no_accessible_alternative',
                'original_target': original_target
            },
            'constraints_respected': True
        }
```

## 4. Complete Constraint Violation Detection System

### 4.1 Multi-Constraint Violation Detection

```python
class ConstraintViolationDetector:
    def __init__(self):
        self.max_payload = 5.0
        self.max_force = 30.0
        self.min_human_distance = 1.0
        self.min_collision_distance = 0.1
        self.restricted_areas = []
        self.human_positions = []

    def detect_all_violations(self, plan, environment_context):
        """Detect all types of constraint violations in a plan"""
        violations = []

        # Update context
        self.human_positions = environment_context.get('humans', [])
        self.restricted_areas = environment_context.get('restricted_areas', [])

        # Check each constraint type
        violations.extend(self._check_navigation_violations(plan, environment_context))
        violations.extend(self._check_manipulation_violations(plan))
        violations.extend(self._check_safety_violations(plan, environment_context))

        return violations

    def _check_navigation_violations(self, plan, environment_context):
        """Check navigation-related constraint violations"""
        violations = []
        environment_map = environment_context.get('map', {})

        for action in plan.actions:
            if action.action_type == 'NavigateToPose':
                # Check collision avoidance
                path = calculate_navigation_path(action.parameters, environment_map)
                obstacles_in_path = find_obstacles_along_path(path, environment_map)

                if obstacles_in_path:
                    violations.append({
                        'type': 'collision_avoidance',
                        'action_id': action.id,
                        'description': f'Navigation path contains {len(obstacles_in_path)} obstacles',
                        'obstacles': obstacles_in_path,
                        'severity': 'high'
                    })

                # Check human safety zones
                target_location = action.parameters.get('location')
                if target_location:
                    min_distance = calculate_min_distance_to_humans(
                        target_location, self.human_positions
                    )

                    if min_distance < self.min_human_distance:
                        violations.append({
                            'type': 'human_safety',
                            'action_id': action.id,
                            'description': f'Navigation target too close to human (distance: {min_distance:.2f}m)',
                            'min_distance_required': self.min_human_distance,
                            'actual_distance': min_distance,
                            'severity': 'critical'
                        })

        return violations

    def _check_manipulation_violations(self, plan):
        """Check manipulation-related constraint violations"""
        violations = []

        for action in plan.actions:
            if action.action_type == 'GraspAction':
                # Check payload limits
                object_weight = action.parameters.get('object_weight', 0)
                if object_weight > self.max_payload:
                    violations.append({
                        'type': 'payload_limit',
                        'action_id': action.id,
                        'description': f'Object weight {object_weight}kg exceeds limit {self.max_payload}kg',
                        'max_allowed': self.max_payload,
                        'actual_weight': object_weight,
                        'severity': 'high'
                    })

                # Check force limits
                grasp_force = action.parameters.get('grasp_force', 0)
                if grasp_force > self.max_force:
                    violations.append({
                        'type': 'force_limit',
                        'action_id': action.id,
                        'description': f'Grasp force {grasp_force}N exceeds limit {self.max_force}N',
                        'max_allowed': self.max_force,
                        'actual_force': grasp_force,
                        'severity': 'critical'
                    })

        return violations

    def _check_safety_violations(self, plan, environment_context):
        """Check general safety constraint violations"""
        violations = []

        # Check for operations in restricted areas
        for action in plan.actions:
            if action.action_type in ['NavigateToPose', 'PerceptionAction']:
                location = action.parameters.get('location')
                if location and is_location_in_restricted_area(location, self.restricted_areas):
                    violations.append({
                        'type': 'no_go_zone',
                        'action_id': action.id,
                        'description': f'Action location {location} is in restricted area',
                        'severity': 'critical'
                    })

        return violations

    def generate_safe_alternatives(self, plan, violations, environment_context):
        """Generate safe alternatives for violated constraints"""
        alternatives = {}

        for violation in violations:
            action_id = violation['action_id']

            if action_id not in alternatives:
                original_action = self.find_action_by_id(plan.actions, action_id)

                if violation['type'] == 'collision_avoidance':
                    alternative = self._generate_navigation_alternative(
                        original_action, environment_context
                    )
                elif violation['type'] == 'human_safety':
                    alternative = self._generate_human_safe_alternative(
                        original_action, environment_context
                    )
                elif violation['type'] == 'payload_limit':
                    alternative = self._generate_payload_safe_alternative(original_action)
                elif violation['type'] == 'force_limit':
                    alternative = self._generate_force_safe_alternative(original_action)
                elif violation['type'] == 'no_go_zone':
                    alternative = self._generate_restricted_area_alternative(
                        original_action, environment_context
                    )
                else:
                    alternative = self._generate_generic_alternative(original_action)

                alternatives[action_id] = alternative

        return alternatives

    def _generate_navigation_alternative(self, original_action, environment_context):
        """Generate safe navigation alternative"""
        # Implementation similar to examples above
        pass

    def _generate_human_safe_alternative(self, original_action, environment_context):
        """Generate alternative that maintains human safety"""
        # Implementation similar to examples above
        pass

    def _generate_payload_safe_alternative(self, original_action):
        """Generate safe alternative for payload violations"""
        # Implementation similar to examples above
        pass

    def _generate_force_safe_alternative(self, original_action):
        """Generate safe alternative for force violations"""
        # Implementation similar to examples above
        pass

    def _generate_restricted_area_alternative(self, original_action, environment_context):
        """Generate alternative that avoids restricted areas"""
        # Implementation similar to examples above
        pass

    def _generate_generic_alternative(self, original_action):
        """Generate generic safe alternative"""
        return {
            'action_type': 'System',
            'parameters': {
                'command': 'safe_standby',
                'original_action': original_action.action_type,
                'reason': 'constraint_violation'
            },
            'constraints_respected': True
        }

    def find_action_by_id(self, actions, action_id):
        """Find action by its ID"""
        for action in actions:
            if action.id == action_id:
                return action
        return None
```

## 5. Real-World Examples

### 5.1 Example 1: Kitchen Navigation with Humans Present

**Original LLM Plan**: "Go to the kitchen and pick up the red cup"

**Detected Violations**:
- Human safety: Robot would navigate too close to human in kitchen
- Force limit: Plan specifies excessive grasp force for delicate cup

**Safe Alternative Generated**:
- Navigate to kitchen using path that maintains 1.5m distance from human
- Approach cup with reduced grasp force of 15N instead of planned 40N
- Confirm grasp success before lifting

### 5.2 Example 2: Heavy Object Manipulation

**Original LLM Plan**: "Lift the heavy box from the floor"

**Detected Violations**:
- Payload limit: Box weighs 8kg, robot limit is 5kg
- Safety: No human supervision for heavy lifting

**Safe Alternative Generated**:
- Request human assistance for heavy object
- Or find lighter alternative task
- Or break task into multiple lighter lifts if possible

### 5.3 Example 3: Restricted Area Navigation

**Original LLM Plan**: "Go to the server room and check the equipment"

**Detected Violations**:
- No-go zone: Server room is restricted access
- Safety: No authorization for restricted area

**Safe Alternative Generated**:
- Navigate to nearest accessible monitoring point
- Request authorized personnel for server room access
- Report status from accessible location

## 6. Implementation in ROS 2

### 6.1 Constraint Validation Node

```python
#!/usr/bin/env python3
# ROS 2 node for constraint violation detection and alternatives

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import TaskPlan, ConstraintViolation, SafeAlternative
from builtin_interfaces.msg import Time

class ConstraintViolationDetectionNode(Node):
    def __init__(self):
        super().__init__('constraint_violation_detection_node')

        # Initialize constraint detector
        self.constraint_detector = ConstraintViolationDetector()

        # Publishers and subscribers
        self.plan_sub = self.create_subscription(
            TaskPlan,
            'task_plans',
            self.plan_callback,
            10
        )

        self.violation_pub = self.create_publisher(
            ConstraintViolation,
            'constraint_violations',
            10
        )

        self.alternative_pub = self.create_publisher(
            SafeAlternative,
            'safe_alternatives',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'constraint_feedback',
            10
        )

        # Environment context
        self.environment_context = {}

        self.get_logger().info('Constraint Violation Detection Node initialized')

    def plan_callback(self, msg):
        """Process incoming task plan for constraint violations"""
        try:
            self.get_logger().info(f'Checking plan for violations: {len(msg.actions)} actions')

            # Detect violations
            violations = self.constraint_detector.detect_all_violations(
                self.convert_ros_plan_to_internal(msg),
                self.environment_context
            )

            if violations:
                # Publish violations
                self.publish_violations(violations, msg.id)

                # Generate safe alternatives
                alternatives = self.constraint_detector.generate_safe_alternatives(
                    self.convert_ros_plan_to_internal(msg),
                    violations,
                    self.environment_context
                )

                # Publish alternatives
                self.publish_alternatives(alternatives, msg.id)

                # Provide feedback
                self.publish_feedback(f'Found {len(violations)} violations, generated {len(alternatives)} alternatives')

            else:
                self.get_logger().info('Plan passed all constraint checks')
                self.publish_feedback('Plan approved - no violations detected')

        except Exception as e:
            self.get_logger().error(f'Error in constraint detection: {e}')
            self.publish_feedback(f'Constraint detection error: {e}')

    def convert_ros_plan_to_internal(self, ros_plan):
        """Convert ROS TaskPlan to internal format"""
        # Implementation to convert ROS message to internal data structure
        pass

    def publish_violations(self, violations, plan_id):
        """Publish constraint violations"""
        for violation in violations:
            violation_msg = ConstraintViolation()
            violation_msg.plan_id = plan_id
            violation_msg.action_id = violation['action_id']
            violation_msg.type = violation['type']
            violation_msg.description = violation['description']
            violation_msg.severity = violation['severity']
            violation_msg.timestamp = self.get_clock().now().to_msg()

            self.violation_pub.publish(violation_msg)

    def publish_alternatives(self, alternatives, plan_id):
        """Publish safe alternatives"""
        for action_id, alternative in alternatives.items():
            alt_msg = SafeAlternative()
            alt_msg.plan_id = plan_id
            alt_msg.original_action_id = action_id
            alt_msg.action_type = alternative['action_type']
            alt_msg.constraints_respected = alternative['constraints_respected']

            # Add parameters
            for param_name, param_value in alternative['parameters'].items():
                param = SafeAlternative.Parameter()
                param.name = param_name
                param.value = str(param_value)
                alt_msg.parameters.append(param)

            self.alternative_pub.publish(alt_msg)

    def publish_feedback(self, message):
        """Publish feedback message"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ConstraintViolationDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down constraint violation detection node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Testing Constraint Violation Detection

### 7.1 Unit Tests

```python
import unittest
from unittest.mock import Mock

class TestConstraintViolationDetection(unittest.TestCase):
    def setUp(self):
        self.detector = ConstraintViolationDetector()

    def test_collision_violation_detection(self):
        """Test detection of collision avoidance violations"""
        # Create plan with collision-prone navigation
        plan = Mock()
        plan.actions = [Mock()]
        plan.actions[0].action_type = 'NavigateToPose'
        plan.actions[0].parameters = {'location': 'kitchen'}
        plan.actions[0].id = 1

        environment_map = {
            'obstacles': [{'x': 2.0, 'y': 1.0, 'radius': 0.5}]  # Obstacle in path
        }

        violations = self.detector._check_navigation_violations(plan, environment_map)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['type'], 'collision_avoidance')

    def test_payload_violation_detection(self):
        """Test detection of payload limit violations"""
        # Create plan with heavy object manipulation
        plan = Mock()
        plan.actions = [Mock()]
        plan.actions[0].action_type = 'GraspAction'
        plan.actions[0].parameters = {'object_weight': 8.0}  # Exceeds 5kg limit
        plan.actions[0].id = 1

        violations = self.detector._check_manipulation_violations(plan)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['type'], 'payload_limit')
        self.assertEqual(violations[0]['actual_weight'], 8.0)

    def test_human_safety_violation_detection(self):
        """Test detection of human safety violations"""
        # Create plan that approaches human too closely
        plan = Mock()
        plan.actions = [Mock()]
        plan.actions[0].action_type = 'NavigateToPose'
        plan.actions[0].parameters = {'location': 'near_human'}
        plan.actions[0].id = 1

        environment_context = {
            'humans': [{'x': 1.0, 'y': 1.0}]  # Human at location that's too close
        }

        violations = self.detector._check_navigation_violations(plan, environment_context)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['type'], 'human_safety')

    def test_no_violations_found(self):
        """Test that no violations are found for safe plan"""
        # Create safe plan
        plan = Mock()
        plan.actions = [Mock()]
        plan.actions[0].action_type = 'System'
        plan.actions[0].parameters = {'command': 'standby'}
        plan.actions[0].id = 1

        violations = self.detector.detect_all_violations(plan, {})

        self.assertEqual(len(violations), 0)

if __name__ == '__main__':
    unittest.main()
```

## 8. Best Practices for Constraint Handling

### 8.1 Progressive Constraint Checking

Implement progressive checking from most critical to least critical constraints:

```python
def progressive_constraint_checking(plan, context):
    """Check constraints in order of criticality"""
    # 1. Critical safety constraints (collision, human safety)
    critical_violations = check_critical_safety_constraints(plan, context)
    if critical_violations:
        return critical_violations, 'critical'

    # 2. Operational safety constraints (force, payload)
    operational_violations = check_operational_constraints(plan, context)
    if operational_violations:
        return operational_violations, 'operational'

    # 3. Environmental constraints (no-go zones, accessibility)
    environmental_violations = check_environmental_constraints(plan, context)
    if environmental_violations:
        return environmental_violations, 'environmental'

    # 4. Performance constraints (efficiency, resource usage)
    performance_violations = check_performance_constraints(plan, context)
    if performance_violations:
        return performance_violations, 'performance'

    return [], 'none'
```

### 8.2 Human-in-the-Loop for Critical Violations

For critical violations, always involve human oversight:

```python
def handle_critical_violations(violations, alternatives):
    """Handle critical violations with human oversight"""
    critical_violations = [v for v in violations if v['severity'] == 'critical']

    if critical_violations:
        # Request human approval for alternatives
        approval_required = request_human_approval(critical_violations, alternatives)

        if not approval_required:
            # Use conservative alternative
            return generate_conservative_alternatives(critical_violations)

        return alternatives
    else:
        return alternatives
```

These examples demonstrate how to effectively detect constraint violations in LLM-based planning systems and generate safe alternatives that respect all safety and operational constraints. The implementation ensures that robotic systems operate safely while maintaining their effectiveness and utility.