# Safety Constraint Validation Mechanisms

## Introduction

Safety constraint validation is a critical component of LLM-based planning systems for humanoid robotics. This document details the mechanisms and processes used to validate that LLM-generated plans comply with safety requirements before execution, ensuring the protection of humans, robots, and the environment.

## Safety Constraint Categories

### 1. Physical Safety Constraints

Physical safety constraints ensure that robot actions do not pose physical risks to humans or the environment.

#### Collision Avoidance
- **Static Obstacles**: Plans must avoid collisions with fixed environmental obstacles
- **Dynamic Obstacles**: Plans must account for moving objects and humans
- **Self-Collision**: Robot movements must avoid self-collision

#### Human Safety Zones
- **Proximity Limits**: Maintain minimum safe distances from humans
- **No-Go Areas**: Areas where human presence is restricted
- **Emergency Paths**: Maintain clear escape routes for humans

#### Manipulation Safety
- **Force Limits**: Manipulation forces must stay within safe thresholds
- **Object Handling**: Only safe objects should be manipulated
- **Workspace Boundaries**: Movements must stay within robot's safe workspace

### 2. Operational Safety Constraints

Operational constraints ensure safe robot operation within operational parameters.

#### Velocity and Acceleration Limits
- **Linear Velocity**: Maximum safe linear movement speeds
- **Angular Velocity**: Maximum safe rotational speeds
- **Acceleration Limits**: Maximum safe acceleration/deceleration

#### Payload and Load Constraints
- **Weight Limits**: Maximum safe payload capacity
- **Center of Mass**: Maintain stable center of mass during operation
- **Dynamic Stability**: Ensure robot remains stable during movements

#### Environmental Constraints
- **Terrain Limitations**: Avoid terrain that exceeds robot capabilities
- **Weather Conditions**: Consider environmental factors (if applicable)
- **Lighting Conditions**: Ensure adequate visibility for safe operation

## Safety Validation Architecture

### Multi-Layer Validation System

```python
class SafetyValidator:
    def __init__(self):
        self.physical_safety_checker = PhysicalSafetyChecker()
        self.operational_safety_checker = OperationalSafetyChecker()
        self.environmental_safety_checker = EnvironmentalSafetyChecker()
        self.human_safety_checker = HumanSafetyChecker()

    def validate_plan_safety(self, plan, context):
        """Validate plan against all safety constraint categories"""
        results = {
            'physical': self.physical_safety_checker.validate(plan, context),
            'operational': self.operational_safety_checker.validate(plan, context),
            'environmental': self.environmental_safety_checker.validate(plan, context),
            'human_safety': self.human_safety_checker.validate(plan, context)
        }

        overall_safe = all(result['safe'] for result in results.values())
        return overall_safe, results

    def validate_action_safety(self, action, context):
        """Validate individual action against safety constraints"""
        physical_ok, physical_issues = self.physical_safety_checker.validate_action(action, context)
        operational_ok, operational_issues = self.operational_safety_checker.validate_action(action, context)
        env_ok, env_issues = self.environmental_safety_checker.validate_action(action, context)
        human_ok, human_issues = self.human_safety_checker.validate_action(action, context)

        safe = physical_ok and operational_ok and env_ok and human_ok
        issues = physical_issues + operational_issues + env_issues + human_issues

        return safe, issues
```

### Physical Safety Validation

```python
class PhysicalSafetyChecker:
    def __init__(self):
        self.safety_limits = {
            'min_collision_distance': 0.1,  # meters
            'max_linear_velocity': 0.5,     # m/s
            'max_angular_velocity': 0.5,    # rad/s
            'max_manipulation_force': 30.0, # Newtons
            'max_payload': 5.0,             # kg
            'min_human_distance': 1.0       # meters
        }

    def validate_action(self, action, context):
        """Validate action against physical safety constraints"""
        issues = []

        # Check collision constraints
        if action.action_type == 'NavigateToPose':
            path = self.calculate_path(action, context)
            if not self.is_path_collision_free(path, context):
                issues.append("Path contains collision risks")

        # Check velocity constraints
        if action.action_type in ['NavigateToPose', 'MoveArm']:
            linear_vel = action.parameters.get('linear_velocity', 0)
            angular_vel = action.parameters.get('angular_velocity', 0)

            if abs(linear_vel) > self.safety_limits['max_linear_velocity']:
                issues.append(f"Linear velocity {linear_vel} exceeds limit {self.safety_limits['max_linear_velocity']}")

            if abs(angular_vel) > self.safety_limits['max_angular_velocity']:
                issues.append(f"Angular velocity {angular_vel} exceeds limit {self.safety_limits['max_angular_velocity']}")

        # Check manipulation constraints
        if action.action_type == 'GraspAction':
            force = action.parameters.get('force', 0)
            if force > self.safety_limits['max_manipulation_force']:
                issues.append(f"Grasp force {force} exceeds limit {self.safety_limits['max_manipulation_force']}")

            payload = action.parameters.get('payload_weight', 0)
            if payload > self.safety_limits['max_payload']:
                issues.append(f"Payload {payload} exceeds limit {self.safety_limits['max_payload']}")

        return len(issues) == 0, issues

    def is_path_collision_free(self, path, context):
        """Check if path is free of collisions"""
        # Implementation to check path against map and obstacles
        obstacles = context.get('obstacles', [])
        robot_radius = context.get('robot_radius', 0.3)

        for point in path:
            for obstacle in obstacles:
                distance = self.calculate_distance(point, obstacle)
                if distance < (robot_radius + obstacle.radius):
                    return False  # Collision detected

        return True

    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        import math
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
```

### Human Safety Validation

```python
class HumanSafetyChecker:
    def __init__(self):
        self.human_safety_zones = {
            'safe_distance': 1.0,    # meters
            'caution_distance': 0.5, # meters
            'danger_distance': 0.2   # meters
        }

    def validate_action(self, action, context):
        """Validate action against human safety constraints"""
        issues = []
        humans = context.get('humans', [])
        robot_position = context.get('robot_position', {'x': 0, 'y': 0})

        for human in humans:
            distance = self.calculate_human_distance(robot_position, human)

            if distance < self.human_safety_zones['danger_distance']:
                issues.append(f"Dangerous proximity to human at distance {distance:.2f}m")
            elif distance < self.human_safety_zones['safe_distance']:
                # Check if action would bring robot closer to human
                if self.action_approaches_human(action, robot_position, human):
                    issues.append(f"Action would reduce distance to human below safe threshold")

        return len(issues) == 0, issues

    def action_approaches_human(self, action, robot_pos, human_pos):
        """Check if action brings robot closer to human"""
        # Implementation to check if action trajectory approaches human
        return False  # Placeholder
```

## Integration with LLM Planning

### Safety-Aware LLM Prompting

Incorporate safety constraints directly into LLM prompts:

```python
class SafetyAwarePlanner:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.safety_validator = SafetyValidator()

    def generate_safe_plan(self, goal, safety_constraints):
        """Generate plan considering safety constraints"""
        safety_prompt = self.format_safety_constraints(safety_constraints)

        prompt = f"""
        You are a safety-aware robotic planner. Generate a plan for the following goal
        while strictly adhering to safety constraints:

        Goal: {goal}

        Safety Constraints:
        {safety_prompt}

        Generate a plan that:
        1. Achieves the stated goal
        2. Respects all safety constraints
        3. Is executable by the robot
        4. Minimizes risk to humans and environment
        5. Includes safety checks and verification steps

        Format the response as a JSON list of actions with the following structure:
        {{
            "actions": [
                {{
                    "id": 1,
                    "action_type": "navigation|manipulation|perception",
                    "description": "detailed description",
                    "parameters": {{"param_name": "value"}},
                    "safety_considerations": ["consideration1", "consideration2"],
                    "risk_assessment": "low|medium|high"
                }}
            ]
        }}

        Prioritize safety over efficiency when there's a conflict.
        """

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        plan = self.parse_plan(response.choices[0].message.content)

        # Validate the generated plan for safety
        is_safe, safety_results = self.safety_validator.validate_plan_safety(plan, safety_constraints)

        if not is_safe:
            # Generate alternative plan or request human intervention
            return self.handle_safety_violations(plan, safety_results)

        return plan

    def format_safety_constraints(self, constraints):
        """Format safety constraints for LLM prompt"""
        formatted = []
        for constraint_type, constraint_list in constraints.items():
            formatted.append(f"{constraint_type.upper()}:")
            for constraint in constraint_list:
                formatted.append(f"  - {constraint}")

        return "\n".join(formatted)
```

### Real-Time Safety Monitoring

Monitor safety during plan execution:

```python
class RealTimeSafetyMonitor:
    def __init__(self, node):
        self.node = node
        self.safety_validator = SafetyValidator()
        self.active_plan = None
        self.current_context = {}
        self.monitor_timer = node.create_timer(0.1, self.safety_check_callback)  # 10 Hz

    def safety_check_callback(self):
        """Perform real-time safety checks"""
        if self.active_plan:
            current_state = self.get_current_state()
            context = self.update_context(current_state)

            # Check if current state violates safety constraints
            for action in self.active_plan.actions:
                if self.is_currently_unsafe(action, context):
                    self.handle_safety_violation(action, context)

    def is_currently_unsafe(self, action, context):
        """Check if current state with planned action is unsafe"""
        # Check for new safety risks
        if self.detect_new_humans_nearby(context):
            return True

        if self.detect_new_obstacles_in_path(action, context):
            return True

        return False

    def handle_safety_violation(self, action, context):
        """Handle detected safety violation"""
        self.node.get_logger().error('Safety violation detected during execution')
        self.emergency_stop()
        self.publish_safety_alert(context)
```

## Safety Validation Node

### Main Implementation

```python
#!/usr/bin/env python3
# ROS 2 node for safety constraint validation

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import TaskPlan, SafetyViolation
from builtin_interfaces.msg import Time

class SafetyValidationNode(Node):
    def __init__(self):
        super().__init__('safety_validation_node')

        # Initialize safety validator
        self.safety_validator = SafetyValidator()

        # Publishers and subscribers
        self.plan_sub = self.create_subscription(
            TaskPlan,
            'task_plans',
            self.plan_callback,
            10
        )

        self.safety_violation_pub = self.create_publisher(
            SafetyViolation,
            'safety_violations',
            10
        )

        self.safety_feedback_pub = self.create_publisher(
            String,
            'safety_feedback',
            10
        )

        self.state_sub = self.create_subscription(
            String,
            'robot_state',
            self.state_callback,
            10
        )

        # Context storage
        self.current_context = {}
        self.safety_constraints = self.load_safety_constraints()

        self.get_logger().info('Safety Validation Node initialized')

    def plan_callback(self, msg):
        """Validate incoming task plan for safety"""
        try:
            self.get_logger().info(f'Validating plan with {len(msg.actions)} actions')

            # Validate the entire plan
            is_safe, safety_results = self.safety_validator.validate_plan_safety(
                self.convert_ros_plan_to_internal(msg),
                self.current_context
            )

            if is_safe:
                self.get_logger().info('Plan passed safety validation')
                self.publish_safety_feedback('Plan approved for execution - all safety checks passed')
            else:
                self.get_logger().warn('Plan failed safety validation')
                self.handle_safety_violations(safety_results, msg)

        except Exception as e:
            self.get_logger().error(f'Error in safety validation: {e}')
            self.publish_safety_feedback(f'Safety validation error: {e}')

    def state_callback(self, msg):
        """Update current robot state"""
        self.current_context = self.parse_state(msg.data)

    def convert_ros_plan_to_internal(self, ros_plan):
        """Convert ROS TaskPlan message to internal format"""
        internal_plan = {
            'original_goal': ros_plan.original_goal,
            'actions': []
        }

        for action_msg in ros_plan.actions:
            action_data = {
                'id': action_msg.id,
                'action_type': action_msg.action_type,
                'description': action_msg.description,
                'parameters': {param.name: param.value for param in action_msg.parameters},
                'dependencies': action_msg.dependencies
            }
            internal_plan['actions'].append(action_data)

        return internal_plan

    def handle_safety_violations(self, safety_results, original_plan):
        """Handle safety violations in the plan"""
        violation_msg = SafetyViolation()
        violation_msg.plan_id = original_plan.id if hasattr(original_plan, 'id') else 'unknown'
        violation_msg.timestamp = self.get_clock().now().to_msg()
        violation_msg.violations = []

        for constraint_type, result in safety_results.items():
            if not result['safe']:
                for issue in result['issues']:
                    violation = SafetyViolation.Violation()
                    violation.type = constraint_type
                    violation.description = issue
                    violation_msg.violations.append(violation)

        # Publish safety violation
        self.safety_violation_pub.publish(violation_msg)

        # Provide feedback
        self.publish_safety_feedback(f'Safety violations detected: {len(violation_msg.violations)} issues found')
        for violation in violation_msg.violations:
            self.get_logger().warn(f'Safety violation: {violation.type} - {violation.description}')

    def load_safety_constraints(self):
        """Load default safety constraints"""
        return {
            'physical': [
                'maintain 0.1m distance from obstacles',
                'limit linear velocity to 0.5 m/s',
                'limit angular velocity to 0.5 rad/s',
                'maximum manipulation force 30N'
            ],
            'human_safety': [
                'maintain 1m distance from humans',
                'avoid approaching humans rapidly',
                'stop if human enters danger zone'
            ],
            'operational': [
                'stay within robot workspace limits',
                'respect payload limits',
                'maintain stability during operation'
            ]
        }

    def publish_safety_feedback(self, message):
        """Publish safety feedback"""
        feedback_msg = String()
        feedback_msg.data = message
        self.safety_feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyValidationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down safety validation node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Safety Features

### Dynamic Safety Adaptation

Adapt safety constraints based on changing conditions:

```python
class AdaptiveSafetySystem:
    def __init__(self, node):
        self.node = node
        self.base_safety_constraints = self.load_base_constraints()
        self.adaptation_rules = self.load_adaptation_rules()

    def adapt_safety_constraints(self, environment_state):
        """Adapt safety constraints based on environment"""
        adapted_constraints = self.base_safety_constraints.copy()

        # Adjust based on environment
        if environment_state.get('crowd_density', 0) > 0.5:
            # High crowd density - increase safety margins
            adapted_constraints['human_safety'][0] = 'maintain 2m distance from humans'

        if environment_state.get('lighting', 'normal') == 'poor':
            # Poor lighting - reduce speed limits
            adapted_constraints['operational'][1] = 'limit linear velocity to 0.2 m/s'

        if environment_state.get('floor_type', 'normal') == 'slippery':
            # Slippery floor - reduce acceleration limits
            adapted_constraints['operational'].append('reduce acceleration by 50%')

        return adapted_constraints

    def load_adaptation_rules(self):
        """Load rules for safety adaptation"""
        return {
            'crowd_density': {
                'high': {'human_distance': 2.0, 'speed_limit': 0.3},
                'medium': {'human_distance': 1.0, 'speed_limit': 0.5},
                'low': {'human_distance': 1.0, 'speed_limit': 0.5}
            },
            'lighting': {
                'poor': {'speed_limit': 0.2, 'detection_range': 1.0},
                'normal': {'speed_limit': 0.5, 'detection_range': 3.0},
                'bright': {'speed_limit': 0.5, 'detection_range': 3.0}
            }
        }
```

### Safety Risk Assessment

Assess and categorize safety risks:

```python
class SafetyRiskAssessor:
    def __init__(self):
        self.risk_matrix = {
            'collision': {'probability': 0.1, 'severity': 'high', 'risk_level': 'critical'},
            'human_collision': {'probability': 0.05, 'severity': 'critical', 'risk_level': 'critical'},
            'manipulation_failure': {'probability': 0.2, 'severity': 'medium', 'risk_level': 'high'},
            'navigation_failure': {'probability': 0.15, 'severity': 'low', 'risk_level': 'medium'}
        }

    def assess_action_risk(self, action, context):
        """Assess risk level of specific action"""
        action_type = action.get('action_type', 'unknown')

        if action_type in self.risk_matrix:
            risk_data = self.risk_matrix[action_type]
            return {
                'risk_level': risk_data['risk_level'],
                'probability': risk_data['probability'],
                'severity': risk_data['severity'],
                'mitigation_required': risk_data['risk_level'] in ['high', 'critical']
            }

        # Default low risk for unknown actions
        return {
            'risk_level': 'low',
            'probability': 0.01,
            'severity': 'low',
            'mitigation_required': False
        }

    def assess_plan_risk(self, plan, context):
        """Assess overall risk of plan"""
        total_risk = 0
        critical_risks = []

        for action in plan.get('actions', []):
            risk_assessment = self.assess_action_risk(action, context)
            risk_score = self.calculate_risk_score(risk_assessment)

            if risk_assessment['risk_level'] in ['high', 'critical']:
                critical_risks.append({
                    'action_id': action.get('id'),
                    'risk_assessment': risk_assessment
                })

            total_risk += risk_score

        return {
            'total_risk_score': total_risk,
            'critical_risks': critical_risks,
            'overall_risk_level': self.categorize_risk_level(total_risk),
            'approval_required': len(critical_risks) > 0
        }

    def calculate_risk_score(self, risk_assessment):
        """Calculate numerical risk score"""
        # Simple calculation: probability * severity factor
        severity_factors = {'low': 1, 'medium': 3, 'high': 5, 'critical': 10}
        severity_factor = severity_factors.get(risk_assessment['severity'], 1)
        return risk_assessment['probability'] * severity_factor
```

## Safety Validation Strategies

### Proactive Safety Validation

Validate safety before plan execution:

```python
class ProactiveSafetyValidator:
    def __init__(self):
        self.simulator = SafetySimulator()

    def validate_before_execution(self, plan, context):
        """Validate plan safety in simulation before execution"""
        # Run plan through safety simulation
        simulation_result = self.simulator.simulate_plan(plan, context)

        # Check for safety violations in simulation
        violations = []
        if simulation_result.has_human_safety_violations():
            violations.append("Human safety violations detected in simulation")

        if simulation_result.has_collision_risks():
            violations.append("Collision risks detected in simulation")

        if simulation_result.exceeds_force_limits():
            violations.append("Force limit violations detected in simulation")

        return len(violations) == 0, violations
```

### Continuous Safety Monitoring

Monitor safety throughout plan execution:

```python
class ContinuousSafetyMonitor:
    def __init__(self, node):
        self.node = node
        self.safety_validator = SafetyValidator()
        self.active_plan = None
        self.execution_context = {}

    def monitor_execution_safety(self, current_state, current_action):
        """Monitor safety during plan execution"""
        context = self.update_context(current_state)

        # Validate current action in current context
        is_safe, issues = self.safety_validator.validate_action_safety(
            current_action, context
        )

        if not is_safe:
            return {
                'violation': True,
                'issues': issues,
                'severity': self.assess_violation_severity(issues),
                'recommended_action': self.get_safety_response(issues)
            }

        return {'violation': False}
```

## Safety Response Mechanisms

### Emergency Response Procedures

Handle safety violations with appropriate responses:

```python
class SafetyResponseHandler:
    def __init__(self, node):
        self.node = node

    def handle_safety_violation(self, violation_type, severity, context):
        """Handle safety violation with appropriate response"""
        if severity == 'critical':
            return self.critical_response(violation_type, context)
        elif severity == 'high':
            return self.high_response(violation_type, context)
        elif severity == 'medium':
            return self.medium_response(violation_type, context)
        else:
            return self.low_response(violation_type, context)

    def critical_response(self, violation_type, context):
        """Critical safety violation response"""
        self.node.get_logger().error(f'CRITICAL SAFETY VIOLATION: {violation_type}')
        self.emergency_stop()
        self.activate_emergency_protocols()
        return 'EMERGENCY_STOP_ACTIVATED'

    def high_response(self, violation_type, context):
        """High severity safety violation response"""
        self.node.get_logger().warn(f'HIGH RISK VIOLATION: {violation_type}')
        self.slow_down_operations()
        self.request_human_intervention()
        return 'HUMAN_INTERVENTION_REQUESTED'

    def medium_response(self, violation_type, context):
        """Medium severity safety violation response"""
        self.node.get_logger().info(f'MEDIUM RISK VIOLATION: {violation_type}')
        self.adjust_behavior_safely()
        return 'BEHAVIOR_ADJUSTED'

    def low_response(self, violation_type, context):
        """Low severity safety violation response"""
        self.node.get_logger().debug(f'LOW RISK VIOLATION: {violation_type}')
        self.log_for_review()
        return 'LOGGED_FOR_REVIEW'

    def emergency_stop(self):
        """Execute emergency stop"""
        # Stop all robot motion immediately
        pass

    def activate_emergency_protocols(self):
        """Activate emergency safety protocols"""
        # Activate safety systems
        pass
```

## Testing Safety Validation

### Unit Tests

```python
import unittest
from unittest.mock import Mock

class TestSafetyValidation(unittest.TestCase):
    def setUp(self):
        self.validator = SafetyValidator()

    def test_collision_avoidance_validation(self):
        """Test collision avoidance safety validation"""
        # Create a plan with navigation action
        plan = {
            'actions': [
                {
                    'action_type': 'NavigateToPose',
                    'parameters': {'linear_velocity': 0.3}
                }
            ]
        }

        context = {
            'obstacles': [{'x': 1.0, 'y': 1.0, 'radius': 0.2}],
            'robot_position': {'x': 0.5, 'y': 0.5},
            'robot_radius': 0.3
        }

        is_safe, results = self.validator.validate_plan_safety(plan, context)

        # Should be safe if path doesn't collide with obstacles
        self.assertTrue(is_safe)

    def test_human_safety_validation(self):
        """Test human safety validation"""
        plan = {
            'actions': [
                {
                    'action_type': 'NavigateToPose',
                    'parameters': {'linear_velocity': 0.5}
                }
            ]
        }

        # Test with human nearby
        context_with_human = {
            'humans': [{'x': 0.8, 'y': 0.8}],
            'robot_position': {'x': 0.5, 'y': 0.5}
        }

        is_safe, results = self.validator.validate_plan_safety(plan, context_with_human)

        # Should be unsafe if too close to human
        self.assertFalse(is_safe)

    def test_velocity_limit_validation(self):
        """Test velocity limit safety validation"""
        plan = {
            'actions': [
                {
                    'action_type': 'NavigateToPose',
                    'parameters': {'linear_velocity': 2.0}  # Exceeds limit of 0.5
                }
            ]
        }

        context = {}
        is_safe, results = self.validator.validate_plan_safety(plan, context)

        # Should be unsafe due to excessive velocity
        self.assertFalse(is_safe)

    def test_manipulation_force_validation(self):
        """Test manipulation force safety validation"""
        plan = {
            'actions': [
                {
                    'action_type': 'GraspAction',
                    'parameters': {'force': 50.0}  # Exceeds limit of 30N
                }
            ]
        }

        context = {}
        is_safe, results = self.validator.validate_plan_safety(plan, context)

        # Should be unsafe due to excessive force
        self.assertFalse(is_safe)
```

## Integration with Human-in-the-Loop

### Safety Approval Workflow

Integrate safety validation with human oversight:

```python
class SafetyApprovalWorkflow:
    def __init__(self, node, safety_validator):
        self.node = node
        self.safety_validator = safety_validator

    def request_safety_approval(self, plan, risk_assessment):
        """Request human approval for potentially risky plan"""
        if risk_assessment['approval_required']:
            self.node.get_logger().info('Requesting safety approval for high-risk plan')

            # Create approval request with safety information
            approval_request = self.create_safety_approval_request(plan, risk_assessment)
            self.publish_approval_request(approval_request)

            # Wait for human approval
            return self.wait_for_safety_approval()

        return True  # No approval needed

    def create_safety_approval_request(self, plan, risk_assessment):
        """Create safety approval request with risk information"""
        request = f"""
        SAFETY APPROVAL REQUEST:
        Plan: {plan.get('original_goal', 'Unknown')}
        Critical Risks: {len(risk_assessment['critical_risks'])}
        Total Risk Score: {risk_assessment['total_risk_score']:.2f}
        Overall Risk Level: {risk_assessment['overall_risk_level']}

        Critical Risks:
        """
        for risk in risk_assessment['critical_risks']:
            request += f"- Action {risk['action_id']}: {risk['risk_assessment']['risk_level']} risk\n"

        request += "\nPlease respond with 'approve' or 'reject'."

        return request
```

Safety constraint validation is fundamental to creating reliable and trustworthy LLM-based planning systems for humanoid robotics. By implementing comprehensive safety validation mechanisms, we ensure that autonomous systems operate safely while maintaining their effectiveness and utility.