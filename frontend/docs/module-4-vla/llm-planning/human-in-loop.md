# Human-in-the-Loop Control for LLM-Based Planning

## Introduction

Human-in-the-loop (HITL) control is a critical component of LLM-based planning systems for humanoid robotics. It provides human oversight, intervention capabilities, and collaborative decision-making to ensure safe and appropriate robot behavior. This section covers the implementation of effective human-in-the-loop mechanisms for LLM-based cognitive planning.

## Importance of Human-in-the-Loop

### Safety Considerations
- **Critical Decision Validation**: Humans can validate potentially unsafe LLM-generated plans
- **Emergency Override**: Immediate human control when situations become dangerous
- **Ambiguity Resolution**: Humans can clarify ambiguous goals or situations
- **Ethical Oversight**: Human judgment for ethically complex scenarios

### System Reliability
- **Error Correction**: Humans can correct LLM mistakes
- **Adaptive Learning**: Human feedback improves system performance
- **Fallback Mechanism**: Human control when autonomous systems fail
- **Quality Assurance**: Human validation of plan quality

## Human-in-the-Loop Architecture

### HITL Control System

```python
class HumanInLoopController:
    def __init__(self, node):
        self.node = node
        self.human_override_active = False
        self.approval_required = False
        self.feedback_buffer = []

        # Subscribers for human input
        self.command_sub = node.create_subscription(
            String,
            'human_commands',
            self.human_command_callback,
            10
        )

        self.approval_sub = node.create_subscription(
            String,
            'human_approval',
            self.approval_callback,
            10
        )

        # Publishers for human interaction
        self.feedback_pub = node.create_publisher(
            String,
            'human_feedback',
            10
        )

        self.request_pub = node.create_publisher(
            String,
            'human_requests',
            10
        )

    def human_command_callback(self, msg):
        """Handle human commands for direct control"""
        command = msg.data.lower().strip()

        if command == 'override':
            self.activate_human_override()
        elif command == 'resume':
            self.deactivate_human_override()
        elif command == 'stop':
            self.emergency_stop()
        elif command.startswith('execute:'):
            # Human wants to execute specific action
            action = command[8:]  # Remove 'execute:' prefix
            self.execute_human_command(action)

    def approval_callback(self, msg):
        """Handle human approval decisions"""
        approval = msg.data.lower().strip()
        if approval == 'yes' or approval == 'approve':
            self.approval_required = False
            # Continue with plan execution
        elif approval == 'no' or approval == 'reject':
            self.approval_required = False
            # Reject current plan and request new one

    def activate_human_override(self):
        """Activate human override mode"""
        self.human_override_active = True
        self.node.get_logger().info('Human override activated')
        self.publish_feedback('Human override mode activated')

    def deactivate_human_override(self):
        """Deactivate human override mode"""
        self.human_override_active = False
        self.node.get_logger().info('Human override deactivated')
        self.publish_feedback('Human override mode deactivated')

    def emergency_stop(self):
        """Emergency stop for safety"""
        self.node.get_logger().warn('Emergency stop activated by human')
        # Stop all robot motion
        self.stop_robot_motion()
        self.publish_feedback('Emergency stop executed')

    def publish_feedback(self, message):
        """Publish feedback to human operator"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

    def publish_request(self, message):
        """Publish request to human operator"""
        request_msg = String()
        request_msg.data = message
        self.request_pub.publish(request_msg)
```

### Integration with Planning System

```python
class HITLPlanningIntegrator:
    def __init__(self, node, human_controller):
        self.node = node
        self.human_controller = human_controller
        self.approval_queue = []
        self.intervention_needed = False

    def check_human_approval_needed(self, plan):
        """Check if human approval is needed for plan"""
        # Check for safety-critical actions
        for action in plan.actions:
            if self.is_action_safety_critical(action):
                return True

        # Check for uncertain situations
        if plan.uncertainty_level > 0.7:
            return True

        # Check for high-impact actions
        if self.is_action_high_impact(plan):
            return True

        return False

    def is_action_safety_critical(self, action):
        """Check if action is safety-critical"""
        safety_critical_types = [
            'grasp_fragile_object',
            'navigate_near_human',
            'manipulate_kitchen_tool',
            'operate_appliance'
        ]

        return any(critical_type in action.action_type for critical_type in safety_critical_types)

    def is_action_high_impact(self, plan):
        """Check if plan has high impact"""
        high_impact_indicators = [
            'leave_building',
            'open_door_to_outside',
            'use_kitchen_appliance',
            'handle_valuable_item'
        ]

        for action in plan.actions:
            if any(indicator in action.description.lower() for indicator in high_impact_indicators):
                return True

        return False

    def request_human_approval(self, plan):
        """Request human approval for plan"""
        self.node.get_logger().info('Requesting human approval for plan')

        # Create approval request
        request_message = self.create_approval_request(plan)
        self.human_controller.publish_request(request_message)

        # Wait for approval
        self.approval_queue.append(plan)
        self.intervention_needed = True

        # Pause plan execution until approval
        return self.wait_for_approval(plan)

    def create_approval_request(self, plan):
        """Create human-readable approval request"""
        request = f"""
        Please approve the following plan:
        Goal: {plan.original_goal}
        Actions: {len(plan.actions)} actions
        Estimated time: {self.estimate_execution_time(plan)} minutes

        Action sequence:
        """
        for i, action in enumerate(plan.actions):
            request += f"{i+1}. {action.description}\n"

        request += "\nPlease respond with 'approve' or 'reject'."

        return request

    def wait_for_approval(self, plan):
        """Wait for human approval with timeout"""
        import time
        start_time = time.time()
        timeout = 30  # 30 seconds timeout

        while self.intervention_needed and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.intervention_needed:
            # Timeout - handle as rejection
            self.node.get_logger().warn('Approval request timed out')
            return False

        return True
```

## Types of Human Intervention

### 1. Pre-Execution Approval

Human approval before plan execution:

```python
class PreExecutionApprover:
    def __init__(self, node):
        self.node = node
        self.approval_thresholds = {
            'safety_risk': 0.8,  # High risk requires approval
            'resource_usage': 0.7,  # High resource usage requires approval
            'privacy_impact': 0.5   # Any privacy impact requires approval
        }

    def evaluate_approval_needed(self, plan):
        """Evaluate if pre-execution approval is needed"""
        risks = self.assess_plan_risks(plan)

        for risk_type, threshold in self.approval_thresholds.items():
            if risks.get(risk_type, 0) > threshold:
                return True, f"High {risk_type} risk requires approval"

        return False, "Plan approved automatically"
```

### 2. Real-Time Monitoring and Intervention

Continuous monitoring with ability to intervene:

```python
class RealTimeMonitor:
    def __init__(self, node):
        self.node = node
        self.monitor_timer = node.create_timer(0.5, self.monitor_callback)  # 2 Hz
        self.current_plan = None
        self.current_action = None

    def monitor_callback(self):
        """Monitor ongoing execution for intervention needs"""
        if self.current_plan and self.current_action:
            # Check if current action requires intervention
            if self.should_intervene(self.current_action):
                self.request_human_intervention()

    def should_intervene(self, action):
        """Check if intervention is needed for current action"""
        # Check for unexpected environmental changes
        if self.environment_changed_dramatically():
            return True

        # Check for safety violations
        if self.detect_safety_violation(action):
            return True

        # Check for plan deviation
        if self.plan_deviation_exceeds_threshold():
            return True

        return False

    def request_human_intervention(self):
        """Request immediate human intervention"""
        self.node.get_logger().warn('Requesting human intervention')
        # Publish alert to human operator
        alert_msg = String()
        alert_msg.data = "Immediate human intervention required!"
        # self.alert_publisher.publish(alert_msg)
```

### 3. Feedback and Learning

Human feedback for system improvement:

```python
class FeedbackProcessor:
    def __init__(self, node):
        self.node = node
        self.feedback_sub = node.create_subscription(
            String,
            'human_feedback',
            self.feedback_callback,
            10
        )

        self.feedback_buffer = []
        self.learning_enabled = True

    def feedback_callback(self, msg):
        """Process human feedback"""
        feedback = msg.data

        # Parse feedback type
        if self.is_positive_feedback(feedback):
            self.process_positive_feedback(feedback)
        elif self.is_negative_feedback(feedback):
            self.process_negative_feedback(feedback)
        elif self.is_suggestion_feedback(feedback):
            self.process_suggestion_feedback(feedback)

    def process_negative_feedback(self, feedback):
        """Process negative feedback for learning"""
        self.node.get_logger().info(f'Received negative feedback: {feedback}')

        # Update planning model based on feedback
        if self.learning_enabled:
            self.update_planning_model(feedback)

        # Log for analysis
        self.log_feedback(feedback, 'negative')

    def update_planning_model(self, feedback):
        """Update planning model based on human feedback"""
        # Implementation to update LLM planning based on feedback
        pass
```

## Implementation of Human Interfaces

### Voice-Based Interface

Voice commands for human interaction:

```python
class VoiceHITLInterface:
    def __init__(self, node):
        self.node = node
        self.voice_command_sub = node.create_subscription(
            String,
            'transcribed_voice_commands',
            self.voice_command_callback,
            10
        )

        self.voice_response_pub = node.create_publisher(
            String,
            'voice_responses',
            10
        )

        self.voice_commands = {
            'approve': ['approve', 'yes', 'okay', 'go ahead'],
            'reject': ['reject', 'no', 'stop', 'cancel'],
            'override': ['override', 'take control', 'manual mode'],
            'resume': ['resume', 'continue', 'go on'],
            'emergency': ['emergency stop', 'stop immediately', 'emergency']
        }

    def voice_command_callback(self, msg):
        """Process voice commands"""
        command = msg.data.lower().strip()

        for command_type, keywords in self.voice_commands.items():
            if any(keyword in command for keyword in keywords):
                self.execute_voice_command(command_type, command)
                break

    def execute_voice_command(self, command_type, original_command):
        """Execute voice command"""
        if command_type == 'approve':
            self.handle_approve_command()
        elif command_type == 'reject':
            self.handle_reject_command()
        elif command_type == 'override':
            self.handle_override_command()
        elif command_type == 'resume':
            self.handle_resume_command()
        elif command_type == 'emergency':
            self.handle_emergency_command()

        # Provide voice confirmation
        response = f"Command {command_type} received and executed"
        self.publish_voice_response(response)

    def publish_voice_response(self, message):
        """Publish voice response"""
        response_msg = String()
        response_msg.data = message
        self.voice_response_pub.publish(response_msg)
```

### Mobile Application Interface

Mobile app for human oversight:

```python
class MobileHITLInterface:
    def __init__(self, node):
        self.node = node
        # This would typically interface with a web server or REST API
        # For simulation purposes, we'll create a mock interface
        self.active_sessions = {}

    def create_approval_request(self, plan_id, plan_description):
        """Create approval request for mobile app"""
        request = {
            'plan_id': plan_id,
            'description': plan_description,
            'actions': plan_description.get('actions', []),
            'estimated_time': self.estimate_time(plan_description),
            'risk_level': self.assess_risk(plan_description)
        }

        # Send to mobile app (simulated)
        self.send_to_mobile_app('approval_request', request)

    def handle_mobile_approval(self, plan_id, approval_status):
        """Handle approval from mobile app"""
        if approval_status == 'approved':
            self.node.get_logger().info(f'Plan {plan_id} approved via mobile')
            # Continue plan execution
        elif approval_status == 'rejected':
            self.node.get_logger().info(f'Plan {plan_id} rejected via mobile')
            # Cancel plan execution
```

## HITL Integration Node

### Main Implementation

```python
#!/usr/bin/env python3
# ROS 2 node for Human-in-the-Loop control

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import TaskPlan, HumanIntervention
from openai import OpenAI  # Example LLM client

class HumanInLoopControlNode(Node):
    def __init__(self):
        super().__init__('human_in_loop_control_node')

        # Initialize components
        self.human_controller = HumanInLoopController(self)
        self.planning_integrator = HITLPlanningIntegrator(self, self.human_controller)
        self.pre_approver = PreExecutionApprover(self)
        self.real_time_monitor = RealTimeMonitor(self)
        self.feedback_processor = FeedbackProcessor(self)
        self.voice_interface = VoiceHITLInterface(self)

        # Publishers and subscribers
        self.plan_sub = self.create_subscription(
            TaskPlan,
            'task_plans',
            self.plan_callback,
            10
        )

        self.intervention_pub = self.create_publisher(
            HumanIntervention,
            'human_interventions',
            10
        )

        self.execution_status_sub = self.create_subscription(
            String,
            'execution_status',
            self.execution_status_callback,
            10
        )

        # State tracking
        self.active_plan = None
        self.human_override_mode = False

        self.get_logger().info('Human-in-the-Loop Control Node initialized')

    def plan_callback(self, msg):
        """Process incoming task plan with HITL considerations"""
        try:
            self.get_logger().info(f'Received plan with {len(msg.actions)} actions')

            # Check if human approval is needed
            needs_approval, reason = self.pre_approver.evaluate_approval_needed(msg)

            if needs_approval:
                self.get_logger().info(f'Plan requires human approval: {reason}')

                # Request human approval
                approved = self.planning_integrator.request_human_approval(msg)

                if not approved:
                    self.get_logger().info('Plan rejected by human')
                    self.publish_intervention('plan_rejected', 'Plan rejected by human operator')
                    return

            # Check for human override
            if self.human_controller.human_override_active:
                self.get_logger().info('Human override active, not executing plan')
                self.publish_intervention('override_active', 'Human override mode active')
                return

            # Execute the plan
            self.active_plan = msg
            self.get_logger().info('Plan approved and executing')

        except Exception as e:
            self.get_logger().error(f'Error in plan processing: {e}')
            self.publish_intervention('error', f'Plan processing error: {e}')

    def execution_status_callback(self, msg):
        """Monitor execution status for intervention needs"""
        status = msg.data

        # Monitor for issues that might require human intervention
        if 'error' in status.lower() or 'failure' in status.lower():
            self.get_logger().warn(f'Execution issue detected: {status}')
            self.request_human_intervention('execution_error', status)

    def request_human_intervention(self, intervention_type, details):
        """Request human intervention for specific issue"""
        intervention_msg = HumanIntervention()
        intervention_msg.type = intervention_type
        intervention_msg.details = details
        intervention_msg.timestamp = self.get_clock().now().to_msg()
        intervention_msg.plan_id = self.active_plan.id if self.active_plan else 'unknown'

        self.intervention_pub.publish(intervention_msg)

        # Provide feedback to human operator
        feedback_msg = String()
        feedback_msg.data = f"Human intervention required: {details}"
        self.human_controller.feedback_pub.publish(feedback_msg)

    def publish_intervention(self, intervention_type, details):
        """Publish intervention message"""
        intervention_msg = HumanIntervention()
        intervention_msg.type = intervention_type
        intervention_msg.details = details
        intervention_msg.timestamp = self.get_clock().now().to_msg()
        intervention_msg.plan_id = self.active_plan.id if self.active_plan else 'unknown'

        self.intervention_pub.publish(intervention_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanInLoopControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Human-in-the-Loop control node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced HITL Features

### Adaptive HITL System

System that adapts to human preferences and availability:

```python
class AdaptiveHITLSystem:
    def __init__(self, node):
        self.node = node
        self.human_availability = True
        self.human_preferences = {}
        self.performance_history = []

    def adapt_to_human_preferences(self):
        """Adapt system behavior based on human preferences"""
        # Learn from feedback patterns
        feedback_patterns = self.analyze_feedback_patterns()

        # Adjust approval thresholds based on human preferences
        if feedback_patterns.get('approval_rate', 0.8) > 0.9:
            # Human is very approving, could reduce approval requirements
            self.adjust_approval_thresholds('reduce')
        elif feedback_patterns.get('approval_rate', 0.8) < 0.5:
            # Human is very cautious, maintain strict requirements
            self.adjust_approval_thresholds('increase')

    def analyze_feedback_patterns(self):
        """Analyze patterns in human feedback"""
        # Implementation to analyze feedback patterns
        return {'approval_rate': 0.8}  # Placeholder
```

### Multi-Human Coordination

Support for multiple human operators:

```python
class MultiHumanCoordinator:
    def __init__(self, node):
        self.node = node
        self.operators = {}
        self.operator_availability = {}
        self.role_assignments = {}

    def assign_role(self, operator_id, role):
        """Assign role to human operator"""
        self.role_assignments[operator_id] = role

    def get_available_operator(self, required_role):
        """Get available operator for specific role"""
        for operator_id, is_available in self.operator_availability.items():
            if is_available and self.role_assignments.get(operator_id) == required_role:
                return operator_id

        return None

    def escalate_to_expert(self, situation):
        """Escalate to expert operator when needed"""
        expert_operator = self.get_available_operator('expert')
        if expert_operator:
            self.request_expert_intervention(expert_operator, situation)
        else:
            self.use_default_procedures(situation)
```

## Safety and Reliability Features

### Fail-Safe Mechanisms

Ensure safety when human input is unavailable:

```python
class FailSafeHITL:
    def __init__(self, node):
        self.node = node
        self.emergency_procedures = {
            'safety_violation': self.safety_emergency_stop,
            'communication_loss': self.communication_recovery,
            'timeout': self.timeout_recovery
        }

    def safety_emergency_stop(self):
        """Execute safety emergency stop"""
        self.node.get_logger().error('Safety emergency stop activated')
        # Stop all robot motion
        self.stop_all_robot_motion()
        # Activate safety protocols

    def communication_recovery(self):
        """Recover from communication loss with human"""
        # Switch to conservative operation mode
        self.switch_to_conservative_mode()
        # Attempt to reestablish communication
        self.attempt_reconnection()

    def timeout_recovery(self):
        """Handle timeout in human response"""
        # Switch to safe default behavior
        self.return_to_safe_state()
```

### Decision Trees for Intervention

Systematic approach to determine when to involve humans:

```python
class InterventionDecisionTree:
    def __init__(self):
        self.decision_rules = [
            {
                'condition': lambda plan: self.is_plan_high_risk(plan),
                'action': 'request_approval',
                'priority': 'high'
            },
            {
                'condition': lambda plan: self.is_environment_uncertain(plan),
                'action': 'request_clarification',
                'priority': 'medium'
            },
            {
                'condition': lambda plan: self.is_human_nearby(),
                'action': 'notify_human',
                'priority': 'low'
            }
        ]

    def determine_intervention_needed(self, plan, context):
        """Determine what type of intervention is needed"""
        interventions = []

        for rule in self.decision_rules:
            if rule['condition'](plan):
                interventions.append({
                    'action': rule['action'],
                    'priority': rule['priority']
                })

        return self.prioritize_interventions(interventions)

    def prioritize_interventions(self, interventions):
        """Prioritize multiple interventions"""
        priority_map = {'high': 3, 'medium': 2, 'low': 1}
        return sorted(interventions, key=lambda x: priority_map[x['priority']], reverse=True)
```

## Performance Optimization

### Efficient Human Interaction

Optimize for minimal human burden while maintaining safety:

```python
class EfficientHITL:
    def __init__(self, node):
        self.node = node
        self.human_interaction_count = 0
        self.human_satisfaction_score = 0.0

    def minimize_human_interactions(self):
        """Minimize unnecessary human interactions"""
        # Use machine learning to predict when human input is truly needed
        # Implement intelligent batching of requests
        # Learn from human response patterns to optimize timing
        pass

    def measure_human_satisfaction(self):
        """Measure and improve human satisfaction"""
        # Track response times
        # Monitor approval rates
        # Gather explicit feedback
        pass
```

## Testing Human-in-the-Loop Systems

### Unit Tests

```python
import unittest

class TestHumanInLoopControl(unittest.TestCase):
    def setUp(self):
        self.mock_node = MockNode()
        self.human_controller = HumanInLoopController(self.mock_node)
        self.planning_integrator = HITLPlanningIntegrator(self.mock_node, self.human_controller)

    def test_human_override_activation(self):
        """Test human override activation"""
        # Simulate human override command
        override_msg = String()
        override_msg.data = 'override'
        self.human_controller.human_command_callback(override_msg)

        self.assertTrue(self.human_controller.human_override_active)

    def test_approval_process(self):
        """Test human approval process"""
        # Create a safety-critical plan
        plan = MockTaskPlan()
        plan.actions = [{'action_type': 'navigate_near_human', 'description': 'Navigate near human'}]

        # Check if approval is requested
        needs_approval, reason = self.planning_integrator.check_human_approval_needed(plan)
        self.assertTrue(needs_approval)

    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        stop_msg = String()
        stop_msg.data = 'stop'
        self.human_controller.human_command_callback(stop_msg)

        # Verify emergency stop was executed
        self.assertTrue(self.human_controller.emergency_stop_called)

    def test_voice_command_processing(self):
        """Test voice command processing"""
        voice_interface = VoiceHITLInterface(self.mock_node)

        # Test approve command
        approve_msg = String()
        approve_msg.data = 'yes, approve this plan'
        voice_interface.voice_command_callback(approve_msg)

        # Verify approve command was processed
        self.assertEqual(voice_interface.last_command, 'approve')
```

## Integration with Safety Systems

### Safety System Coordination

Coordinate with safety systems for comprehensive protection:

```python
class SafetyHITLCoordinator:
    def __init__(self, node):
        self.node = node
        # Interface with safety system
        self.safety_client = node.create_client(
            Trigger,  # Example safety service
            'safety_override'
        )

    def coordinate_with_safety_system(self, intervention_needed):
        """Coordinate HITL with safety system"""
        if intervention_needed:
            # Request safety system to pause autonomous operations
            safety_request = Trigger.Request()
            safety_response = self.safety_client.call(safety_request)

            if safety_response.success:
                self.node.get_logger().info('Safety system paused, human control enabled')
                return True
            else:
                self.node.get_logger().error('Failed to coordinate with safety system')
                return False

        return True
```

Human-in-the-loop control is essential for safe and reliable LLM-based planning in humanoid robotics. By providing appropriate mechanisms for human oversight, intervention, and feedback, we create systems that leverage the strengths of both artificial intelligence and human judgment.