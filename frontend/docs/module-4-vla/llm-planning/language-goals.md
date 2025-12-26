# Translating Language Goals to ROS 2 Action Sequences

## Introduction

Translating high-level language goals into executable ROS 2 action sequences is a critical capability in Vision-Language-Action (VLA) systems. This process involves converting natural language instructions into specific ROS 2 messages, services, and actions that can control the robot. This section covers techniques for effective language-to-action translation using LLMs.

## Understanding Language Goals

### Goal Types and Categories

Language goals can be categorized into different types based on their complexity and required robot capabilities:

#### 1. Navigation Goals
Goals that primarily involve robot movement:
- "Go to the kitchen"
- "Navigate to the charging station"
- "Follow me to the conference room"

#### 2. Manipulation Goals
Goals that involve object interaction:
- "Pick up the red cup and put it on the table"
- "Open the door"
- "Hand me the book"

#### 3. Perception Goals
Goals that require sensing and recognition:
- "Find all blue objects in the room"
- "Tell me what you see on the table"
- "Count the number of chairs"

#### 4. Composite Goals
Complex goals combining multiple capabilities:
- "Go to the kitchen, find a clean glass, fill it with water, and bring it to me"
- "Clean the table and then charge yourself"

## Language-to-Action Translation Pipeline

### 1. Goal Understanding

The first step is understanding the user's intent from the language goal:

```python
class LanguageGoalUnderstanding:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def understand_goal(self, goal_text):
        """Analyze and understand the language goal"""
        prompt = f"""
        Analyze the following language goal and extract key information:

        Goal: {goal_text}

        Extract the following information:
        1. Main objective
        2. Required robot capabilities (navigation, manipulation, perception, etc.)
        3. Objects involved
        4. Locations involved
        5. Sequence of operations needed
        6. Safety considerations
        7. Expected outcome

        Format as JSON with these fields.
        """

        response = self.llm_client.generate(prompt)
        return self.parse_understanding(response)
```

### 2. Action Mapping

Map understood goals to specific ROS 2 actions:

```python
class ActionMapper:
    def __init__(self):
        self.action_mappings = {
            'navigation': {
                'go to': 'NavigateToPose',
                'move to': 'NavigateToPose',
                'navigate to': 'NavigateToPose',
                'follow': 'FollowWaypoints'
            },
            'manipulation': {
                'pick up': 'PickPlaceAction',
                'grasp': 'GraspAction',
                'place': 'PlaceAction',
                'open': 'OpenGripperAction',
                'close': 'CloseGripperAction'
            },
            'perception': {
                'find': 'FindObjectAction',
                'detect': 'DetectObjectAction',
                'recognize': 'RecognizeObjectAction'
            }
        }

    def map_to_ros_actions(self, understood_goal):
        """Map understood goal to ROS 2 actions"""
        main_objective = understood_goal.get('main_objective', '').lower()
        required_capabilities = understood_goal.get('required_capabilities', [])

        actions = []
        for capability in required_capabilities:
            if capability in self.action_mappings:
                capability_actions = self.action_mappings[capability]
                for keyword, action_type in capability_actions.items():
                    if keyword in main_objective:
                        action = self.create_ros_action(action_type, understood_goal)
                        actions.append(action)

        return actions
```

### 3. Parameter Extraction

Extract parameters needed for ROS 2 action execution:

```python
class ParameterExtractor:
    def __init__(self):
        self.location_keywords = ['kitchen', 'bedroom', 'office', 'table', 'chair']
        self.object_keywords = ['cup', 'book', 'ball', 'box', 'bottle']
        self.quantity_keywords = ['one', 'two', 'three', 'several', 'all']

    def extract_parameters(self, goal_text, understood_goal):
        """Extract parameters for ROS 2 actions"""
        parameters = {}

        # Extract locations
        for location in self.location_keywords:
            if location in goal_text.lower():
                parameters['location'] = location

        # Extract objects
        for obj in self.object_keywords:
            if obj in goal_text.lower():
                parameters['object'] = obj

        # Extract quantities
        for qty in self.quantity_keywords:
            if qty in goal_text.lower():
                parameters['quantity'] = qty

        # Extract additional parameters from context
        context_params = understood_goal.get('additional_parameters', {})
        parameters.update(context_params)

        return parameters
```

## Complete Language-to-Action Translation System

### Main Translation Node

```python
#!/usr/bin/env python3
# ROS 2 node for translating language goals to ROS 2 actions

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from vla_interfaces.msg import LanguageGoal, ActionSequence
from openai import OpenAI  # Example LLM client

class LanguageToActionTranslator(Node):
    def __init__(self):
        super().__init__('language_to_action_translator')

        # Initialize LLM client
        self.llm_client = OpenAI(api_key=self.get_parameter_or('openai_api_key', ''))

        # Initialize components
        self.goal_understanding = LanguageGoalUnderstanding(self.llm_client)
        self.action_mapper = ActionMapper()
        self.parameter_extractor = ParameterExtractor()

        # Publishers and subscribers
        self.goal_sub = self.create_subscription(
            String,
            'language_goals',
            self.language_goal_callback,
            10
        )

        self.action_sequence_pub = self.create_publisher(
            ActionSequence,
            'action_sequences',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'translation_feedback',
            10
        )

        self.get_logger().info('Language to Action Translator initialized')

    def language_goal_callback(self, msg):
        """Process incoming language goal"""
        try:
            goal_text = msg.data
            self.get_logger().info(f'Received language goal: {goal_text}')

            # Step 1: Understand the goal
            understood_goal = self.goal_understanding.understand_goal(goal_text)

            # Step 2: Extract parameters
            parameters = self.parameter_extractor.extract_parameters(goal_text, understood_goal)

            # Step 3: Map to ROS 2 actions
            ros_actions = self.action_mapper.map_to_ros_actions(understood_goal)

            # Step 4: Create action sequence
            action_sequence = self.create_action_sequence(ros_actions, parameters, goal_text)

            # Step 5: Validate the sequence
            if self.validate_action_sequence(action_sequence):
                # Publish the action sequence
                self.action_sequence_pub.publish(action_sequence)
                self.publish_feedback(f'Action sequence created successfully for: {goal_text}')
                self.get_logger().info(f'Published action sequence with {len(action_sequence.actions)} actions')
            else:
                self.publish_feedback(f'Action sequence validation failed for: {goal_text}')
                self.get_logger().error('Action sequence validation failed')

        except Exception as e:
            self.get_logger().error(f'Error in language to action translation: {e}')
            self.publish_feedback(f'Translation error: {e}')

    def create_action_sequence(self, ros_actions, parameters, original_goal):
        """Create ROS 2 action sequence message"""
        sequence_msg = ActionSequence()
        sequence_msg.original_goal = original_goal
        sequence_msg.timestamp = self.get_clock().now().to_msg()

        for i, action_data in enumerate(ros_actions):
            action_msg = ActionSequence.Action()
            action_msg.id = i + 1
            action_msg.action_type = action_data.get('action_type', '')
            action_msg.description = action_data.get('description', '')

            # Add parameters
            for param_name, param_value in parameters.items():
                param = ActionSequence.Parameter()
                param.name = param_name
                param.value = str(param_value)
                action_msg.parameters.append(param)

            # Set dependencies
            if i > 0:
                action_msg.dependencies = [i]  # Depends on previous action

            sequence_msg.actions.append(action_msg)

        return sequence_msg

    def validate_action_sequence(self, sequence):
        """Validate action sequence for safety and feasibility"""
        # Check for empty sequence
        if len(sequence.actions) == 0:
            return False

        # Check for valid action types
        valid_action_types = ['NavigateToPose', 'PickPlaceAction', 'FindObjectAction',
                             'FollowWaypoints', 'GraspAction', 'PlaceAction']

        for action in sequence.actions:
            if action.action_type not in valid_action_types:
                self.get_logger().warn(f'Invalid action type: {action.action_type}')
                return False

        # Additional validation checks can be added here
        return True

    def publish_feedback(self, message):
        """Publish feedback message"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LanguageToActionTranslator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down language to action translator')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Translation Techniques

### Semantic Role Labeling

Use semantic role labeling to better understand goal structure:

```python
class SemanticRoleLabeler:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def label_semantic_roles(self, goal_text):
        """Label semantic roles in the goal text"""
        prompt = f"""
        Perform semantic role labeling on the following sentence:

        Sentence: {goal_text}

        Identify the following semantic roles:
        - Agent (who performs the action)
        - Patient (what is affected by the action)
        - Theme (what is moved or affected)
        - Goal (destination or target)
        - Source (starting point)
        - Instrument (tool used)
        - Location (where action occurs)

        Format as JSON with role-label pairs.
        """

        response = self.llm_client.generate(prompt)
        return self.parse_semantic_roles(response)
```

### Context-Aware Translation

Consider the current context when translating goals:

```python
class ContextAwareTranslator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.current_context = {}

    def translate_with_context(self, goal_text, context):
        """Translate goal considering current context"""
        self.current_context = context

        context_prompt = f"""
        Translate the following language goal to ROS 2 actions, considering the current context:

        Current Context:
        - Robot Location: {context.get('robot_location', 'unknown')}
        - Available Objects: {context.get('available_objects', [])}
        - Known Locations: {context.get('known_locations', [])}
        - Robot Capabilities: {context.get('capabilities', [])}
        - Safety Constraints: {context.get('safety_constraints', [])}

        Goal: {goal_text}

        Provide appropriate ROS 2 action sequence considering the context.
        """

        response = self.llm_client.generate(context_prompt)
        return self.parse_action_sequence(response)
```

## Handling Ambiguous Goals

### Clarification Requests

When goals are ambiguous, request clarification:

```python
class AmbiguityResolver:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def check_ambiguity(self, goal_text):
        """Check if goal is ambiguous and needs clarification"""
        prompt = f"""
        Analyze the following goal for ambiguity:

        Goal: {goal_text}

        Identify potential ambiguities such as:
        - Unclear object references
        - Vague locations
        - Multiple possible interpretations
        - Missing information needed for execution

        Return JSON with:
        - 'is_ambiguous': boolean
        - 'ambiguities': list of identified ambiguities
        - 'clarification_questions': list of questions to resolve ambiguities
        """

        response = self.llm_client.generate(prompt)
        return self.parse_ambiguity_analysis(response)

    def generate_clarification_request(self, ambiguities):
        """Generate clarification request based on identified ambiguities"""
        questions = ambiguities.get('clarification_questions', [])
        if questions:
            return "I need clarification: " + "; ".join(questions)
        return None
```

## Error Handling and Recovery

### Translation Error Handling

Handle cases where translation fails:

```python
class TranslationErrorHandler:
    def __init__(self):
        self.error_recovery_strategies = {
            'unknown_action': self.handle_unknown_action,
            'invalid_parameters': self.handle_invalid_parameters,
            'unsafe_action': self.handle_unsafe_action,
            'communication_error': self.handle_communication_error
        }

    def handle_translation_error(self, error_type, goal_text, error_details):
        """Handle different types of translation errors"""
        if error_type in self.error_recovery_strategies:
            return self.error_recovery_strategies[error_type](goal_text, error_details)
        else:
            return self.default_error_handling(goal_text, error_details)

    def handle_unknown_action(self, goal_text, details):
        """Handle unknown action types"""
        return {
            'status': 'REQUIRES_HUMAN_INTERVENTION',
            'message': f'Unknown action in goal: {goal_text}',
            'suggested_alternatives': self.get_alternative_actions(goal_text)
        }
```

## Integration with Planning Systems

### Coordination with Task Planners

Coordinate with higher-level task planning systems:

```python
class PlanningCoordinator:
    def __init__(self, node):
        self.node = node
        self.translation_queue = []
        self.planning_queue = []

    def coordinate_translation_and_planning(self, goal_text):
        """Coordinate between language translation and task planning"""
        # First, translate the language goal
        action_sequence = self.translate_language_goal(goal_text)

        # Then, pass to task planner for refinement
        refined_plan = self.refine_with_task_planner(action_sequence)

        # Finally, execute or schedule the plan
        return self.execute_or_schedule(refined_plan)

    def refine_with_task_planner(self, action_sequence):
        """Refine action sequence with task planning system"""
        # Integrate with existing task planning system
        # Add safety checks, optimize sequence, resolve conflicts
        pass
```

## Performance Optimization

### Caching Translations

Cache frequently used translations:

```python
from functools import lru_cache

class CachedTranslator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.cache_size = 1000

    @lru_cache(maxsize=1000)
    def cached_translate(self, goal_text):
        """Cached translation of language goals"""
        return self.translate_language_goal(goal_text)
```

### Batch Processing

Process multiple goals efficiently:

```python
class BatchTranslator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def translate_batch(self, goal_list):
        """Translate multiple goals in batch"""
        # Use batch processing capabilities of LLM
        batch_prompt = "Translate the following goals to ROS 2 action sequences:\n\n"
        for i, goal in enumerate(goal_list):
            batch_prompt += f"{i+1}. {goal}\n"

        batch_prompt += "\nProvide all translations in a single response with clear separation."

        response = self.llm_client.generate(batch_prompt)
        return self.parse_batch_response(response)
```

## Testing and Validation

### Unit Tests

```python
import unittest

class TestLanguageToActionTranslation(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MockLLMClient()
        self.translator = LanguageToActionTranslator(self.mock_llm_client)

    def test_simple_navigation_translation(self):
        """Test translation of simple navigation goal"""
        goal = "Go to the kitchen"
        sequence = self.translator.translate_language_goal(goal)

        self.assertEqual(len(sequence.actions), 1)
        self.assertEqual(sequence.actions[0].action_type, 'NavigateToPose')

    def test_manipulation_translation(self):
        """Test translation of manipulation goal"""
        goal = "Pick up the red ball"
        sequence = self.translator.translate_language_goal(goal)

        self.assertGreater(len(sequence.actions), 0)
        # Should include navigation to object, then manipulation

    def test_complex_goal_translation(self):
        """Test translation of complex goal with multiple steps"""
        goal = "Go to the kitchen, find a cup, pick it up, and bring it to me"
        sequence = self.translator.translate_language_goal(goal)

        # Should have multiple actions for navigation, perception, and manipulation
        action_types = [action.action_type for action in sequence.actions]
        self.assertIn('NavigateToPose', action_types)
        self.assertIn('FindObjectAction', action_types)
        self.assertIn('GraspAction', action_types)
```

## Safety Considerations

### Safety Validation

Validate translated actions for safety:

```python
class SafetyValidator:
    def __init__(self):
        self.safety_rules = {
            'max_speed': 1.0,  # m/s
            'max_force': 50.0,  # Newtons
            'min_distance': 0.5,  # meters from humans
            'safe_locations': ['kitchen', 'living_room', 'office']  # Safe areas
        }

    def validate_translated_actions(self, action_sequence):
        """Validate translated actions for safety"""
        issues = []

        for action in action_sequence.actions:
            if action.action_type == 'NavigateToPose':
                # Check if destination is in safe location
                if not self.is_safe_destination(action.parameters):
                    issues.append(f"Unsafe destination in action: {action.id}")

            elif action.action_type == 'GraspAction':
                # Check if object is safe to grasp
                if not self.is_safe_to_grasp(action.parameters):
                    issues.append(f"Unsafe to grasp object in action: {action.id}")

        return len(issues) == 0, issues
```

The translation of language goals to ROS 2 action sequences enables humanoid robots to understand and execute complex commands expressed in natural language. By combining LLM capabilities with structured action mapping and safety validation, we create systems that can interpret high-level goals and convert them into executable robot behaviors.