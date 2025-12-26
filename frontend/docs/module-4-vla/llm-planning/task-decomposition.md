# Task Decomposition with LLMs

## Introduction

Task decomposition is the process of breaking down complex, high-level goals into sequences of executable subtasks. Large Language Models (LLMs) excel at this cognitive function, making them ideal for robotic planning applications. This section explores how to leverage LLMs for effective task decomposition in humanoid robotics.

## Understanding Task Decomposition

Task decomposition involves analyzing a high-level goal and identifying the sequence of subtasks required to achieve it. For example, the goal "Go to the kitchen, get a cup, and bring it to me" decomposes into:

1. Navigate to the kitchen
2. Identify and locate a cup
3. Approach and grasp the cup
4. Navigate back to the user
5. Deliver the cup to the user

### Key Elements of Task Decomposition

- **Goal Analysis**: Understanding the overall objective
- **Subtask Identification**: Recognizing individual steps
- **Dependency Mapping**: Determining task order and dependencies
- **Resource Assessment**: Identifying required capabilities and resources
- **Constraint Consideration**: Accounting for safety and operational constraints

## LLM-Based Task Decomposition Approaches

### 1. Prompt Engineering Approach

Using carefully crafted prompts to guide LLMs in task decomposition:

```python
class LLMTaskDecomposer:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def decompose_task(self, goal_description):
        """Decompose a high-level goal using LLM"""
        prompt = f"""
        You are a task decomposition expert for humanoid robotics. Given a high-level goal,
        break it down into specific, executable subtasks that a robot can perform.

        Goal: {goal_description}

        Decompose this goal into a sequence of subtasks. Each subtask should be:
        1. Specific and actionable
        2. Executable by a humanoid robot
        3. In logical order
        4. Include any necessary parameters

        Format the response as a JSON list with the following structure:
        {{
            "subtasks": [
                {{
                    "id": 1,
                    "name": "subtask_name",
                    "description": "detailed description",
                    "type": "navigation|manipulation|perception|system",
                    "parameters": {{"param_name": "value"}},
                    "dependencies": [list_of_subtask_ids]
                }}
            ]
        }}

        Be specific about locations, objects, and actions.
        """

        response = self.llm_client.generate(prompt)
        return self.parse_response(response)

    def parse_response(self, response):
        """Parse LLM response and validate structure"""
        try:
            import json
            data = json.loads(response)
            return data.get('subtasks', [])
        except json.JSONDecodeError:
            # Handle malformed response
            return self.fallback_decomposition(response)
```

### 2. Chain-of-Thought Reasoning

Using chain-of-thought prompting for more detailed decomposition:

```python
def decompose_with_chain_of_thought(self, goal_description):
    """Use chain-of-thought reasoning for task decomposition"""
    prompt = f"""
    Let's decompose the following robotic task step by step:

    Goal: {goal_description}

    Step 1: Analyze the goal components
    - What is the final objective?
    - What intermediate states are required?
    - What resources are needed?

    Step 2: Identify major phases
    - What are the main phases of this task?
    - How do they connect to achieve the goal?

    Step 3: Break down each phase
    - What specific actions are needed in each phase?
    - What are the preconditions and postconditions?

    Step 4: Sequence the actions
    - In what order should the actions be performed?
    - What dependencies exist between actions?

    Step 5: Add safety considerations
    - What safety checks are needed?
    - What constraints must be respected?

    Now provide the final decomposition in the JSON format specified earlier.
    """

    response = self.llm_client.generate(prompt)
    return self.parse_response(response)
```

### 3. Few-Shot Learning

Providing examples to guide the LLM:

```python
def decompose_with_examples(self, goal_description):
    """Use few-shot learning with examples"""
    examples = [
        {
            "goal": "Go to the kitchen and bring me a glass of water",
            "decomposition": [
                {
                    "id": 1,
                    "name": "navigate_to_kitchen",
                    "description": "Navigate to the kitchen area",
                    "type": "navigation",
                    "parameters": {"location": "kitchen"},
                    "dependencies": []
                },
                {
                    "id": 2,
                    "name": "find_glass",
                    "description": "Locate a glass in the kitchen",
                    "type": "perception",
                    "parameters": {"object_type": "glass"},
                    "dependencies": [1]
                },
                {
                    "id": 3,
                    "name": "grasp_glass",
                    "description": "Pick up the glass",
                    "type": "manipulation",
                    "parameters": {"object_id": "glass_1"},
                    "dependencies": [2]
                },
                {
                    "id": 4,
                    "name": "navigate_to_user",
                    "description": "Return to the user location",
                    "type": "navigation",
                    "parameters": {"location": "user"},
                    "dependencies": [3]
                },
                {
                    "id": 5,
                    "name": "offer_glass",
                    "description": "Present the glass to the user",
                    "type": "manipulation",
                    "parameters": {"action": "offer"},
                    "dependencies": [4]
                }
            ]
        }
    ]

    prompt = f"""
    You are a task decomposition expert for humanoid robotics. Here are some examples of how to decompose complex goals:

    Example 1:
    Goal: {examples[0]['goal']}
    Decomposition: {examples[0]['decomposition']}

    Now decompose the following goal in the same format:

    Goal: {goal_description}

    Decomposition:
    """

    response = self.llm_client.generate(prompt)
    return self.parse_response(response)
```

## Integration with ROS 2

### Task Decomposition Node

```python
#!/usr/bin/env python3
# ROS 2 node for LLM-based task decomposition

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import TaskPlan, Task
from openai import OpenAI  # Example LLM client

class LLMTaskDecompositionNode(Node):
    def __init__(self):
        super().__init__('llm_task_decomposition_node')

        # Initialize LLM client
        self.llm_client = OpenAI(api_key=self.get_parameter_or('openai_api_key', ''))

        # Initialize task decomposer
        self.decomposer = LLMTaskDecomposer(self.llm_client)

        # Subscribe to high-level goals
        self.goal_sub = self.create_subscription(
            String,
            'high_level_goals',
            self.goal_callback,
            10
        )

        # Publish task plans
        self.plan_pub = self.create_publisher(
            TaskPlan,
            'decomposed_tasks',
            10
        )

        # Subscribe to current robot state
        self.state_sub = self.create_subscription(
            String,
            'robot_state',
            self.state_callback,
            10
        )

        self.current_state = {}
        self.get_logger().info('LLM Task Decomposition Node initialized')

    def goal_callback(self, msg):
        """Process high-level goal and decompose into tasks"""
        try:
            goal_description = msg.data
            self.get_logger().info(f'Decomposing goal: {goal_description}')

            # Decompose the task using LLM
            subtasks = self.decomposer.decompose_task(goal_description)

            # Create task plan message
            plan_msg = TaskPlan()
            plan_msg.goal_description = goal_description
            plan_msg.timestamp = self.get_clock().now().to_msg()

            # Convert subtasks to ROS 2 messages
            for subtask_data in subtasks:
                task_msg = Task()
                task_msg.id = subtask_data.get('id', 0)
                task_msg.name = subtask_data.get('name', 'unknown')
                task_msg.description = subtask_data.get('description', '')
                task_msg.task_type = subtask_data.get('type', 'unknown')
                task_msg.dependencies = subtask_data.get('dependencies', [])

                # Add parameters
                for param_name, param_value in subtask_data.get('parameters', {}).items():
                    param = Task.Parameter()
                    param.name = param_name
                    param.value = str(param_value)
                    task_msg.parameters.append(param)

                plan_msg.tasks.append(task_msg)

            # Publish the task plan
            self.plan_pub.publish(plan_msg)
            self.get_logger().info(f'Published task plan with {len(subtasks)} subtasks')

        except Exception as e:
            self.get_logger().error(f'Error in task decomposition: {e}')
            self.publish_error_feedback(f'Task decomposition failed: {e}')

    def state_callback(self, msg):
        """Update current robot state"""
        self.current_state = self.parse_state(msg.data)

    def publish_error_feedback(self, error_message):
        """Publish error feedback"""
        feedback_msg = String()
        feedback_msg.data = error_message
        # Assuming there's a feedback topic
        # self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMTaskDecompositionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM task decomposition node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Decomposition Techniques

### Context-Aware Decomposition

Consider the robot's current state and environment:

```python
class ContextAwareDecomposer(LLMTaskDecomposer):
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.robot_state = {}
        self.environment_map = {}
        self.object_locations = {}

    def decompose_with_context(self, goal_description, context):
        """Decompose task considering current context"""
        # Update context
        self.robot_state = context.get('robot_state', {})
        self.environment_map = context.get('environment_map', {})
        self.object_locations = context.get('object_locations', {})

        # Create context-aware prompt
        context_prompt = f"""
        You are a task decomposition expert for humanoid robotics. Consider the following context:

        Current Robot State:
        {self.robot_state}

        Environment Map:
        {self.environment_map}

        Known Object Locations:
        {self.object_locations}

        Given this context, decompose the following goal:

        Goal: {goal_description}

        Take into account the robot's current location, available objects, and environmental constraints.
        """

        response = self.llm_client.generate(context_prompt)
        return self.parse_response(response)
```

### Hierarchical Task Decomposition

Break down tasks into multiple levels of abstraction:

```python
class HierarchicalDecomposer:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def decompose_hierarchically(self, goal_description):
        """Decompose task in a hierarchical manner"""
        # First, decompose into high-level phases
        high_level_prompt = f"""
        Decompose the goal into high-level phases:

        Goal: {goal_description}

        Provide phases like: Planning, Navigation, Manipulation, etc.
        """

        high_level_response = self.llm_client.generate(high_level_prompt)
        phases = self.parse_phases(high_level_response)

        # Then decompose each phase into subtasks
        full_decomposition = []
        for phase in phases:
            phase_decomposition = self.decompose_phase(phase, goal_description)
            full_decomposition.extend(phase_decomposition)

        return full_decomposition

    def decompose_phase(self, phase, overall_goal):
        """Decompose a specific phase into subtasks"""
        phase_prompt = f"""
        For the phase "{phase}" of the overall goal "{overall_goal}",
        decompose into specific, executable subtasks.
        """

        response = self.llm_client.generate(phase_prompt)
        return self.parse_response(response)
```

## Quality and Validation

### Task Plan Validation

Validate the decomposed tasks before execution:

```python
class TaskPlanValidator:
    def __init__(self):
        self.known_capabilities = {
            'navigation': ['move_to_location', 'follow_path'],
            'manipulation': ['grasp_object', 'release_object', 'move_arm'],
            'perception': ['detect_object', 'localize_object'],
            'system': ['wait', 'report_status']
        }

    def validate_plan(self, task_plan):
        """Validate task plan for feasibility"""
        issues = []

        # Check if all task types are supported
        for task in task_plan.tasks:
            if task.task_type not in self.known_capabilities:
                issues.append(f"Unknown task type: {task.task_type}")

        # Check dependencies
        task_ids = [task.id for task in task_plan.tasks]
        for task in task_plan.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    issues.append(f"Invalid dependency: {dep_id} for task {task.id}")

        # Check for circular dependencies
        if self.has_circular_dependencies(task_plan):
            issues.append("Circular dependencies detected")

        return len(issues) == 0, issues

    def has_circular_dependencies(self, task_plan):
        """Check for circular dependencies in task plan"""
        # Implementation of cycle detection algorithm
        pass
```

### Confidence Scoring

Assess the quality of LLM-generated decompositions:

```python
def score_decomposition_quality(self, decomposition, original_goal):
    """Score the quality of task decomposition"""
    score = 0.0

    # Check if decomposition addresses the original goal
    if self.decomposition_addresses_goal(decomposition, original_goal):
        score += 0.3

    # Check completeness
    if self.is_decomposition_complete(decomposition, original_goal):
        score += 0.3

    # Check executability
    if self.are_tasks_executable(decomposition):
        score += 0.2

    # Check logical ordering
    if self.has_logical_ordering(decomposition):
        score += 0.2

    return score

def decomposition_addresses_goal(self, decomposition, goal):
    """Check if decomposition addresses the original goal"""
    # Implementation to verify goal alignment
    pass
```

## Performance Considerations

### Caching Common Decompositions

Cache frequently requested task decompositions:

```python
from functools import lru_cache

class CachedDecomposer(LLMTaskDecomposer):
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.cache_size = 100

    @lru_cache(maxsize=100)
    def cached_decompose_task(self, goal_description):
        """Cached version of task decomposition"""
        return self.decompose_task(goal_description)
```

### Parallel Processing

For complex goals, process multiple aspects in parallel:

```python
import concurrent.futures

class ParallelDecomposer:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def decompose_parallel_components(self, goal_description):
        """Decompose parallelizable components of a goal"""
        # Identify parallelizable aspects
        aspects = self.identify_aspects(goal_description)

        # Process aspects in parallel
        futures = []
        for aspect in aspects:
            future = self.executor.submit(self.decompose_aspect, aspect)
            futures.append(future)

        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

        return self.merge_results(results)
```

## Error Handling and Fallbacks

### Handling LLM Limitations

Implement fallback strategies when LLM fails:

```python
class RobustDecomposer:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.rule_based_fallback = RuleBasedDecomposer()

    def decompose_task_robust(self, goal_description):
        """Decompose task with fallback mechanisms"""
        try:
            # Try LLM decomposition
            result = self.decompose_task(goal_description)

            # Validate result
            if self.is_valid_decomposition(result):
                return result
            else:
                self.get_logger().warn("LLM decomposition failed validation, using fallback")
        except Exception as e:
            self.get_logger().warn(f"LLM decomposition failed: {e}, using fallback")

        # Use fallback method
        return self.rule_based_fallback.decompose_task(goal_description)
```

## Testing Task Decomposition

### Unit Tests

```python
import unittest

class TestLLMTaskDecomposition(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MockLLMClient()
        self.decomposer = LLMTaskDecomposer(self.mock_llm_client)

    def test_simple_navigation_decomposition(self):
        """Test decomposition of simple navigation goal"""
        goal = "Go to the kitchen"
        decomposition = self.decomposer.decompose_task(goal)

        self.assertEqual(len(decomposition), 1)
        self.assertEqual(decomposition[0]['type'], 'navigation')

    def test_complex_manipulation_decomposition(self):
        """Test decomposition of complex manipulation goal"""
        goal = "Pick up the red ball and place it on the table"
        decomposition = self.decomposer.decompose_task(goal)

        # Should have navigation, perception, and manipulation tasks
        task_types = [task['type'] for task in decomposition]
        self.assertIn('navigation', task_types)
        self.assertIn('perception', task_types)
        self.assertIn('manipulation', task_types)
```

Task decomposition with LLMs enables humanoid robots to understand and execute complex, high-level goals expressed in natural language. By combining LLM capabilities with ROS 2 integration and safety validation, we can create sophisticated planning systems that adapt to various scenarios and environments.