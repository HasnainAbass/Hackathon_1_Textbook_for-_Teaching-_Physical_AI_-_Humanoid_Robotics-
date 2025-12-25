# Acceptance Scenario: Complex Goal Decomposition ("Navigate to kitchen, identify cup, pick up and return")

## Scenario Overview

This document describes the acceptance scenario for processing the complex language goal "Navigate to kitchen, identify a cup, pick it up, and return to current location" in a humanoid robotics simulation environment. This scenario tests the complete LLM-based cognitive planning pipeline from goal understanding to task execution.

## Pre-conditions

- ROS 2 environment is running with required nodes
- LLM planning pipeline is initialized and active
- Simulation environment is loaded with humanoid robot, kitchen area, and objects
- Robot is positioned in starting location (living room)
- Cup object is placed in kitchen area
- All required dependencies (LLM client, ROS 2 interfaces) are loaded
- Safety systems and constraint validators are operational
- Human-in-the-loop interface is available for approval if needed

## Test Environment Setup

### Required Nodes
- `llm_planning_node` - Main LLM-based planning node
- `simulated_robot` - Robot simulation in Gazebo
- `object_detector` - Object detection and recognition
- `manipulation_controller` - Robot arm and gripper control
- `navigation_controller` - Robot navigation system
- `constraint_validator_node` - Constraint validation system
- `human_interface_node` - Human-in-the-loop interface

### Required Topics
- `/high_level_goals` - Input for complex language goals
- `/generated_task_plans` - Output for decomposed task plans
- `/constraint_violations` - Output for constraint violations
- `/planning_feedback` - Output for planning feedback
- `/robot_state` - Robot state information
- `/execution_status` - Task execution status

## Test Steps

### Step 1: Initialize Test Environment
1. Launch the simulation environment with humanoid robot, kitchen area, and objects
2. Start all required ROS 2 nodes including LLM planning system
3. Verify all nodes are communicating properly
4. Confirm robot is at starting position (e.g., living room)
5. Verify cup object is placed in kitchen area
6. Confirm LLM client is properly configured and accessible

### Step 2: Send Complex Language Goal
1. Publish the complex language goal: "Navigate to kitchen, identify a cup, pick it up, and return to current location"
   ```bash
   ros2 topic pub /high_level_goals std_msgs/String "data: 'Navigate to kitchen, identify a cup, pick it up, and return to current location'"
   ```

### Step 3: Verify Goal Understanding and Decomposition
1. Monitor `/planning_feedback` topic
2. Verify LLM processes the goal: "Processing goal: Navigate to kitchen, identify a cup, pick it up, and return to current location"
3. Check for decomposition feedback: "Decomposing goal: Navigate to kitchen, identify a cup, pick it up, and return to current location"

### Step 4: Verify Task Decomposition
1. Monitor `/generated_task_plans` topic
2. Verify plan contains appropriate sequence of actions:
   - Navigation to kitchen
   - Object detection (cup identification)
   - Manipulation (grasping the cup)
   - Navigation back to starting location
3. Verify action dependencies are properly set
4. Check action parameters are correctly populated

### Step 5: Verify Constraint Validation
1. Monitor constraint validation process
2. Verify plan passes safety constraints (collision avoidance, human safety, etc.)
3. Check environmental constraints (navigable paths, reachable objects)
4. Confirm capability constraints (robot can grasp cup, lift appropriate weight)

### Step 6: Verify Human-in-the-Loop Process (if required)
1. Check if human approval is needed for this plan
2. If approval required, verify request is sent to human operator
3. If approval needed, provide approval to continue
4. Monitor human interface for status updates

### Step 7: Verify Plan Execution
1. Monitor robot navigation to kitchen area
2. Verify object detection and identification of cup
3. Confirm robot grasping of cup
4. Monitor robot navigation back to starting location
5. Verify cup delivery to original location

### Step 8: Verify Execution Feedback
1. Monitor `/planning_feedback` for execution updates
2. Verify feedback messages throughout execution:
   - "Navigating to kitchen"
   - "Identifying cup in kitchen"
   - "Grasping cup"
   - "Returning to starting location"
   - "Task completed successfully"

## Expected Results

### Primary Success Scenarios
1. **Goal Understanding**: Complex goal is correctly understood by LLM
2. **Task Decomposition**: Goal decomposes into 4-6 specific subtasks in correct sequence
3. **Constraint Validation**: Plan passes all safety and operational constraints
4. **Plan Execution**: Robot successfully completes all subtasks
5. **Object Manipulation**: Cup is successfully identified and grasped
6. **Navigation**: Robot navigates to kitchen and back safely
7. **Feedback**: User receives appropriate feedback throughout process

### Success Criteria
- **Recognition Accuracy**: Goal is correctly understood with > 90% confidence
- **Decomposition Quality**: Plan contains 4-6 logical, executable subtasks
- **Constraint Compliance**: All safety and operational constraints are respected
- **Execution Success**: All subtasks complete successfully > 80% of attempts
- **Execution Time**: Complete operation completes within 120 seconds
- **Safety**: Robot operates safely without collisions or violations
- **Feedback**: Clear feedback provided to user at each step

## Alternative Scenarios

### Scenario A: No Cup Available in Kitchen
1. If no cup is detected in kitchen, system should inform user
2. Feedback should be: "No cup detected in kitchen area"
3. Robot should return to starting position or ask for clarification
4. Plan should adapt to alternative available objects if specified

### Scenario B: Path to Kitchen Blocked
1. If path to kitchen is blocked, system should find alternative route
2. If no alternative route available, system should inform user
3. Feedback should indicate path issues and proposed solutions

### Scenario C: Cup Too Heavy or Fragile
1. If cup exceeds weight or fragility constraints, system should not attempt to grasp
2. Feedback should be: "Object cannot be safely grasped"
3. Robot should return without object or seek alternative

## Error Conditions

### Error Condition 1: LLM Processing Failure
- **Trigger**: LLM service unavailable or processing error
- **Response**: "LLM processing failed, using fallback planning"
- **Action**: System attempts rule-based decomposition

### Error Condition 2: Constraint Violation
- **Trigger**: Generated plan violates safety constraints
- **Response**: "Plan violates safety constraints, generating alternative"
- **Action**: System generates constraint-compliant alternative plan

### Error Condition 3: Navigation Failure
- **Trigger**: Robot unable to navigate to kitchen
- **Response**: "Unable to navigate to kitchen, path blocked"
- **Action**: Robot stops and requests human assistance

### Error Condition 4: Grasping Failure
- **Trigger**: Robot unable to successfully grasp the cup
- **Response**: "Unable to grasp object safely"
- **Action**: Robot returns to neutral position

## Validation Steps

### Automated Validation
```python
def validate_complex_goal_scenario():
    """Validate the complex goal decomposition scenario"""

    # Check initial conditions
    robot_start_position = get_robot_position()
    cup_present = check_object_in_kitchen("cup")
    assert cup_present, "Cup not present in kitchen"

    # Send complex goal
    send_language_goal("Navigate to kitchen, identify a cup, pick it up, and return to current location")

    # Wait for task plan generation
    plan = wait_for_task_plan(timeout=30)
    assert plan is not None, "No task plan generated"

    # Verify plan structure
    assert len(plan.actions) >= 4, f"Expected at least 4 actions, got {len(plan.actions)}"

    # Check for navigation action
    nav_actions = [a for a in plan.actions if 'navigate' in a.action_type.lower()]
    assert len(nav_actions) >= 2, "Expected at least 2 navigation actions (to kitchen and back)"

    # Check for manipulation action
    manipulation_actions = [a for a in plan.actions if 'grasp' in a.action_type.lower() or 'manipulation' in a.action_type.lower()]
    assert len(manipulation_actions) >= 1, "Expected at least 1 manipulation action"

    # Wait for execution completion
    execution_result = wait_for_execution_completion(timeout=120)

    # Verify final state
    final_position = get_robot_position()
    final_object_state = get_robot_object_state()

    # Validate success criteria
    assert final_position == robot_start_position, "Robot did not return to starting location"
    assert final_object_state.has_object, "Robot did not successfully grasp object"

    # Check feedback messages
    feedback_history = get_planning_feedback_history()
    expected_messages = ["Navigating to kitchen", "Identifying cup", "Grasping cup", "Returning to starting location"]
    for msg in expected_messages:
        assert any(msg in f for f in feedback_history), f"Missing feedback: {msg}"

    return True
```

### Manual Validation
1. **Visual Confirmation**: Observe robot navigation and manipulation in simulation
2. **Object Verification**: Confirm cup is grasped and returned to starting location
3. **Feedback Verification**: Check appropriate user feedback at each step
4. **Safety Check**: Verify robot operates safely throughout execution
5. **Timing Verification**: Confirm execution completes within expected time

## Performance Metrics

### Response Time
- **Goal Processing**: < 10 seconds from goal receipt to plan generation
- **Maximum**: < 30 seconds for complete processing

### Accuracy
- **Goal Understanding**: > 90% accuracy for clear complex goals
- **Task Decomposition**: > 85% of decomposed tasks are executable
- **Constraint Validation**: 100% of generated plans pass safety validation

### Resource Usage
- **CPU**: < 50% during LLM processing (may be higher during processing)
- **Memory**: Stable memory usage without leaks
- **Network**: Minimal message overhead for ROS 2 communication

## Test Data

### Input Data
- Complex goal: "Navigate to kitchen, identify a cup, pick it up, and return to current location"
- Expected subtasks: Navigation → Object detection → Manipulation → Navigation
- Expected objects: Cup in kitchen area

### Expected Output Data
- Task plan: 4-6 actions in sequence
- Navigation: To kitchen and back to start
- Manipulation: Grasp cup
- Execution confirmation: Task completed successfully

## Dependencies

### Software Dependencies
- ROS 2 Humble Hawksbill or later
- OpenAI API access (or equivalent LLM service)
- Gazebo simulation environment with manipulation support
- Object detection and recognition packages
- Navigation and manipulation control packages

### Hardware Dependencies (Simulation)
- CPU: Multi-core processor (LLM processing is compute-intensive)
- RAM: 16GB minimum, 32GB recommended for LLM operations
- GPU: For visualization and perception processing

## Environment Variables

### Required Environment Variables
```bash
export ROS_DOMAIN_ID=0
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/models
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO
export OPENAI_API_KEY=your_api_key_here
export MOVEIT_ROBOT_DESCRIPTION=humanoid_robot
```

## Rollback Plan

If the test fails:
1. Stop all ROS 2 nodes
2. Reset simulation environment to initial state
3. Clear any persistent data or cache
4. Release any grasped objects in simulation
5. Return robot to neutral position
6. Restart nodes and repeat test
7. If persistent failure, check LLM API access and configuration

## Success Metrics Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Goal Understanding Confidence | > 0.9 | TBD | TBD |
| Task Decomposition Quality | 4-6 actions | TBD | TBD |
| Constraint Validation Pass Rate | 100% | TBD | TBD |
| Execution Success Rate | > 0.8 | TBD | TBD |
| Response Time | < 10s | TBD | TBD |
| User Feedback | Clear & Timely | TBD | TBD |

## Post-conditions

- Robot has successfully completed the complex task
- Cup has been grasped and returned to starting location
- All ROS 2 nodes remain active and responsive
- System is ready for next command
- Feedback indicates successful completion
- No error states or safety violations occurred
- Robot maintains safe operational state
- Task plan is properly logged for future reference