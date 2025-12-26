# Acceptance Scenario: "Move forward 2 meters" Command Processing

## Scenario Overview

This document describes the acceptance scenario for processing the voice command "Move forward 2 meters" in a humanoid robotics simulation environment. This scenario tests the complete voice-to-action pipeline from speech recognition to robot execution.

## Pre-conditions

- ROS 2 environment is running with required nodes
- Voice processing pipeline is initialized and active
- Simulation environment is loaded with humanoid robot
- Robot is positioned at known starting location
- Voice command simulator is available for testing
- All required dependencies (Whisper model, ROS 2 interfaces) are loaded

## Test Environment Setup

### Required Nodes
- `voice_command_processor` - Main voice processing node
- `simulated_robot` - Robot simulation in Gazebo
- `robot_state_publisher` - Publishes robot state
- `joint_state_publisher` - Publishes joint states
- `voice_feedback_publisher` - Provides feedback to user

### Required Topics
- `/simulated_voice_input` - Input for voice commands
- `/cmd_vel` - Output for velocity commands
- `/voice_feedback` - Output for user feedback
- `/extracted_intent` - Output for extracted intents
- `/robot/position` - Robot position feedback

## Test Steps

### Step 1: Initialize Test Environment
1. Launch the simulation environment with humanoid robot
2. Start all required ROS 2 nodes
3. Verify all nodes are communicating properly
4. Confirm robot is at starting position (0, 0, 0)

### Step 2: Send Voice Command
1. Publish the voice command: "Move forward 2 meters"
   ```bash
   ros2 topic pub /simulated_voice_input std_msgs/String "data: 'Move forward 2 meters'"
   ```

### Step 3: Verify Speech-to-Text Processing
1. Monitor `/transcribed_text` topic (if available)
2. Verify transcription accuracy: "move forward 2 meters"
3. Check transcription confidence is > 0.8

### Step 4: Verify Intent Extraction
1. Monitor `/extracted_intent` topic
2. Verify intent: `navigation`
3. Verify action: `move_forward`
4. Verify parameters: `distance=2`, `direction=forward`
5. Verify confidence: > 0.7

### Step 5: Verify Robot Action Execution
1. Monitor `/cmd_vel` topic
2. Verify linear velocity: `linear.x = 0.5` m/s (or similar forward velocity)
3. Verify no angular velocity: `angular.z = 0.0`
4. Monitor robot position changes over time

### Step 6: Verify Movement Execution
1. Track robot position from `/robot/position` or TF
2. Verify robot moves approximately 2 meters forward
3. Verify movement takes expected time (2m / 0.5m/s = 4 seconds)
4. Verify final position is (0, 2, 0) or similar forward displacement

### Step 7: Verify Feedback
1. Monitor `/voice_feedback` topic
2. Verify feedback message: "Moving forward 2 meters"
3. Verify success confirmation after completion

## Expected Results

### Primary Success Scenarios
1. **Voice Recognition**: Command "Move forward 2 meters" is correctly transcribed
2. **Intent Extraction**: Navigation intent with forward movement is correctly identified
3. **Action Mapping**: Appropriate velocity commands are published to `/cmd_vel`
4. **Robot Movement**: Robot moves forward approximately 2 meters
5. **Feedback**: User receives appropriate feedback throughout the process

### Success Criteria
- **Recognition Accuracy**: Voice command is recognized with > 85% confidence
- **Movement Accuracy**: Robot moves within 10% of specified distance (1.8-2.2m)
- **Execution Time**: Movement completes within expected time frame
- **Safety**: Robot stops safely after completing movement
- **Feedback**: Clear feedback provided to user

## Alternative Scenarios

### Scenario A: Low Confidence Recognition
1. If transcription confidence < 0.7, system should request clarification
2. Feedback should indicate uncertainty
3. System should not execute potentially incorrect command

### Scenario B: Invalid Distance
1. If distance is invalid (negative, extremely large), system should reject command
2. Appropriate error feedback should be provided
3. Robot should not move

### Scenario C: Obstacle Detection
1. If robot detects obstacle in path, movement should be stopped
2. System should provide feedback about obstacle
3. Alternative navigation should be attempted if possible

## Error Conditions

### Error Condition 1: Unrecognized Command
- **Trigger**: Command not matching any known patterns
- **Response**: "I didn't understand that command"
- **Action**: No robot movement

### Error Condition 2: Robot Movement Failure
- **Trigger**: Robot unable to execute movement
- **Response**: "Unable to move forward, please check robot status"
- **Action**: Robot remains stationary

### Error Condition 3: Timeout
- **Trigger**: Command takes longer than expected to complete
- **Response**: "Command timed out, movement stopped"
- **Action**: Robot stops current movement

## Validation Steps

### Automated Validation
```python
def validate_move_forward_scenario():
    """Validate the 'Move forward 2 meters' scenario"""

    # Check initial conditions
    initial_position = get_robot_position()
    assert initial_position.z == 0  # Robot is on ground

    # Send command
    send_voice_command("Move forward 2 meters")

    # Wait for completion
    time.sleep(5)  # Allow time for movement

    # Check final position
    final_position = get_robot_position()
    distance_moved = abs(final_position.y - initial_position.y)

    # Validate movement distance (within 10% tolerance)
    assert 1.8 <= distance_moved <= 2.2, f"Expected 1.8-2.2m, got {distance_moved}m"

    # Check feedback
    feedback = get_latest_feedback()
    assert "Moving forward 2 meters" in feedback

    return True
```

### Manual Validation
1. **Visual Confirmation**: Observe robot movement in simulation
2. **Position Verification**: Check robot coordinates before and after
3. **Feedback Verification**: Confirm appropriate user feedback
4. **Safety Check**: Verify robot stops safely after movement

## Performance Metrics

### Response Time
- **Target**: < 2 seconds from command to start of movement
- **Maximum**: < 5 seconds for complete processing

### Accuracy
- **Recognition**: > 90% accuracy for clear commands
- **Movement**: Within 10% of specified distance
- **Direction**: Within 5 degrees of intended direction

### Resource Usage
- **CPU**: < 30% during processing
- **Memory**: Stable memory usage without leaks
- **Network**: Minimal message overhead

## Test Data

### Input Data
- Voice command: "Move forward 2 meters"
- Expected distance: 2.0 meters
- Expected direction: Forward (positive Y in robot frame)

### Expected Output Data
- Intent: `navigation`, `move_forward`
- Velocity command: `linear.x = 0.5`, `angular.z = 0.0`
- Movement duration: ~4 seconds
- Final displacement: ~2.0 meters forward

## Dependencies

### Software Dependencies
- ROS 2 Humble Hawksbill or later
- OpenAI Whisper model (small or medium)
- Gazebo simulation environment
- Required ROS 2 packages for navigation

### Hardware Dependencies (Simulation)
- CPU: Multi-core processor
- RAM: 8GB minimum, 16GB recommended
- GPU: For visualization (not required for processing)

## Environment Variables

### Required Environment Variables
```bash
export ROS_DOMAIN_ID=0
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/models
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO
```

## Rollback Plan

If the test fails:
1. Stop all ROS 2 nodes
2. Reset simulation environment to initial state
3. Clear any persistent data or cache
4. Restart nodes and repeat test

## Success Metrics Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Recognition Confidence | > 0.7 | TBD | TBD |
| Movement Distance | 2.0m ± 0.2m | TBD | TBD |
| Response Time | < 2s | TBD | TBD |
| User Feedback | Clear & Timely | TBD | TBD |

## Post-conditions

- Robot has moved forward approximately 2 meters from starting position
- All ROS 2 nodes remain active and responsive
- System is ready for next command
- Feedback indicates successful completion
- No error states or safety violations occurred