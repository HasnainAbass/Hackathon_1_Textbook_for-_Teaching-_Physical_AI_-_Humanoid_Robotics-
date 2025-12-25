# Troubleshooting Common Voice Processing Issues

## Introduction

This document provides guidance for identifying, diagnosing, and resolving common issues that may arise in voice-to-action processing systems. The troubleshooting approach follows a systematic methodology to ensure efficient problem resolution while maintaining system safety and reliability.

## General Troubleshooting Methodology

### 1. Problem Identification
- Clearly define the issue with specific symptoms
- Identify when the problem occurs (always, sometimes, under specific conditions)
- Document any error messages or unexpected behavior

### 2. Information Gathering
- Check system logs and error messages
- Verify system status and node connectivity
- Review recent changes or updates

### 3. Isolation and Testing
- Test individual components separately
- Use simplified test cases
- Verify inputs and outputs at each stage

### 4. Resolution and Verification
- Apply appropriate fixes
- Test the resolution
- Verify system returns to normal operation

## Common Voice Processing Issues

### Issue 1: Poor Speech Recognition Accuracy

**Symptoms:**
- Commands are frequently misinterpreted
- Low confidence scores for transcriptions
- High error rate in voice command processing

**Possible Causes:**
- Poor audio quality or background noise
- Microphone positioning or hardware issues
- Inappropriate Whisper model for the environment
- Audio format incompatibility

**Troubleshooting Steps:**
1. **Check Audio Input Quality:**
   ```bash
   # Monitor audio input
   ros2 topic echo /audio_input --field data
   ```

2. **Verify Audio Format:**
   ```python
   # Check audio sample rate and format
   audio_msg = get_audio_message()
   print(f"Sample rate: {audio_msg.sample_rate}")
   print(f"Encoding: {audio_msg.encoding}")
   ```

3. **Test with Clean Audio:**
   - Use pre-recorded clean audio samples
   - Test with simple, clear commands
   - Verify baseline recognition accuracy

4. **Adjust Model Settings:**
   ```python
   # Try different Whisper model sizes
   model = whisper.load_model("medium")  # or "large" for better accuracy
   # Or adjust language-specific settings
   result = model.transcribe(audio, language="en", temperature=0.0)
   ```

**Solutions:**
- Use noise-cancelling microphones
- Implement audio preprocessing (noise reduction)
- Adjust Whisper model settings for environment
- Ensure proper audio format (16kHz, mono recommended)

### Issue 2: Intent Extraction Failures

**Symptoms:**
- Commands result in "unknown" or incorrect intents
- High number of unprocessed voice commands
- Confusion between similar commands

**Possible Causes:**
- Insufficient training data for intent classifier
- Ambiguous command structures
- Rule-based patterns not covering all cases
- Context-aware processing not configured

**Troubleshooting Steps:**
1. **Review Intent Patterns:**
   ```python
   # Check current patterns
   for intent_type, patterns in self.patterns.items():
       print(f"{intent_type}: {patterns}")
   ```

2. **Test with Known Commands:**
   ```python
   test_commands = [
       "Move forward 2 meters",
       "Turn left 90 degrees",
       "Pick up the red ball"
   ]

   for cmd in test_commands:
       result = self.extract_intent(cmd)
       print(f"Command: {cmd} -> Intent: {result}")
   ```

3. **Analyze Failed Commands:**
   - Review logs of failed intent extractions
   - Identify common patterns in failed commands
   - Look for edge cases not covered by current rules

**Solutions:**
- Expand rule-based patterns
- Implement machine learning-based intent classification
- Add context-aware processing
- Improve command standardization

### Issue 3: ROS 2 Communication Problems

**Symptoms:**
- Voice commands not reaching processing nodes
- No feedback from robot after commands
- Nodes not communicating properly

**Possible Causes:**
- ROS 2 network configuration issues
- Topic/service name mismatches
- QoS profile incompatibilities
- Node discovery problems

**Troubleshooting Steps:**
1. **Check ROS 2 Nodes:**
   ```bash
   ros2 node list
   ros2 node info /voice_command_processor
   ```

2. **Verify Topic Connections:**
   ```bash
   ros2 topic list
   ros2 topic info /simulated_voice_input
   ros2 topic info /voice_feedback
   ```

3. **Test Topic Communication:**
   ```bash
   # Test publishing to voice input
   ros2 topic pub /simulated_voice_input std_msgs/String "data: 'test command'"

   # Monitor feedback
   ros2 topic echo /voice_feedback
   ```

4. **Check QoS Settings:**
   ```python
   # Verify QoS compatibility
   qos_profile = QoSProfile(
       depth=10,
       reliability=ReliabilityPolicy.RELIABLE
   )
   ```

**Solutions:**
- Verify ROS_DOMAIN_ID settings
- Check network configuration for multi-machine setups
- Ensure QoS profiles are compatible between publishers/subscribers
- Use proper topic/service names

### Issue 4: Performance and Latency Issues

**Symptoms:**
- Delay between voice command and robot response
- System appears unresponsive
- High CPU or memory usage

**Possible Causes:**
- Large Whisper model causing processing delays
- Insufficient computational resources
- Inefficient code implementation
- Thread blocking issues

**Troubleshooting Steps:**
1. **Monitor System Resources:**
   ```bash
   # Check CPU and memory usage
   htop
   # Monitor ROS 2 node resource usage
   ros2 run top ros2_top
   ```

2. **Measure Processing Times:**
   ```python
   import time

   start_time = time.time()
   result = self.process_voice_command(command)
   processing_time = time.time() - start_time
   print(f"Processing time: {processing_time:.2f}s")
   ```

3. **Profile Node Performance:**
   ```bash
   ros2 run performance_test performance_tester
   ```

**Solutions:**
- Use smaller Whisper models (tiny or base) for real-time applications
- Implement threading for processing-intensive tasks
- Optimize code for performance
- Add processing queues to handle load

### Issue 5: Safety System False Positives

**Symptoms:**
- Safe commands being blocked incorrectly
- System being overly conservative
- User frustration with safety restrictions

**Possible Causes:**
- Overly restrictive safety parameters
- Environmental sensing issues
- Incorrect safety validation logic

**Troubleshooting Steps:**
1. **Review Safety Parameters:**
   ```python
   # Check current safety limits
   print(f"Max distance: {self.safety_limits['max_distance']}")
   print(f"Min human distance: {self.safety_limits['min_approach_distance']}")
   ```

2. **Test Safety Validation:**
   ```python
   # Test specific safety checks
   command = {'action': 'navigation', 'parameters': {'distance': '2'}}
   is_safe, reason = self.validate_command_safety(command)
   print(f"Command safe: {is_safe}, Reason: {reason}")
   ```

3. **Check Sensor Data:**
   ```bash
   # Monitor sensor data used for safety
   ros2 topic echo /scan
   ros2 topic echo /detected_objects
   ```

**Solutions:**
- Adjust safety parameters to appropriate levels
- Improve sensor accuracy and reliability
- Refine safety validation logic
- Add user override options for safe situations

## Diagnostic Tools and Commands

### ROS 2 Diagnostic Commands

```bash
# Check overall system health
ros2 doctor

# Monitor all topics
ros2 topic list
ros2 topic hz /simulated_voice_input  # Check message rate

# Monitor node health
ros2 lifecycle list  # If using lifecycle nodes
ros2 run rqt_graph rqt_graph  # Visualize node connections

# Check system performance
ros2 run top ros2_top
```

### Voice Processing Specific Diagnostics

```python
class VoiceDiagnostics:
    def __init__(self):
        self.diagnostics = {
            'stt_accuracy': 0.0,
            'intent_confidence': 0.0,
            'processing_time': 0.0,
            'command_success_rate': 0.0
        }

    def run_diagnostics(self):
        """Run comprehensive voice system diagnostics"""
        results = {}

        # Test STT accuracy
        results['stt_test'] = self.test_speech_recognition()

        # Test intent extraction
        results['intent_test'] = self.test_intent_extraction()

        # Test ROS 2 communication
        results['ros2_test'] = self.test_ros2_communication()

        # Test safety systems
        results['safety_test'] = self.test_safety_systems()

        return results

    def test_speech_recognition(self):
        """Test speech-to-text accuracy with known audio samples"""
        test_samples = [
            ("move forward", "move forward 2 meters"),
            ("turn left", "turn left 90 degrees"),
            ("stop", "stop the robot")
        ]

        correct = 0
        for expected, audio_text in test_samples:
            # Simulate STT process
            result = self.simulate_stt(audio_text)
            if expected in result.lower():
                correct += 1

        accuracy = correct / len(test_samples)
        return {'accuracy': accuracy, 'passed': accuracy >= 0.8}
```

## Log Analysis

### Common Log Patterns

**Normal Operation:**
```
[INFO] [timestamp]: Processing command: "move forward 2 meters"
[INFO] [timestamp]: Transcribed: move forward 2 meters
[INFO] [timestamp]: Intent extracted: navigation, confidence: 0.9
[INFO] [timestamp]: Command executed successfully
```

**Error Conditions:**
```
[ERROR] [timestamp]: STT processing failed: [error details]
[WARN] [timestamp]: Low confidence intent: 0.4
[ERROR] [timestamp]: Safety validation failed: obstacle detected
```

### Log Monitoring Commands

```bash
# Monitor voice processing logs
ros2 run rcl_logging_spdlog view_logs | grep -i "voice\|stt\|intent"

# Monitor specific node logs
ros2 run rcl_logging_spdlog view_logs | grep "voice_command_processor"

# Monitor safety-related logs
ros2 run rcl_logging_spdlog view_logs | grep -i "safety\|emergency\|blocked"
```

## Recovery Procedures

### System Recovery Steps

1. **Immediate Response:**
   - Activate emergency stop if safety is compromised
   - Document the issue and system state
   - Check for any physical damage

2. **System Reset:**
   ```bash
   # Stop all ROS 2 nodes
   ros2 lifecycle set /voice_command_processor deactivate  # If using lifecycle nodes

   # Or kill nodes if needed
   ros2 node kill /voice_command_processor
   ```

3. **Component Restart:**
   ```bash
   # Restart individual components
   ros2 run vla_examples voice_command_processor
   ros2 run vla_examples voice_command_simulator
   ```

4. **System Verification:**
   - Test basic functionality with simple commands
   - Verify all safety systems are operational
   - Run diagnostic tests

## Prevention Strategies

### Regular Maintenance

1. **System Health Checks:**
   - Schedule regular diagnostic tests
   - Monitor system performance metrics
   - Update safety parameters based on usage patterns

2. **Environmental Monitoring:**
   - Regularly check microphone and sensor placement
   - Verify audio quality in operational environment
   - Monitor background noise levels

3. **Software Updates:**
   - Keep ROS 2 and dependencies updated
   - Regularly update Whisper models if needed
   - Apply security patches promptly

### Best Practices

1. **Gradual Rollout:**
   - Test new features in simulation first
   - Use staging environment before production
   - Monitor system behavior after updates

2. **Comprehensive Testing:**
   - Test edge cases and error conditions
   - Validate safety systems regularly
   - Perform load testing under expected usage

3. **Documentation:**
   - Maintain up-to-date troubleshooting guides
   - Document known issues and workarounds
   - Keep system architecture documentation current

## Support Resources

### When to Escalate

Contact advanced support for:
- Persistent system failures
- Safety system malfunctions
- Complex integration issues
- Performance problems requiring architecture changes

### Useful Commands Summary

```bash
# Basic system check
ros2 node list
ros2 topic list
ros2 service list

# Voice system specific
ros2 topic echo /simulated_voice_input
ros2 topic echo /voice_feedback
ros2 topic echo /extracted_intent

# Performance monitoring
ros2 run top ros2_top
ros2 doctor

# Log analysis
ros2 run rcl_logging_spdlog view_logs
```

## Conclusion

Effective troubleshooting of voice-to-action systems requires a systematic approach that addresses both the technical components and the integration between them. Regular monitoring, proactive maintenance, and comprehensive testing help prevent issues before they impact system operation. Always prioritize safety in troubleshooting and recovery procedures.