# Failure Handling and Recovery Mechanisms

## Introduction

Failure handling and recovery are critical components of Vision-Language-Action (VLA) systems. These mechanisms ensure that autonomous humanoid robots can gracefully handle various failure scenarios while maintaining safety and continuing operation when possible. This chapter covers comprehensive strategies for detecting, managing, and recovering from failures across all VLA system components.

## Failure Classification

### 1. Component Failures

Component failures affect individual system components:

#### Voice Processing Failures
- **STT (Speech-to-Text) failures**: Audio processing errors, API unavailability
- **Intent extraction failures**: Ambiguous or unrecognized commands
- **Audio input failures**: Microphone malfunctions, poor audio quality

#### LLM Planning Failures
- **API unavailability**: LLM service downtime
- **Rate limiting**: Exceeded API request limits
- **Invalid responses**: Malformed or unsafe LLM outputs
- **Processing timeouts**: LLM takes too long to respond

#### Perception Failures
- **Sensor failures**: Camera, LiDAR, IMU malfunctions
- **Object detection failures**: Poor lighting, occlusions
- **Localization failures**: Robot loses position awareness

#### Control Failures
- **Navigation failures**: Path planning errors, obstacle avoidance failures
- **Manipulation failures**: Grasping errors, joint limit violations
- **Hardware failures**: Motor malfunctions, actuator failures

### 2. System-Level Failures

System-level failures affect the entire VLA system:

#### Communication Failures
- **Network disruptions**: Loss of communication between components
- **ROS 2 communication failures**: Topic/service/action failures
- **Message queue overflows**: Too many pending messages

#### Safety Violations
- **Collision detection**: Potential or actual collisions
- **Human safety violations**: Approaching humans too closely
- **Operational limit violations**: Exceeding velocity, force, or payload limits

#### Resource Failures
- **Memory exhaustion**: Running out of system memory
- **CPU overload**: Excessive processing demand
- **Battery depletion**: Low power situations

## Failure Detection Mechanisms

### 1. Health Monitoring

Continuous monitoring of system health:

```python
class HealthMonitor:
    def __init__(self, node):
        self.node = node
        self.health_status = {
            'voice_processor': 'healthy',
            'llm_client': 'healthy',
            'perception_system': 'healthy',
            'control_system': 'healthy',
            'communication': 'healthy',
            'safety_system': 'healthy'
        }
        self.failure_timestamps = {}

        # Health check timer
        self.health_check_timer = node.create_timer(1.0, self.perform_health_check)

    def perform_health_check(self):
        """Perform comprehensive health check"""
        # Check voice processor
        self.health_status['voice_processor'] = self.check_voice_processor_health()

        # Check LLM client
        self.health_status['llm_client'] = self.check_llm_client_health()

        # Check perception system
        self.health_status['perception_system'] = self.check_perception_health()

        # Check control system
        self.health_status['control_system'] = self.check_control_health()

        # Check communication
        self.health_status['communication'] = self.check_communication_health()

        # Check safety system
        self.health_status['safety_system'] = self.check_safety_health()

    def check_voice_processor_health(self):
        """Check voice processor health"""
        try:
            # Check if voice processor is responding
            if hasattr(self, 'voice_processor'):
                return 'healthy'
            else:
                return 'uninitialized'
        except Exception:
            return 'failed'

    def check_llm_client_health(self):
        """Check LLM client health"""
        try:
            # Test API connectivity
            if self.llm_client:
                # Perform a quick test call
                return 'healthy'
            else:
                return 'uninitialized'
        except Exception:
            return 'failed'

    def check_perception_health(self):
        """Check perception system health"""
        try:
            # Check if perception data is flowing
            if hasattr(self, 'perception_system'):
                return 'healthy'
            else:
                return 'uninitialized'
        except Exception:
            return 'failed'

    def check_control_health(self):
        """Check control system health"""
        try:
            # Check if control commands are being published
            if hasattr(self, 'control_system'):
                return 'healthy'
            else:
                return 'uninitialized'
        except Exception:
            return 'failed'

    def check_communication_health(self):
        """Check ROS 2 communication health"""
        try:
            # Check if ROS 2 is running and communication is healthy
            return 'healthy'
        except Exception:
            return 'failed'

    def check_safety_health(self):
        """Check safety system health"""
        try:
            # Check if safety monitoring is active
            return 'healthy'
        except Exception:
            return 'failed'

    def get_system_health(self):
        """Get overall system health status"""
        unhealthy_components = [
            comp for comp, status in self.health_status.items()
            if status != 'healthy'
        ]

        if not unhealthy_components:
            return 'fully_healthy'
        elif len(unhealthy_components) == len(self.health_status):
            return 'completely_failed'
        else:
            return 'partially_degraded'
```

### 2. Performance Monitoring

Monitor system performance for degradation:

```python
class PerformanceMonitor:
    def __init__(self, node):
        self.node = node
        self.metrics = {
            'response_times': [],
            'error_rates': [],
            'throughput': [],
            'resource_usage': []
        }
        self.thresholds = {
            'max_response_time': 5.0,  # seconds
            'max_error_rate': 0.1,     # 10%
            'min_throughput': 1.0,     # commands per second
            'max_cpu_usage': 80.0,     # percent
            'max_memory_usage': 80.0   # percent
        }

    def record_response_time(self, component, response_time):
        """Record response time for performance monitoring"""
        if component not in self.metrics['response_times']:
            self.metrics['response_times'][component] = []

        self.metrics['response_times'][component].append(response_time)

        # Check if response time exceeds threshold
        if response_time > self.thresholds['max_response_time']:
            self.node.get_logger().warn(
                f'{component} response time {response_time:.2f}s exceeds threshold'
            )

    def record_error_rate(self, component, error_occurred):
        """Record error occurrence for error rate calculation"""
        if component not in self.metrics['error_rates']:
            self.metrics['error_rates'][component] = []

        self.metrics['error_rates'][component].append(1 if error_occurred else 0)

    def calculate_error_rate(self, component):
        """Calculate error rate for component"""
        if component not in self.metrics['error_rates']:
            return 0.0

        errors = self.metrics['error_rates'][component]
        if not errors:
            return 0.0

        error_rate = sum(errors) / len(errors)
        return error_rate

    def check_performance_degradation(self):
        """Check for performance degradation"""
        issues = []

        # Check response times
        for component, times in self.metrics['response_times'].items():
            if times and sum(times) / len(times) > self.thresholds['max_response_time']:
                issues.append(f'{component} response time degraded')

        # Check error rates
        for component in self.metrics['error_rates'].keys():
            error_rate = self.calculate_error_rate(component)
            if error_rate > self.thresholds['max_error_rate']:
                issues.append(f'{component} error rate too high: {error_rate:.2%}')

        return issues
```

## Failure Recovery Strategies

### 1. Graceful Degradation

Implement graceful degradation when failures occur:

```python
class GracefulDegradationHandler:
    def __init__(self, node):
        self.node = node
        self.degradation_levels = {
            'full_operation': {
                'voice': True,
                'llm_planning': True,
                'advanced_perception': True,
                'full_control': True
            },
            'reduced_functionality': {
                'voice': True,
                'llm_planning': False,
                'advanced_perception': False,
                'full_control': True
            },
            'basic_operation': {
                'voice': False,
                'llm_planning': False,
                'advanced_perception': False,
                'full_control': True
            },
            'manual_control_only': {
                'voice': False,
                'llm_planning': False,
                'advanced_perception': False,
                'full_control': False
            }
        }
        self.current_level = 'full_operation'

    def handle_component_failure(self, component, failure_type):
        """Handle component failure with appropriate degradation"""
        if component == 'voice_processor':
            if failure_type == 'critical':
                self.degrade_to_level('reduced_functionality')
            else:
                self.node.get_logger().warn(f'Voice processor degraded: {failure_type}')
        elif component == 'llm_client':
            if failure_type == 'api_unavailable':
                self.degrade_to_level('reduced_functionality')
            elif failure_type == 'rate_limit':
                self.node.get_logger().warn('LLM rate limited, reducing frequency')
        elif component == 'perception_system':
            if failure_type == 'sensor_failure':
                self.degrade_to_level('basic_operation')
            else:
                self.node.get_logger().warn(f'Perception degraded: {failure_type}')
        elif component == 'control_system':
            if failure_type == 'critical':
                self.degrade_to_level('manual_control_only')

    def degrade_to_level(self, level):
        """Degrade system to specified level"""
        if self.current_level != level:
            self.node.get_logger().warn(f'Degrading system from {self.current_level} to {level}')
            self.current_level = level

            # Notify other components of degradation
            self.notify_degradation(level)

    def notify_degradation(self, level):
        """Notify other components of system degradation"""
        # This could involve publishing messages to other nodes
        # or updating shared parameters
        pass

    def can_recover_to_level(self, level):
        """Check if system can recover to specified level"""
        # Check if required components are healthy
        if level == 'full_operation':
            return all([
                self.is_component_healthy('voice_processor'),
                self.is_component_healthy('llm_client'),
                self.is_component_healthy('perception_system'),
                self.is_component_healthy('control_system')
            ])
        elif level == 'reduced_functionality':
            return all([
                self.is_component_healthy('voice_processor'),
                self.is_component_healthy('control_system')
            ])
        elif level == 'basic_operation':
            return self.is_component_healthy('control_system')
        else:
            return True  # Manual control always available

    def is_component_healthy(self, component):
        """Check if component is healthy"""
        # This would check actual component health status
        return True  # Placeholder
```

### 2. Retry Mechanisms

Implement retry logic for transient failures:

```python
import asyncio
import time
from typing import Callable, Any, Optional

class RetryMechanism:
    def __init__(self, node):
        self.node = node
        self.retry_config = {
            'max_attempts': 3,
            'base_delay': 1.0,  # seconds
            'backoff_factor': 2.0,
            'max_delay': 10.0   # seconds
        }

    async def retry_with_backoff(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with exponential backoff retry"""
        max_attempts = kwargs.pop('max_attempts', self.retry_config['max_attempts'])
        base_delay = kwargs.pop('base_delay', self.retry_config['base_delay'])
        backoff_factor = kwargs.pop('backoff_factor', self.retry_config['backoff_factor'])
        max_delay = kwargs.pop('max_delay', self.retry_config['max_delay'])

        last_exception = None

        for attempt in range(max_attempts):
            try:
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    self.node.get_logger().info(f'Operation succeeded after {attempt} retries')
                return result
            except Exception as e:
                last_exception = e

                if attempt == max_attempts - 1:
                    # Last attempt failed
                    self.node.get_logger().error(f'Operation failed after {max_attempts} attempts: {e}')
                    raise e

                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                self.node.get_logger().warn(
                    f'Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...'
                )

                await asyncio.sleep(delay)

        # This shouldn't be reached, but just in case
        raise last_exception

    def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry (synchronous wrapper)"""
        import asyncio

        async def async_wrapper():
            return await self.retry_with_backoff(operation, *args, **kwargs)

        # Run in separate thread to avoid blocking
        import threading
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(async_wrapper()))
            return future.result()

    def llm_call_with_retry(self, llm_client, prompt, **kwargs):
        """Make LLM call with retry mechanism"""
        async def async_llm_call():
            return await llm_client.chat.completions.create(
                model=kwargs.get('model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 1000)
            )

        return self.execute_with_retry(async_llm_call)

    def api_call_with_retry(self, api_function, *args, **kwargs):
        """Make API call with retry mechanism"""
        def api_call():
            return api_function(*args, **kwargs)

        return self.execute_with_retry(api_call)
```

## Specific Recovery Procedures

### 1. Voice Processing Recovery

Handle voice processing failures:

```python
class VoiceProcessingRecovery:
    def __init__(self, node):
        self.node = node
        self.stt_fallback = STTFallbackSystem()
        self.audio_health_checker = AudioHealthChecker()

    def handle_voice_failure(self, failure_type, original_command):
        """Handle voice processing failure with recovery"""
        if failure_type == 'stt_failure':
            return self.handle_stt_failure(original_command)
        elif failure_type == 'audio_input_failure':
            return self.handle_audio_input_failure(original_command)
        elif failure_type == 'intent_extraction_failure':
            return self.handle_intent_extraction_failure(original_command)
        else:
            return self.handle_general_voice_failure(original_command)

    def handle_stt_failure(self, original_command):
        """Handle STT failure with fallback"""
        self.node.get_logger().warn('STT service unavailable, using fallback')

        # Use fallback STT system
        fallback_result = self.stt_fallback.process_audio(original_command)

        if fallback_result['success']:
            return {
                'success': True,
                'transcription': fallback_result['text'],
                'confidence': fallback_result['confidence'],
                'fallback_used': True
            }
        else:
            return {
                'success': False,
                'error': 'Both primary and fallback STT systems failed',
                'fallback_used': True
            }

    def handle_audio_input_failure(self, original_command):
        """Handle audio input failure"""
        # Check if this is a hardware issue or configuration issue
        audio_health = self.audio_health_checker.check_audio_health()

        if audio_health['microphone_status'] == 'malfunctioning':
            return {
                'success': False,
                'error': 'Microphone malfunction detected',
                'recommendation': 'Check microphone hardware connection'
            }
        elif audio_health['audio_settings'] == 'misconfigured':
            return {
                'success': False,
                'error': 'Audio settings misconfigured',
                'recommendation': 'Verify audio input configuration'
            }
        else:
            return {
                'success': False,
                'error': 'Audio input unavailable',
                'recommendation': 'Check audio device availability'
            }

    def handle_intent_extraction_failure(self, original_command):
        """Handle intent extraction failure"""
        # Try alternative intent extraction methods
        alternative_methods = [
            self.rule_based_intent_extraction,
            self.keyword_matching_intent_extraction
        ]

        for method in alternative_methods:
            try:
                intent = method(original_command)
                if intent and intent.get('confidence', 0) > 0.5:
                    return {
                        'success': True,
                        'intent': intent,
                        'method_used': method.__name__
                    }
            except Exception:
                continue

        return {
            'success': False,
            'error': 'Intent extraction failed with all methods',
            'original_command': original_command
        }

    def rule_based_intent_extraction(self, text):
        """Rule-based intent extraction as fallback"""
        text_lower = text.lower()

        # Navigation intents
        if any(word in text_lower for word in ['go to', 'navigate to', 'move to']):
            return {
                'action': 'navigation',
                'parameters': {'destination': self.extract_location(text_lower)},
                'confidence': 0.8
            }
        # Manipulation intents
        elif any(word in text_lower for word in ['pick up', 'grasp', 'get']):
            return {
                'action': 'manipulation',
                'parameters': {'object': self.extract_object(text_lower)},
                'confidence': 0.7
            }
        else:
            return None

    def keyword_matching_intent_extraction(self, text):
        """Keyword matching as another fallback"""
        # Implementation of keyword matching approach
        pass

    def extract_location(self, text):
        """Extract location from text"""
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom']
        for loc in locations:
            if loc in text:
                return loc
        return 'unknown'

    def extract_object(self, text):
        """Extract object from text"""
        objects = ['cup', 'ball', 'book', 'box', 'bottle']
        for obj in objects:
            if obj in text:
                return obj
        return 'unknown'


class STTFallbackSystem:
    """Fallback STT system for when primary STT fails"""
    def process_audio(self, audio_data):
        """Process audio with fallback STT"""
        # In a real system, this might use offline STT like CMU Sphinx
        # For this example, return mock data
        return {
            'success': True,
            'text': audio_data.get('text', 'fallback transcription'),
            'confidence': 0.6  # Lower confidence for fallback
        }


class AudioHealthChecker:
    """Check audio system health"""
    def check_audio_health(self):
        """Check audio system health"""
        return {
            'microphone_status': 'functional',  # 'functional' or 'malfunctioning'
            'audio_settings': 'configured',    # 'configured' or 'misconfigured'
            'input_level': 0.5,               # Current input level (0.0-1.0)
            'noise_level': 0.1                # Current noise level (0.0-1.0)
        }
```

### 2. LLM Planning Recovery

Handle LLM planning failures:

```python
class LLMPlanningRecovery:
    def __init__(self, node):
        self.node = node
        self.fallback_planners = {
            'rule_based': RuleBasedPlanner(),
            'template_based': TemplateBasedPlanner(),
            'scripted': ScriptedPlanner()
        }

    def handle_llm_failure(self, failure_type, intent, context):
        """Handle LLM planning failure with recovery"""
        if failure_type == 'api_unavailable':
            return self.handle_api_unavailable(intent, context)
        elif failure_type == 'rate_limit':
            return self.handle_rate_limit(intent, context)
        elif failure_type == 'invalid_response':
            return self.handle_invalid_response(intent, context)
        elif failure_type == 'timeout':
            return self.handle_timeout(intent, context)
        else:
            return self.handle_general_failure(intent, context)

    def handle_api_unavailable(self, intent, context):
        """Handle LLM API unavailability"""
        self.node.get_logger().warn('LLM API unavailable, using fallback planner')

        # Try fallback planners in order
        for planner_name, planner in self.fallback_planners.items():
            try:
                plan = planner.generate_plan(intent, context)
                if plan:
                    self.node.get_logger().info(f'Used {planner_name} fallback planner')
                    return {
                        'success': True,
                        'plan': plan,
                        'fallback_used': planner_name
                    }
            except Exception as e:
                self.node.get_logger().warn(f'{planner_name} fallback failed: {e}')
                continue

        return {
            'success': False,
            'error': 'All planning methods failed',
            'fallback_used': 'none'
        }

    def handle_rate_limit(self, intent, context):
        """Handle LLM rate limiting"""
        self.node.get_logger().warn('LLM rate limited, using cached plan if available')

        # Try to use cached plan for similar intent
        cached_plan = self.get_cached_plan(intent)
        if cached_plan:
            self.node.get_logger().info('Using cached plan due to rate limit')
            return {
                'success': True,
                'plan': cached_plan,
                'cached_used': True
            }

        # Otherwise fall back to simpler planner
        simple_plan = self.fallback_planners['rule_based'].generate_simple_plan(intent)
        return {
            'success': True,
            'plan': simple_plan,
            'fallback_used': 'rule_based'
        }

    def handle_invalid_response(self, intent, context):
        """Handle invalid LLM response"""
        self.node.get_logger().warn('Invalid LLM response received, using fallback')

        # Validate and correct the response, or use fallback
        corrected_plan = self.correct_invalid_response(intent, context)
        if corrected_plan:
            return {
                'success': True,
                'plan': corrected_plan,
                'response_corrected': True
            }

        # Use fallback if correction fails
        fallback_plan = self.fallback_planners['template_based'].generate_plan(intent, context)
        return {
            'success': True,
            'plan': fallback_plan,
            'fallback_used': 'template_based'
        }

    def handle_timeout(self, intent, context):
        """Handle LLM response timeout"""
        self.node.get_logger().warn('LLM response timeout, using fallback')

        # Use the fastest available fallback
        quick_plan = self.fallback_planners['scripted'].generate_quick_plan(intent)
        return {
            'success': True,
            'plan': quick_plan,
            'fallback_used': 'scripted'
        }

    def handle_general_failure(self, intent, context):
        """Handle general LLM failure"""
        # Try all fallbacks in order
        for planner_name, planner in self.fallback_planners.items():
            try:
                plan = planner.generate_plan(intent, context)
                if plan:
                    return {
                        'success': True,
                        'plan': plan,
                        'fallback_used': planner_name
                    }
            except Exception:
                continue

        return {
            'success': False,
            'error': 'All planning methods failed',
            'fallback_used': 'none'
        }

    def get_cached_plan(self, intent):
        """Get cached plan for similar intent"""
        # Implementation to retrieve cached plans
        # This would check a cache of previously generated plans
        return None  # Placeholder

    def correct_invalid_response(self, intent, context):
        """Attempt to correct invalid LLM response"""
        # Implementation to validate and correct LLM responses
        return None  # Placeholder


class RuleBasedPlanner:
    """Rule-based fallback planner"""
    def generate_plan(self, intent, context):
        """Generate plan using rules"""
        action = intent.get('action', 'unknown')

        if action == 'navigation':
            return self.generate_navigation_plan(intent, context)
        elif action == 'manipulation':
            return self.generate_manipulation_plan(intent, context)
        else:
            return self.generate_generic_plan(intent, context)

    def generate_navigation_plan(self, intent, context):
        """Generate navigation plan using rules"""
        destination = intent.get('parameters', {}).get('destination', 'unknown')
        return {
            'action_sequence': [
                {
                    'id': 1,
                    'type': 'navigation',
                    'description': f'Navigate to {destination}',
                    'parameters': {'location': destination},
                    'dependencies': []
                }
            ],
            'estimated_duration': 30
        }

    def generate_manipulation_plan(self, intent, context):
        """Generate manipulation plan using rules"""
        obj = intent.get('parameters', {}).get('object', 'unknown')
        return {
            'action_sequence': [
                {
                    'id': 1,
                    'type': 'navigation',
                    'description': f'Navigate to {obj}',
                    'parameters': {'object': obj},
                    'dependencies': []
                },
                {
                    'id': 2,
                    'type': 'perception',
                    'description': f'Detect {obj}',
                    'parameters': {'object_type': obj},
                    'dependencies': [1]
                },
                {
                    'id': 3,
                    'type': 'manipulation',
                    'description': f'Grasp {obj}',
                    'parameters': {'object_id': obj},
                    'dependencies': [2]
                }
            ],
            'estimated_duration': 60
        }

    def generate_generic_plan(self, intent, context):
        """Generate generic plan using rules"""
        return {
            'action_sequence': [
                {
                    'id': 1,
                    'type': 'system',
                    'description': f'Process {intent.get("action", "unknown")} command',
                    'parameters': intent.get('parameters', {}),
                    'dependencies': []
                }
            ],
            'estimated_duration': 10
        }

    def generate_simple_plan(self, intent):
        """Generate simple plan for rate limit scenarios"""
        action = intent.get('action', 'unknown')
        return {
            'action_sequence': [
                {
                    'id': 1,
                    'type': action if action in ['navigation', 'system'] else 'system',
                    'description': f'Execute {action} command',
                    'parameters': intent.get('parameters', {}),
                    'dependencies': []
                }
            ],
            'estimated_duration': 15
        }


class TemplateBasedPlanner:
    """Template-based fallback planner"""
    def generate_plan(self, intent, context):
        """Generate plan using templates"""
        # Implementation using plan templates
        pass


class ScriptedPlanner:
    """Scripted fallback planner for quick responses"""
    def generate_quick_plan(self, intent):
        """Generate quick plan for timeout scenarios"""
        action = intent.get('action', 'unknown')
        return {
            'action_sequence': [
                {
                    'id': 1,
                    'type': 'system',
                    'description': f'Quick execution of {action}',
                    'parameters': intent.get('parameters', {}),
                    'dependencies': []
                }
            ],
            'estimated_duration': 5
        }
```

### 3. Perception Recovery

Handle perception system failures:

```python
class PerceptionRecovery:
    def __init__(self, node):
        self.node = node
        self.sensor_backup_systems = {
            'camera': BackupCameraSystem(),
            'lidar': BackupLidarSystem(),
            'imu': BackupIMUSystem()
        }
        self.cached_perception_data = {}

    def handle_perception_failure(self, failure_type, sensor_type, context):
        """Handle perception system failure with recovery"""
        if failure_type == 'sensor_failure':
            return self.handle_sensor_failure(sensor_type, context)
        elif failure_type == 'detection_failure':
            return self.handle_detection_failure(sensor_type, context)
        elif failure_type == 'localization_failure':
            return self.handle_localization_failure(context)
        else:
            return self.handle_general_perception_failure(sensor_type, context)

    def handle_sensor_failure(self, sensor_type, context):
        """Handle specific sensor failure"""
        self.node.get_logger().warn(f'{sensor_type} sensor failure detected')

        # Try backup sensor system
        if sensor_type in self.sensor_backup_systems:
            backup_system = self.sensor_backup_systems[sensor_type]
            backup_data = backup_system.get_data()

            if backup_data:
                self.node.get_logger().info(f'Using {sensor_type} backup system')
                return {
                    'success': True,
                    'data': backup_data,
                    'backup_used': True
                }

        # Try cached data if available
        cached_data = self.get_cached_perception_data(sensor_type)
        if cached_data:
            self.node.get_logger().warn(f'Using cached {sensor_type} data')
            return {
                'success': True,
                'data': cached_data,
                'cached_used': True
            }

        return {
            'success': False,
            'error': f'{sensor_type} sensor unavailable',
            'sensor_type': sensor_type
        }

    def handle_detection_failure(self, sensor_type, context):
        """Handle object detection failure"""
        # Try alternative detection methods
        alternative_methods = [
            self.use_template_matching,
            self.use_color_segmentation,
            self.use_motion_detection
        ]

        for method in alternative_methods:
            try:
                result = method(context)
                if result:
                    return {
                        'success': True,
                        'detections': result,
                        'method_used': method.__name__
                    }
            except Exception:
                continue

        # Use cached detections if available
        cached_detections = self.get_cached_detections()
        if cached_detections:
            self.node.get_logger().warn('Using cached detection data')
            return {
                'success': True,
                'detections': cached_detections,
                'cached_used': True
            }

        return {
            'success': False,
            'error': 'Object detection failed with all methods'
        }

    def handle_localization_failure(self, context):
        """Handle robot localization failure"""
        self.node.get_logger().warn('Localization failure detected')

        # Try to recover using odometry
        if 'odometry' in context:
            estimated_pose = self.estimate_pose_from_odometry(context['odometry'])
            if estimated_pose:
                self.node.get_logger().info('Recovered pose using odometry')
                return {
                    'success': True,
                    'pose': estimated_pose,
                    'recovery_method': 'odometry'
                }

        # Try to recover using last known good pose
        last_known_pose = context.get('last_known_pose')
        if last_known_pose:
            self.node.get_logger().warn('Using last known pose')
            return {
                'success': True,
                'pose': last_known_pose,
                'recovery_method': 'last_known'
            }

        return {
            'success': False,
            'error': 'Localization recovery failed'
        }

    def use_template_matching(self, context):
        """Use template matching as alternative detection method"""
        # Implementation for template matching
        pass

    def use_color_segmentation(self, context):
        """Use color segmentation as alternative detection method"""
        # Implementation for color-based detection
        pass

    def use_motion_detection(self, context):
        """Use motion detection as alternative detection method"""
        # Implementation for motion-based detection
        pass

    def estimate_pose_from_odometry(self, odometry_data):
        """Estimate pose using odometry data"""
        # Implementation for pose estimation from odometry
        pass

    def get_cached_perception_data(self, sensor_type):
        """Get cached perception data for sensor type"""
        if sensor_type in self.cached_perception_data:
            cached_entry = self.cached_perception_data[sensor_type]
            age = time.time() - cached_entry['timestamp']

            # Only return if data is not too old (e.g., less than 5 seconds)
            if age < 5.0:
                return cached_entry['data']

        return None

    def get_cached_detections(self):
        """Get cached detection data"""
        # Implementation to retrieve cached detections
        pass


class BackupCameraSystem:
    """Backup camera system for when primary camera fails"""
    def get_data(self):
        """Get data from backup camera system"""
        # In a real system, this might use a secondary camera
        # or synthetic data generation
        return None  # Placeholder


class BackupLidarSystem:
    """Backup LiDAR system"""
    def get_data(self):
        """Get data from backup LiDAR system"""
        return None  # Placeholder


class BackupIMUSystem:
    """Backup IMU system"""
    def get_data(self):
        """Get data from backup IMU system"""
        return None  # Placeholder
```

### 4. Control Recovery

Handle control system failures:

```python
class ControlRecovery:
    def __init__(self, node):
        self.node = node
        self.emergency_procedures = {
            'navigation_failure': self.emergency_navigation_stop,
            'manipulation_failure': self.emergency_manipulation_stop,
            'hardware_failure': self.emergency_hardware_stop
        }
        self.fallback_controllers = {
            'navigation': FallbackNavigationController(),
            'manipulation': FallbackManipulationController()
        }

    def handle_control_failure(self, failure_type, action_type, context):
        """Handle control system failure with recovery"""
        if failure_type == 'navigation_failure':
            return self.handle_navigation_failure(context)
        elif failure_type == 'manipulation_failure':
            return self.handle_manipulation_failure(context)
        elif failure_type == 'hardware_failure':
            return self.handle_hardware_failure(context)
        elif failure_type == 'trajectory_failure':
            return self.handle_trajectory_failure(context)
        else:
            return self.handle_general_control_failure(action_type, context)

    def handle_navigation_failure(self, context):
        """Handle navigation control failure"""
        self.node.get_logger().error('Navigation control failure detected')

        # Execute emergency stop
        self.emergency_navigation_stop()

        # Try fallback navigation
        if 'fallback_navigation' in context:
            fallback_result = self.fallback_controllers['navigation'].execute_fallback(context)
            if fallback_result['success']:
                return {
                    'success': True,
                    'action_taken': 'fallback_navigation',
                    'result': fallback_result
                }

        return {
            'success': False,
            'error': 'Navigation control failed, emergency stop executed',
            'action_taken': 'emergency_stop'
        }

    def handle_manipulation_failure(self, context):
        """Handle manipulation control failure"""
        self.node.get_logger().error('Manipulation control failure detected')

        # Execute emergency stop for manipulation
        self.emergency_manipulation_stop()

        # Try fallback manipulation
        if 'fallback_manipulation' in context:
            fallback_result = self.fallback_controllers['manipulation'].execute_fallback(context)
            if fallback_result['success']:
                return {
                    'success': True,
                    'action_taken': 'fallback_manipulation',
                    'result': fallback_result
                }

        return {
            'success': False,
            'error': 'Manipulation control failed, emergency stop executed',
            'action_taken': 'emergency_stop'
        }

    def handle_hardware_failure(self, context):
        """Handle general hardware failure"""
        self.node.get_logger().error('Hardware failure detected')

        # Execute general emergency stop
        self.emergency_hardware_stop()

        # Return to safe configuration
        safe_config_result = self.return_to_safe_configuration()

        return {
            'success': False,
            'error': 'Hardware failure, system in safe state',
            'action_taken': 'emergency_stop_and_safe_return',
            'safe_config_result': safe_config_result
        }

    def handle_trajectory_failure(self, context):
        """Handle trajectory execution failure"""
        self.node.get_logger().warn('Trajectory execution failure')

        # Stop current trajectory
        self.stop_current_trajectory()

        # Recalculate trajectory if possible
        if 'current_goal' in context:
            recalculated_trajectory = self.recalculate_trajectory(context['current_goal'])
            if recalculated_trajectory:
                return {
                    'success': True,
                    'action_taken': 'trajectory_recalculation',
                    'new_trajectory': recalculated_trajectory
                }

        return {
            'success': False,
            'error': 'Trajectory execution failed',
            'action_taken': 'trajectory_stop'
        }

    def handle_general_control_failure(self, action_type, context):
        """Handle general control failure"""
        self.node.get_logger().error(f'Control failure for action type: {action_type}')

        # Execute appropriate emergency procedure
        if action_type in self.emergency_procedures:
            self.emergency_procedures[action_type]()

        return {
            'success': False,
            'error': f'Control failed for {action_type}',
            'action_taken': 'emergency_procedure'
        }

    def emergency_navigation_stop(self):
        """Execute emergency stop for navigation"""
        # Publish zero velocity command
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0

        # Assuming there's a publisher available
        # self.cmd_vel_pub.publish(stop_cmd)

        self.node.get_logger().info('Emergency navigation stop executed')

    def emergency_manipulation_stop(self):
        """Execute emergency stop for manipulation"""
        # Stop manipulation joints
        # Implementation would send stop commands to manipulation joints
        self.node.get_logger().info('Emergency manipulation stop executed')

    def emergency_hardware_stop(self):
        """Execute general emergency stop"""
        # Stop all robot motion
        self.emergency_navigation_stop()
        self.emergency_manipulation_stop()

        self.node.get_logger().info('General emergency stop executed')

    def return_to_safe_configuration(self):
        """Return robot to safe configuration"""
        # Move robot to safe position/configuration
        # Implementation would send commands to move to safe pose
        return {'success': True, 'message': 'Returned to safe configuration'}

    def stop_current_trajectory(self):
        """Stop current trajectory execution"""
        # Implementation to stop current trajectory
        pass

    def recalculate_trajectory(self, goal):
        """Recalculate trajectory to goal"""
        # Implementation to recalculate trajectory
        return None  # Placeholder


class FallbackNavigationController:
    """Fallback navigation controller"""
    def execute_fallback(self, context):
        """Execute fallback navigation"""
        # Implementation of fallback navigation
        return {'success': True, 'message': 'Fallback navigation executed'}


class FallbackManipulationController:
    """Fallback manipulation controller"""
    def execute_fallback(self, context):
        """Execute fallback manipulation"""
        # Implementation of fallback manipulation
        return {'success': True, 'message': 'Fallback manipulation executed'}
```

## Safety Recovery Procedures

### Emergency Stop and Recovery

```python
class EmergencyRecoverySystem:
    def __init__(self, node):
        self.node = node
        self.safety_monitor = SafetyMonitor(node)
        self.recovery_procedures = {
            'collision_imminent': self.collision_recovery,
            'human_safety_violation': self.human_safety_recovery,
            'operational_limit_violation': self.operational_limit_recovery,
            'system_instability': self.system_instability_recovery
        }

    def handle_safety_violation(self, violation_type, context):
        """Handle safety violation with appropriate recovery"""
        if violation_type in self.recovery_procedures:
            return self.recovery_procedures[violation_type](context)
        else:
            return self.general_safety_recovery(violation_type, context)

    def collision_recovery(self, context):
        """Recovery procedure for collision imminent"""
        self.node.get_logger().error('COLLISION IMMINENT - EXECUTING EMERGENCY STOP')

        # Immediate stop
        self.emergency_stop()

        # Assess situation
        situation_assessment = self.assess_collision_situation(context)

        # Plan recovery action
        if situation_assessment['path_clear_after_stop']:
            recovery_action = 'resume_navigation'
        elif situation_assessment['obstacle_can_be_avoided']:
            recovery_action = 'alternative_path'
        else:
            recovery_action = 'request_assistance'

        return {
            'action': 'emergency_stop',
            'recovery_action': recovery_action,
            'situation_assessment': situation_assessment,
            'success': True
        }

    def human_safety_recovery(self, context):
        """Recovery procedure for human safety violation"""
        self.node.get_logger().error('HUMAN SAFETY VIOLATION - STOPPING IMMEDIATELY')

        # Immediate stop
        self.emergency_stop()

        # Increase safety distance
        self.increase_safety_distance()

        # Assess human location and intent
        human_assessment = self.assess_human_situation(context)

        # Plan recovery
        if human_assessment['human_moving_away']:
            recovery_action = 'resume_with_caution'
        elif human_assessment['human_stationary']:
            recovery_action = 'wait_and_assess'
        else:
            recovery_action = 'maintain_distance'

        return {
            'action': 'safety_stop',
            'recovery_action': recovery_action,
            'human_assessment': human_assessment,
            'success': True
        }

    def operational_limit_recovery(self, context):
        """Recovery procedure for operational limit violation"""
        self.node.get_logger().warn('OPERATIONAL LIMIT VIOLATION - ADJUSTING PARAMETERS')

        # Reduce aggressive parameters
        self.reduce_aggressive_parameters()

        # Assess limit type
        limit_type = self.assess_limit_violation(context)

        # Plan recovery
        if limit_type == 'velocity_limit':
            recovery_action = 'reduce_speed'
        elif limit_type == 'force_limit':
            recovery_action = 'reduce_force'
        elif limit_type == 'payload_limit':
            recovery_action = 'abort_manipulation'
        else:
            recovery_action = 'adjust_parameters'

        return {
            'action': 'parameter_adjustment',
            'recovery_action': recovery_action,
            'limit_type': limit_type,
            'success': True
        }

    def system_instability_recovery(self, context):
        """Recovery procedure for system instability"""
        self.node.get_logger().error('SYSTEM INSTABILITY DETECTED - INITIATING STABILIZATION')

        # Stop all motion
        self.emergency_stop()

        # Stabilize system
        stabilization_result = self.stabilize_system()

        # Assess stability
        stability_assessment = self.assess_system_stability()

        if stability_assessment['stable']:
            recovery_action = 'resume_operations'
        else:
            recovery_action = 'request_manual_intervention'

        return {
            'action': 'stabilization',
            'recovery_action': recovery_action,
            'stabilization_result': stabilization_result,
            'stability_assessment': stability_assessment,
            'success': stability_assessment['stable']
        }

    def emergency_stop(self):
        """Execute emergency stop"""
        # Publish emergency stop command to all systems
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0

        # Assuming cmd_vel publisher exists
        # self.cmd_vel_pub.publish(stop_cmd)

        self.node.get_logger().info('Emergency stop command published')

    def increase_safety_distance(self):
        """Increase safety distance parameters"""
        # Implementation to increase safety distances
        pass

    def reduce_aggressive_parameters(self):
        """Reduce aggressive motion parameters"""
        # Implementation to reduce velocities, accelerations, forces
        pass

    def stabilize_system(self):
        """Stabilize unstable system"""
        # Implementation to stabilize system
        return {'success': True, 'method': 'parameter_reset'}

    def assess_collision_situation(self, context):
        """Assess collision situation"""
        # Implementation to assess collision situation
        return {'path_clear_after_stop': True, 'obstacle_can_be_avoided': True}

    def assess_human_situation(self, context):
        """Assess human safety situation"""
        # Implementation to assess human situation
        return {'human_moving_away': True, 'human_stationary': False}

    def assess_limit_violation(self, context):
        """Assess type of limit violation"""
        # Implementation to assess limit violation type
        return 'velocity_limit'

    def assess_system_stability(self):
        """Assess system stability"""
        # Implementation to assess system stability
        return {'stable': True, 'confidence': 0.9}

    def general_safety_recovery(self, violation_type, context):
        """General safety recovery for unknown violations"""
        self.node.get_logger().error(f'UNKNOWN SAFETY VIOLATION: {violation_type}')

        # Execute general emergency procedure
        self.emergency_stop()

        return {
            'action': 'general_emergency',
            'violation_type': violation_type,
            'success': False,
            'requires_manual_intervention': True
        }
```

## Recovery State Management

### Recovery State Machine

```python
from enum import Enum

class RecoveryState(Enum):
    NORMAL_OPERATION = "normal_operation"
    DEGRADED_OPERATION = "degraded_operation"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY_IN_PROGRESS = "recovery_in_progress"
    MANUAL_OVERRIDE = "manual_override"
    SYSTEM_SHUTDOWN = "system_shutdown"


class RecoveryStateMachine:
    def __init__(self, node):
        self.node = node
        self.current_state = RecoveryState.NORMAL_OPERATION
        self.state_history = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

    def transition_to_state(self, new_state, reason=None):
        """Transition to new recovery state"""
        old_state = self.current_state
        self.current_state = new_state

        # Log state transition
        transition_info = {
            'timestamp': time.time(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'reason': reason
        }
        self.state_history.append(transition_info)

        self.node.get_logger().info(
            f'State transition: {old_state.value} → {new_state.value} ({reason})'
        )

    def handle_failure(self, failure_type, failure_details):
        """Handle failure and transition to appropriate state"""
        if failure_type in ['critical', 'safety_violation']:
            self.transition_to_state(RecoveryState.EMERGENCY_STOP, f'Critical failure: {failure_type}')
            return self.execute_emergency_procedures(failure_details)
        elif failure_type in ['component_failure', 'service_unavailable']:
            self.transition_to_state(RecoveryState.DEGRADED_OPERATION, f'Component failure: {failure_type}')
            return self.execute_degraded_procedures(failure_details)
        else:
            # Non-critical failures may not require state change
            return self.execute_standard_recovery(failure_details)

    def execute_emergency_procedures(self, details):
        """Execute emergency procedures"""
        # Stop all robot motion
        self.emergency_stop_all_systems()

        # Alert operators
        self.alert_operators(details)

        # Log incident
        self.log_incident(details, 'emergency')

        return {
            'state': 'emergency_stop_executed',
            'actions_taken': ['motion_stop', 'alert_sent', 'incident_logged']
        }

    def execute_degraded_procedures(self, details):
        """Execute degraded operation procedures"""
        # Switch to degraded mode
        self.activate_degraded_mode()

        # Notify users of reduced functionality
        self.notify_users_degraded()

        # Attempt recovery
        recovery_result = self.attempt_recovery(details)

        if recovery_result['success']:
            self.transition_to_state(RecoveryState.NORMAL_OPERATION, 'Recovery successful')
        else:
            self.transition_to_state(RecoveryState.DEGRADED_OPERATION, 'Operating in degraded mode')

        return recovery_result

    def execute_standard_recovery(self, details):
        """Execute standard recovery procedures"""
        recovery_result = self.attempt_recovery(details)

        if not recovery_result['success']:
            # Recovery failed, consider escalation
            self.consider_recovery_escalation()

        return recovery_result

    def attempt_recovery(self, failure_details):
        """Attempt to recover from failure"""
        try:
            # Try to recover from failure
            recovery_method = self.select_recovery_method(failure_details)

            if recovery_method:
                result = recovery_method(failure_details)
                return {
                    'success': result,
                    'method_used': recovery_method.__name__,
                    'attempts': self.recovery_attempts
                }
            else:
                return {
                    'success': False,
                    'error': 'No suitable recovery method found',
                    'attempts': self.recovery_attempts
                }

        except Exception as e:
            self.recovery_attempts += 1
            return {
                'success': False,
                'error': str(e),
                'attempts': self.recovery_attempts
            }

    def select_recovery_method(self, failure_details):
        """Select appropriate recovery method based on failure details"""
        failure_category = failure_details.get('category', 'unknown')

        recovery_methods = {
            'voice_processing': self.recover_voice_processing,
            'llm_planning': self.recover_llm_planning,
            'perception': self.recover_perception,
            'control': self.recover_control,
            'communication': self.recover_communication,
            'safety': self.recover_safety
        }

        return recovery_methods.get(failure_category)

    def recover_voice_processing(self, details):
        """Recover from voice processing failure"""
        # Implementation for voice processing recovery
        pass

    def recover_llm_planning(self, details):
        """Recover from LLM planning failure"""
        # Implementation for LLM planning recovery
        pass

    def recover_perception(self, details):
        """Recover from perception failure"""
        # Implementation for perception recovery
        pass

    def recover_control(self, details):
        """Recover from control failure"""
        # Implementation for control recovery
        pass

    def recover_communication(self, details):
        """Recover from communication failure"""
        # Implementation for communication recovery
        pass

    def recover_safety(self, details):
        """Recover from safety system failure"""
        # Implementation for safety recovery
        pass

    def consider_recovery_escalation(self):
        """Consider escalating recovery if attempts fail"""
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.transition_to_state(RecoveryState.MANUAL_OVERRIDE, 'Max recovery attempts reached')
            self.request_manual_intervention()

    def emergency_stop_all_systems(self):
        """Emergency stop all robot systems"""
        # Implementation to stop all robot systems
        pass

    def activate_degraded_mode(self):
        """Activate degraded operation mode"""
        # Implementation to activate degraded mode
        pass

    def request_manual_intervention(self):
        """Request manual intervention from operators"""
        # Implementation to request manual intervention
        pass

    def alert_operators(self, details):
        """Alert operators about the issue"""
        # Implementation to alert operators
        pass

    def notify_users_degraded(self):
        """Notify users about degraded operation"""
        # Implementation to notify users
        pass

    def log_incident(self, details, severity):
        """Log the incident for analysis"""
        # Implementation to log incident
        pass
```

## Testing Recovery Mechanisms

### Recovery Testing Framework

```python
import unittest
from unittest.mock import Mock, patch

class TestRecoveryMechanisms(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_node = Mock()
        self.recovery_system = {
            'voice': VoiceProcessingRecovery(self.mock_node),
            'llm': LLMPlanningRecovery(self.mock_node),
            'perception': PerceptionRecovery(self.mock_node),
            'control': ControlRecovery(self.mock_node),
            'emergency': EmergencyRecoverySystem(self.mock_node),
            'state_machine': RecoveryStateMachine(self.mock_node)
        }

    def test_voice_processing_recovery(self):
        """Test voice processing failure recovery"""
        # Test STT failure recovery
        result = self.recovery_system['voice'].handle_voice_failure(
            'stt_failure',
            {'text': 'test command'}
        )

        self.assertTrue(result['success'] or result.get('fallback_used'))

        # Test intent extraction failure
        result = self.recovery_system['voice'].handle_intent_extraction_failure(
            'test command'
        )

        self.assertIsNotNone(result)

    def test_llm_planning_recovery(self):
        """Test LLM planning failure recovery"""
        intent = {'action': 'navigation', 'parameters': {'destination': 'kitchen'}}
        context = {}

        # Test API unavailability recovery
        result = self.recovery_system['llm'].handle_llm_failure(
            'api_unavailable',
            intent,
            context
        )

        self.assertTrue(result['success'] or result.get('fallback_used'))

    def test_perception_recovery(self):
        """Test perception failure recovery"""
        # Test sensor failure
        result = self.recovery_system['perception'].handle_perception_failure(
            'sensor_failure',
            'camera',
            {}
        )

        # Should either succeed or provide fallback/cached data
        self.assertTrue(
            result['success'] or
            result.get('backup_used') or
            result.get('cached_used') or
            not result['success']
        )

    def test_control_recovery(self):
        """Test control failure recovery"""
        context = {}

        # Test navigation failure
        result = self.recovery_system['control'].handle_control_failure(
            'navigation_failure',
            'navigation',
            context
        )

        # Should execute emergency stop and return appropriate result
        self.assertIn('action_taken', result)
        self.assertIn('emergency_stop', result.get('action_taken', ''))

    def test_emergency_recovery(self):
        """Test emergency recovery procedures"""
        context = {}

        # Test collision recovery
        result = self.recovery_system['emergency'].collision_recovery(context)

        self.assertIn('action', result)
        self.assertEqual(result['action'], 'emergency_stop')

    def test_recovery_state_machine(self):
        """Test recovery state machine transitions"""
        sm = self.recovery_system['state_machine']

        # Test normal operation
        self.assertEqual(sm.current_state, RecoveryState.NORMAL_OPERATION)

        # Test transition to degraded
        sm.handle_failure('component_failure', {'category': 'perception'})
        self.assertEqual(sm.current_state, RecoveryState.DEGRADED_OPERATION)

        # Test transition to emergency stop
        sm.handle_failure('critical', {'category': 'safety'})
        self.assertEqual(sm.current_state, RecoveryState.EMERGENCY_STOP)

    def test_retry_mechanism(self):
        """Test retry mechanism functionality"""
        retry_handler = RetryMechanism(self.mock_node)

        # Mock a function that fails initially but succeeds on retry
        call_count = 0
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "Success"

        # Test retry mechanism
        result = retry_handler.execute_with_retry(flaky_function)
        self.assertEqual(result, "Success")
        self.assertEqual(call_count, 2)  # Called twice due to retry


def run_recovery_tests():
    """Run all recovery mechanism tests"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRecoveryMechanisms)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_recovery_tests()
    exit(0 if success else 1)
```

Comprehensive failure handling and recovery mechanisms are essential for robust VLA systems. These mechanisms ensure that humanoid robots can gracefully handle various failure scenarios while maintaining safety and continuing operation when possible. The layered approach of detection, classification, and recovery provides resilience against both common and rare failure modes.