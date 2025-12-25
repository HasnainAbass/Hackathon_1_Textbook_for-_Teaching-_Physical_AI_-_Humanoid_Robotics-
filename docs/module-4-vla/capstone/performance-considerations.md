# Performance Considerations for Complete VLA Pipeline

## Introduction

Performance optimization is crucial for Vision-Language-Action (VLA) systems, especially when deployed on resource-constrained robotic platforms. This chapter covers performance considerations across all VLA system components, including optimization strategies, resource management, and real-time performance requirements. The goal is to maintain system responsiveness and reliability while maximizing the effectiveness of autonomous behaviors.

## Performance Requirements

### Real-Time Constraints

VLA systems must meet specific real-time constraints to operate effectively:

#### Response Time Requirements
- **Voice Command Processing**: < 2 seconds from voice input to action initiation
- **LLM Query Response**: < 5 seconds for complex planning queries
- **Perception Processing**: < 100ms for basic object detection
- **Navigation Commands**: < 50ms for reactive obstacle avoidance
- **Manipulation Planning**: < 2 seconds for grasp planning

#### Throughput Requirements
- **Command Processing Rate**: ≥ 1 command per 2 seconds
- **Perception Update Rate**: ≥ 10 Hz for navigation, ≥ 5 Hz for manipulation
- **Control Update Rate**: ≥ 50 Hz for stable control
- **System Monitoring**: ≥ 1 Hz for safety checks

#### Latency Budget Allocation
```
Total Response Time Budget: 2000ms
├── Voice Processing: 300ms (15%)
├── Intent Extraction: 200ms (10%)
├── LLM Planning: 1000ms (50%)
├── Perception Integration: 300ms (15%)
├── Safety Validation: 100ms (5%)
└── Control Execution: 100ms (5%)
```

## System Architecture for Performance

### Component Optimization

#### Asynchronous Processing Architecture

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, PriorityQueue
import time
import multiprocessing as mp


class AsyncVLAProcessor:
    def __init__(self, node):
        self.node = node

        # Thread pools for I/O-bound operations
        self.io_pool = ThreadPoolExecutor(max_workers=4)

        # Process pools for CPU-bound operations
        self.cpu_pool = ProcessPoolExecutor(max_workers=2)

        # Priority queues for different task types
        self.high_priority_queue = PriorityQueue()
        self.normal_priority_queue = PriorityQueue()
        self.low_priority_queue = PriorityQueue()

        # Async event loop
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self.loop_thread.start()

        # Performance monitors
        self.performance_monitors = PerformanceMonitors(node)

    def run_event_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def process_voice_command_async(self, command_text):
        """Process voice command asynchronously"""
        start_time = time.time()

        # Process in parallel where possible
        tasks = [
            self.extract_intent_async(command_text),
            self.get_perception_context_async(),
            self.check_safety_context_async()
        ]

        intent, perception_context, safety_context = await asyncio.gather(*tasks)

        # Plan using LLM (CPU-intensive)
        plan = await self.plan_with_llm_async(intent, perception_context, safety_context)

        # Execute plan
        execution_result = await self.execute_plan_async(plan)

        processing_time = time.time() - start_time
        self.performance_monitors.record_response_time('voice_processing', processing_time)

        return {
            'success': True,
            'execution_result': execution_result,
            'processing_time': processing_time
        }

    async def extract_intent_async(self, text):
        """Extract intent asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, self.sync_extract_intent, text)

    async def get_perception_context_async(self):
        """Get perception context asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, self.sync_get_perception_context)

    async def check_safety_context_async(self):
        """Check safety context asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, self.sync_check_safety_context)

    async def plan_with_llm_async(self, intent, perception_context, safety_context):
        """Plan with LLM using process pool for CPU-intensive work"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_pool,
            self.sync_plan_with_llm,
            intent, perception_context, safety_context
        )

    async def execute_plan_async(self, plan):
        """Execute plan asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, self.sync_execute_plan, plan)

    def sync_extract_intent(self, text):
        """Synchronous intent extraction (runs in thread pool)"""
        # Implementation of intent extraction
        return {'action': 'navigation', 'parameters': {'location': 'kitchen'}, 'confidence': 0.9}

    def sync_get_perception_context(self):
        """Synchronous perception context retrieval"""
        # Implementation of perception context retrieval
        return {'objects': [], 'map': {}, 'timestamp': time.time()}

    def sync_check_safety_context(self):
        """Synchronous safety context check"""
        # Implementation of safety context check
        return {'safe': True, 'violations': []}

    def sync_plan_with_llm(self, intent, perception_context, safety_context):
        """Synchronous LLM planning (runs in process pool)"""
        # Implementation of LLM-based planning
        return {'action_sequence': [], 'estimated_duration': 30}

    def sync_execute_plan(self, plan):
        """Synchronous plan execution"""
        # Implementation of plan execution
        return {'success': True, 'message': 'Plan executed successfully'}


class PerformanceMonitors:
    def __init__(self, node):
        self.node = node
        self.metrics = {
            'response_times': {},
            'throughput': {},
            'resource_usage': {},
            'error_rates': {}
        }
        self.performance_thresholds = {
            'max_response_time': 2.0,  # seconds
            'min_throughput': 0.5,     # commands per second
            'max_cpu_usage': 80.0,     # percent
            'max_memory_usage': 80.0   # percent
        }

    def record_response_time(self, component, response_time):
        """Record response time for performance monitoring"""
        if component not in self.metrics['response_times']:
            self.metrics['response_times'][component] = []

        self.metrics['response_times'][component].append(response_time)

        # Check against threshold
        if response_time > self.performance_thresholds['max_response_time']:
            self.node.get_logger().warn(
                f'{component} response time {response_time:.3f}s exceeds threshold'
            )

    def record_throughput(self, component, commands_processed, time_period):
        """Record throughput for component"""
        if component not in self.metrics['throughput']:
            self.metrics['throughput'][component] = []

        throughput = commands_processed / time_period
        self.metrics['throughput'][component].append(throughput)

        # Check against threshold
        if throughput < self.performance_thresholds['min_throughput']:
            self.node.get_logger().warn(
                f'{component} throughput {throughput:.2f} below threshold'
            )

    def record_resource_usage(self):
        """Record system resource usage"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        cpu_percent = psutil.cpu_percent()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = psutil.virtual_memory().percent

        usage_data = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'memory_percent': memory_percent,
            'process_count': len(psutil.pids())
        }

        if 'system' not in self.metrics['resource_usage']:
            self.metrics['resource_usage']['system'] = []
        self.metrics['resource_usage']['system'].append(usage_data)

        # Check against thresholds
        if cpu_percent > self.performance_thresholds['max_cpu_usage']:
            self.node.get_logger().warn(f'CPU usage {cpu_percent:.1f}% exceeds threshold')
        if memory_percent > self.performance_thresholds['max_memory_usage']:
            self.node.get_logger().warn(f'Memory usage {memory_percent:.1f}% exceeds threshold')

    def get_performance_summary(self):
        """Get performance summary"""
        summary = {}

        for category, data in self.metrics.items():
            summary[category] = {}
            for component, values in data.items():
                if values:
                    if isinstance(values[0], dict) and 'cpu_percent' in values[0]:
                        # Resource usage data
                        cpu_values = [v['cpu_percent'] for v in values]
                        mem_values = [v['memory_percent'] for v in values]

                        summary[category][component] = {
                            'avg_cpu': sum(cpu_values) / len(cpu_values),
                            'avg_memory': sum(mem_values) / len(mem_values),
                            'sample_count': len(values)
                        }
                    else:
                        # Numeric performance data
                        avg_value = sum(values) / len(values)
                        min_value = min(values)
                        max_value = max(values)

                        summary[category][component] = {
                            'average': avg_value,
                            'min': min_value,
                            'max': max_value,
                            'sample_count': len(values)
                        }

        return summary
```

### Pipeline Optimization

#### Parallel Processing Pipeline

```python
class ParallelProcessingPipeline:
    def __init__(self, node):
        self.node = node
        self.parallel_components = {
            'perception': ThreadPoolExecutor(max_workers=2),
            'planning': ProcessPoolExecutor(max_workers=1),  # LLM calls are sequential
            'control': ThreadPoolExecutor(max_workers=2),
            'monitoring': ThreadPoolExecutor(max_workers=1)
        }
        self.shared_context = SharedContext()
        self.pipeline_scheduler = PipelineScheduler()

    def execute_pipeline_parallel(self, voice_command):
        """Execute VLA pipeline with parallel components"""
        start_time = time.time()

        # Start perception processing in background
        perception_future = self.parallel_components['perception'].submit(
            self.process_perception_data
        )

        # Process voice command
        intent = self.process_voice_command(voice_command)

        # Get perception data (with timeout)
        try:
            perception_data = perception_future.result(timeout=1.0)
        except TimeoutError:
            perception_data = self.get_cached_perception_data()
            self.node.get_logger().warn('Perception data timeout, using cached data')

        # Plan using LLM
        plan = self.plan_with_llm(intent, perception_data)

        # Integrate safety checks
        safe_plan = self.validate_plan_safety(plan)

        # Execute plan
        execution_result = self.execute_plan(safe_plan)

        total_time = time.time() - start_time
        self.record_pipeline_performance(total_time)

        return {
            'success': True,
            'execution_result': execution_result,
            'pipeline_time': total_time,
            'components_used': ['voice', 'perception', 'planning', 'control']
        }

    def process_perception_data(self):
        """Process perception data in parallel"""
        # Get latest perception data
        latest_image = self.get_latest_image()
        latest_scan = self.get_latest_scan()

        # Process in parallel
        object_detections = self.detect_objects(latest_image)
        obstacle_map = self.process_scan_data(latest_scan)

        return {
            'objects': object_detections,
            'obstacles': obstacle_map,
            'timestamp': time.time()
        }

    def record_pipeline_performance(self, total_time):
        """Record overall pipeline performance"""
        self.node.get_logger().debug(f'Pipeline completed in {total_time:.3f}s')
        # Could add to performance metrics database
```

## Resource Management

### Memory Optimization

```python
import gc
import weakref
from collections import OrderedDict
import numpy as np


class MemoryOptimizer:
    def __init__(self, node):
        self.node = node
        self.lru_cache = LRUCache(max_size=100)
        self.tensor_pool = TensorPool()
        self.memory_threshold = 0.8  # 80% memory usage threshold

    def optimize_memory_usage(self):
        """Optimize memory usage across system"""
        import psutil
        memory_percent = psutil.virtual_memory().percent

        if memory_percent > self.memory_threshold * 100:
            self.perform_memory_cleanup()
            self.trigger_gc_collection()

    def perform_memory_cleanup(self):
        """Perform memory cleanup operations"""
        # Clear LRU cache if needed
        if len(self.lru_cache) > 80:  # 80% of capacity
            self.lru_cache.clear_old_entries()

        # Clear tensor pool
        self.tensor_pool.cleanup_unused_tensors()

        # Clear perception data cache
        self.clear_perception_cache()

        self.node.get_logger().info('Memory cleanup performed')

    def trigger_gc_collection(self):
        """Trigger garbage collection"""
        collected = gc.collect()
        self.node.get_logger().debug(f'Garbage collection collected {collected} objects')

    def clear_perception_cache(self):
        """Clear perception data cache"""
        # Implementation to clear cached perception data
        pass


class LRUCache:
    """Least Recently Used Cache for frequently accessed data"""
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        """Get value from cache, move to end (most recently used)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        """Put value in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            self.cache.popitem(last=False)

        self.cache[key] = value

    def clear_old_entries(self):
        """Clear oldest entries to free memory"""
        items_to_remove = len(self.cache) // 4  # Remove 25% of entries
        for _ in range(items_to_remove):
            if self.cache:
                self.cache.popitem(last=False)


class TensorPool:
    """Pool for reusing tensors to reduce allocation overhead"""
    def __init__(self):
        self.pool = []
        self.max_pool_size = 10

    def get_tensor(self, shape, dtype=np.float32):
        """Get tensor from pool or create new one"""
        for i, (tensor_shape, tensor_dtype, tensor) in enumerate(self.pool):
            if tensor_shape == shape and tensor_dtype == dtype:
                # Found matching tensor, remove from pool and return
                del self.pool[i]
                return tensor

        # No matching tensor found, create new one
        return np.zeros(shape, dtype=dtype)

    def return_tensor(self, tensor):
        """Return tensor to pool for reuse"""
        if len(self.pool) < self.max_pool_size:
            self.pool.append((tensor.shape, tensor.dtype, tensor))

    def cleanup_unused_tensors(self):
        """Clean up unused tensors"""
        self.pool.clear()
```

### CPU Optimization

```python
import multiprocessing as mp
import numba
from functools import lru_cache


class CPUOptimizer:
    def __init__(self, node):
        self.node = node
        self.num_cores = mp.cpu_count()
        self.process_pool = mp.Pool(processes=max(1, self.num_cores - 1))  # Leave one core for main thread
        self.optimization_thresholds = {
            'cpu_usage_limit': 80.0,  # percent
            'process_count_limit': self.num_cores * 2
        }

    def optimize_cpu_usage(self):
        """Optimize CPU usage across system"""
        import psutil
        cpu_percent = psutil.cpu_percent()

        if cpu_percent > self.optimization_thresholds['cpu_usage_limit']:
            self.reduce_computation_intensity()

    def reduce_computation_intensity(self):
        """Reduce computation intensity when CPU is overloaded"""
        # Reduce perception processing frequency
        self.reduce_perception_frequency()

        # Use simpler planning algorithms
        self.use_simplified_planning()

        # Reduce control update rate
        self.reduce_control_frequency()

        self.node.get_logger().warn('CPU usage high, reducing computation intensity')

    def reduce_perception_frequency(self):
        """Reduce perception processing frequency"""
        # Implementation to reduce perception processing rate
        pass

    def use_simplified_planning(self):
        """Use simplified planning algorithms under high CPU load"""
        # Implementation to use faster but less sophisticated planning
        pass

    def reduce_control_frequency(self):
        """Reduce control system update frequency"""
        # Implementation to reduce control system update rate
        pass


# Numba-optimized functions for performance-critical operations
@numba.jit(nopython=True)
def optimized_distance_calculation(points1, points2):
    """Optimized distance calculation using Numba"""
    distances = np.zeros(len(points1))
    for i in range(len(points1)):
        dx = points1[i, 0] - points2[i, 0]
        dy = points1[i, 1] - points2[i, 1]
        distances[i] = np.sqrt(dx*dx + dy*dy)
    return distances


@numba.jit(nopython=True)
def optimized_collision_check(robot_pose, obstacles, robot_radius):
    """Optimized collision checking"""
    for obs in obstacles:
        dx = robot_pose[0] - obs[0]
        dy = robot_pose[1] - obs[1]
        distance = np.sqrt(dx*dx + dy*dy)
        if distance < (robot_radius + obs[2]):  # obs[2] is obstacle radius
            return True
    return False
```

## LLM Performance Optimization

### Caching and Batching

```python
import hashlib
import asyncio
from typing import Dict, Any, Optional


class LLMPerformanceOptimizer:
    def __init__(self, node, llm_client):
        self.node = node
        self.llm_client = llm_client
        self.response_cache = ResponseCache(max_size=50)
        self.request_batcher = RequestBatcher(batch_size=3, timeout=2.0)
        self.rate_limiter = RateLimiter(max_requests_per_minute=100)
        self.compression_enabled = True

    async def optimized_llm_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Optimized LLM call with caching and rate limiting"""
        # Check cache first
        cache_key = self.generate_cache_key(prompt, kwargs)
        cached_response = self.response_cache.get(cache_key)

        if cached_response is not None:
            self.node.get_logger().debug('LLM response served from cache')
            return cached_response

        # Check rate limit
        if not self.rate_limiter.can_make_request():
            self.node.get_logger().warn('LLM rate limit reached, using fallback')
            return self.generate_fallback_response(prompt)

        # Use batcher for multiple requests
        if self.request_batcher.has_pending_requests():
            return await self.request_batcher.add_to_batch(prompt, **kwargs)

        # Make direct call
        start_time = time.time()
        try:
            response = await self.make_llm_call(prompt, **kwargs)

            # Cache response
            self.response_cache.put(cache_key, response)

            processing_time = time.time() - start_time
            self.node.get_logger().debug(f'LLM call completed in {processing_time:.3f}s')

            return response

        except Exception as e:
            self.node.get_logger().error(f'LLM call failed: {e}')
            return self.generate_fallback_response(prompt)

    def generate_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters"""
        combined = prompt + str(sorted(kwargs.items()))
        return hashlib.md5(combined.encode()).hexdigest()

    async def make_llm_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Make actual LLM call"""
        return await self.llm_client.chat.completions.create(
            model=kwargs.get('model', 'gpt-3.5-turbo'),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.1),
            max_tokens=kwargs.get('max_tokens', 1000)
        )

    def generate_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Generate fallback response when LLM is unavailable"""
        # Implement rule-based or template-based fallback
        return {
            'choices': [{
                'message': {
                    'content': f'LLM unavailable, fallback response for: {prompt[:50]}...'
                }
            }],
            'usage': {'total_tokens': 0}
        }


class ResponseCache:
    """Cache for LLM responses"""
    def __init__(self, max_size=50):
        self.cache = LRUCache(max_size=max_size)
        self.ttl = 300  # 5 minutes TTL

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cached_item = self.cache.get(key)
        if cached_item is None:
            return None

        response, timestamp = cached_item
        if time.time() - timestamp > self.ttl:
            # Expired, remove from cache
            self.cache.cache.pop(key, None)
            return None

        return response

    def put(self, key: str, response: Dict[str, Any]):
        """Put response in cache"""
        self.cache.put(key, (response, time.time()))


class RequestBatcher:
    """Batch multiple LLM requests for efficiency"""
    def __init__(self, batch_size=3, timeout=2.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.futures = []

    def has_pending_requests(self) -> bool:
        """Check if there are pending requests"""
        return len(self.pending_requests) > 0

    async def add_to_batch(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Add request to batch"""
        future = asyncio.Future()
        request = {
            'prompt': prompt,
            'kwargs': kwargs,
            'future': future
        }
        self.pending_requests.append(request)
        self.futures.append(future)

        if len(self.pending_requests) >= self.batch_size:
            await self.process_batch()

        return await future

    async def process_batch(self):
        """Process batch of requests"""
        # Process all pending requests
        for request in self.pending_requests:
            response = await self.process_single_request(request['prompt'], **request['kwargs'])
            request['future'].set_result(response)

        # Clear processed requests
        self.pending_requests.clear()
        self.futures.clear()


class RateLimiter:
    """Simple rate limiter for LLM API calls"""
    def __init__(self, max_requests_per_minute=100):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.window = 60  # 1 minute window

    def can_make_request(self) -> bool:
        """Check if request can be made"""
        current_time = time.time()

        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests
                        if current_time - req_time < self.window]

        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True

        return False
```

## Real-Time Performance Monitoring

### Performance Monitoring System

```python
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from collections import deque
import json


@dataclass
class PerformanceMetric:
    """Data class for performance metrics"""
    component: str
    metric_type: str
    value: float
    timestamp: float
    unit: str


class RealTimePerformanceMonitor:
    def __init__(self, node):
        self.node = node
        self.metrics_buffer = deque(maxlen=1000)  # Circular buffer for metrics
        self.component_stats = {}
        self.performance_alerts = []
        self.monitoring_enabled = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.run_monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Performance thresholds
        self.thresholds = {
            'voice_processing_time': 2.0,      # seconds
            'llm_response_time': 5.0,          # seconds
            'perception_rate': 10.0,           # Hz
            'control_rate': 50.0,              # Hz
            'cpu_usage': 80.0,                 # percent
            'memory_usage': 80.0,              # percent
            'error_rate': 0.1                  # fraction
        }

    def record_metric(self, component: str, metric_type: str, value: float, unit: str = ''):
        """Record performance metric"""
        if not self.monitoring_enabled:
            return

        metric = PerformanceMetric(
            component=component,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            unit=unit
        )

        self.metrics_buffer.append(metric)

        # Update component stats
        if component not in self.component_stats:
            self.component_stats[component] = {
                'metrics': {},
                'alerts': []
            }

        if metric_type not in self.component_stats[component]['metrics']:
            self.component_stats[component]['metrics'][metric_type] = deque(maxlen=100)

        self.component_stats[component]['metrics'][metric_type].append(metric)

        # Check thresholds
        self.check_threshold_violations(metric)

    def check_threshold_violations(self, metric: PerformanceMetric):
        """Check if metric violates performance thresholds"""
        threshold_key = f"{metric.component}_{metric.metric_type}".replace(' ', '_')

        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]

            if metric.value > threshold:
                alert = {
                    'timestamp': metric.timestamp,
                    'component': metric.component,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'threshold': threshold,
                    'severity': 'warning' if metric.value < threshold * 1.2 else 'critical'
                }

                self.performance_alerts.append(alert)
                self.node.get_logger().warn(
                    f"PERFORMANCE ALERT: {metric.component}.{metric.metric_type} "
                    f"({metric.value}{metric.unit}) exceeds threshold ({threshold})"
                )

    def run_monitoring_loop(self):
        """Run continuous monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Record system metrics
                self.record_system_metrics()

                # Sleep for monitoring interval
                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.node.get_logger().error(f'Performance monitoring error: {e}')
                time.sleep(1.0)  # Continue monitoring even if error occurs

    def record_system_metrics(self):
        """Record system-level metrics"""
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.record_metric('system', 'cpu_usage', cpu_percent, '%')

        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        self.record_metric('system', 'memory_usage', memory_percent, '%')

        # Disk usage
        disk_percent = psutil.disk_usage('/').percent
        self.record_metric('system', 'disk_usage', disk_percent, '%')

        # Network I/O
        net_io = psutil.net_io_counters()
        self.record_metric('system', 'network_bytes_sent', net_io.bytes_sent, 'bytes')
        self.record_metric('system', 'network_bytes_recv', net_io.bytes_recv, 'bytes')

    def get_component_performance(self, component: str) -> Dict:
        """Get performance summary for specific component"""
        if component not in self.component_stats:
            return {}

        summary = {'component': component, 'metrics': {}}

        for metric_type, metrics in self.component_stats[component]['metrics'].items():
            if metrics:
                values = [m.value for m in metrics]
                summary['metrics'][metric_type] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'recent_values': values[-10:]  # Last 10 values
                }

        return summary

    def get_system_performance_summary(self) -> Dict:
        """Get overall system performance summary"""
        summary = {
            'timestamp': time.time(),
            'total_metrics_recorded': len(self.metrics_buffer),
            'active_alerts': len(self.performance_alerts),
            'component_summaries': {}
        }

        for component in self.component_stats.keys():
            summary['component_summaries'][component] = self.get_component_performance(component)

        return summary

    def generate_performance_report(self) -> str:
        """Generate performance report"""
        summary = self.get_system_performance_summary()

        report_lines = [
            "# VLA System Performance Report",
            f"Generated: {time.ctime(summary['timestamp'])}",
            f"Total Metrics Recorded: {summary['total_metrics_recorded']}",
            f"Active Alerts: {summary['active_alerts']}",
            ""
        ]

        # Add component summaries
        for comp_name, comp_summary in summary['component_summaries'].items():
            report_lines.append(f"## Component: {comp_name}")

            for metric_name, metric_data in comp_summary.get('metrics', {}).items():
                report_lines.append(f"  - {metric_name}:")
                report_lines.append(f"    - Average: {metric_data['average']:.3f}")
                report_lines.append(f"    - Range: {metric_data['min']:.3f} - {metric_data['max']:.3f}")
                report_lines.append(f"    - Count: {metric_data['count']}")

            report_lines.append("")  # Empty line between components

        return "\n".join(report_lines)

    def save_performance_data(self, filename: str):
        """Save performance data to file"""
        data = {
            'metrics': [
                {
                    'component': m.component,
                    'metric_type': m.metric_type,
                    'value': m.value,
                    'timestamp': m.timestamp,
                    'unit': m.unit
                }
                for m in self.metrics_buffer
            ],
            'alerts': self.performance_alerts,
            'summary': self.get_system_performance_summary()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        self.node.get_logger().info(f'Performance data saved to {filename}')
```

## Performance Testing and Benchmarking

### Performance Benchmark Suite

```python
import time
import statistics
from typing import Dict, List, Callable
import asyncio


class PerformanceBenchmarkSuite:
    def __init__(self, node):
        self.node = node
        self.test_results = {}
        self.benchmark_functions = {
            'voice_processing': self.benchmark_voice_processing,
            'llm_query': self.benchmark_llm_query,
            'perception': self.benchmark_perception,
            'navigation': self.benchmark_navigation,
            'manipulation': self.benchmark_manipulation,
            'end_to_end': self.benchmark_end_to_end_pipeline
        }

    def run_all_benchmarks(self) -> Dict[str, Dict]:
        """Run all performance benchmarks"""
        results = {}

        for test_name, benchmark_func in self.benchmark_functions.items():
            self.node.get_logger().info(f'Running benchmark: {test_name}')
            try:
                result = benchmark_func()
                results[test_name] = result
                self.test_results[test_name] = result
            except Exception as e:
                self.node.get_logger().error(f'Benchmark {test_name} failed: {e}')
                results[test_name] = {'error': str(e), 'success': False}

        return results

    def benchmark_voice_processing(self) -> Dict:
        """Benchmark voice processing performance"""
        test_commands = [
            "Move forward 2 meters",
            "Go to the kitchen",
            "Pick up the red cup",
            "Turn left 90 degrees",
            "Stop the robot"
        ]

        times = []
        for command in test_commands:
            start_time = time.time()

            # Simulate voice processing
            result = self.simulate_voice_processing(command)

            processing_time = time.time() - start_time
            times.append(processing_time)

        return {
            'success': True,
            'average_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
            'throughput': len(test_commands) / sum(times),  # commands per second
            'sample_size': len(times)
        }

    def benchmark_llm_query(self) -> Dict:
        """Benchmark LLM query performance"""
        test_prompts = [
            "Plan navigation to kitchen",
            "Plan manipulation to pick up cup",
            "Plan complex task with multiple steps",
            "Plan safety-conscious navigation",
            "Plan human-aware manipulation"
        ]

        times = []
        for prompt in test_prompts:
            start_time = time.time()

            # Simulate LLM call
            result = self.simulate_llm_call(prompt)

            processing_time = time.time() - start_time
            times.append(processing_time)

        return {
            'success': True,
            'average_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
            'throughput': len(test_prompts) / sum(times),  # queries per second
            'sample_size': len(times)
        }

    def benchmark_perception(self) -> Dict:
        """Benchmark perception system performance"""
        import numpy as np

        # Simulate different image sizes and complexities
        test_configs = [
            {'width': 640, 'height': 480, 'objects': 1},
            {'width': 640, 'height': 480, 'objects': 5},
            {'width': 1280, 'height': 720, 'objects': 1},
            {'width': 1280, 'height': 720, 'objects': 10}
        ]

        times = []
        for config in test_configs:
            # Create simulated image
            image = np.random.randint(0, 255, (config['height'], config['width'], 3), dtype=np.uint8)

            start_time = time.time()

            # Simulate perception processing
            result = self.simulate_perception_processing(image, config['objects'])

            processing_time = time.time() - start_time
            times.append(processing_time)

        return {
            'success': True,
            'average_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
            'throughput': len(test_configs) / sum(times),  # detections per second
            'sample_size': len(times)
        }

    def benchmark_navigation(self) -> Dict:
        """Benchmark navigation performance"""
        test_routes = [
            {'start': (0, 0), 'goal': (5, 5), 'obstacles': 0},
            {'start': (0, 0), 'goal': (5, 5), 'obstacles': 5},
            {'start': (0, 0), 'goal': (10, 10), 'obstacles': 0},
            {'start': (0, 0), 'goal': (10, 10), 'obstacles': 10}
        ]

        times = []
        for route in test_routes:
            start_time = time.time()

            # Simulate navigation planning and execution
            result = self.simulate_navigation(route)

            processing_time = time.time() - start_time
            times.append(processing_time)

        return {
            'success': True,
            'average_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
            'throughput': len(test_routes) / sum(times),  # routes per second
            'sample_size': len(times)
        }

    def benchmark_manipulation(self) -> Dict:
        """Benchmark manipulation performance"""
        test_objects = [
            {'type': 'cup', 'size': 'small', 'weight': 0.1},
            {'type': 'ball', 'size': 'medium', 'weight': 0.2},
            {'type': 'box', 'size': 'large', 'weight': 0.5},
            {'type': 'bottle', 'size': 'medium', 'weight': 0.3}
        ]

        times = []
        for obj in test_objects:
            start_time = time.time()

            # Simulate manipulation planning and execution
            result = self.simulate_manipulation(obj)

            processing_time = time.time() - start_time
            times.append(processing_time)

        return {
            'success': True,
            'average_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
            'throughput': len(test_objects) / sum(times),  # objects per second
            'sample_size': len(times)
        }

    def benchmark_end_to_end_pipeline(self) -> Dict:
        """Benchmark complete end-to-end pipeline"""
        test_scenarios = [
            {'command': 'Go to kitchen and pick up cup', 'complexity': 'high'},
            {'command': 'Move forward 2 meters', 'complexity': 'low'},
            {'command': 'Navigate to charging station', 'complexity': 'medium'},
            {'command': 'Find red ball and grasp it', 'complexity': 'high'}
        ]

        times = []
        for scenario in test_scenarios:
            start_time = time.time()

            # Simulate complete pipeline execution
            result = self.simulate_complete_pipeline(scenario)

            processing_time = time.time() - start_time
            times.append(processing_time)

        return {
            'success': True,
            'average_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
            'throughput': len(test_scenarios) / sum(times),  # scenarios per second
            'sample_size': len(times)
        }

    # Simulation functions (would be replaced with actual implementations)
    def simulate_voice_processing(self, command):
        """Simulate voice processing"""
        time.sleep(0.1)  # Simulate processing time
        return {'success': True, 'intent': {'action': 'navigation', 'parameters': {}}}

    def simulate_llm_call(self, prompt):
        """Simulate LLM call"""
        time.sleep(0.5)  # Simulate API call time
        return {'success': True, 'plan': {'actions': []}}

    def simulate_perception_processing(self, image, num_objects):
        """Simulate perception processing"""
        time.sleep(0.05)  # Simulate processing time
        return {'success': True, 'objects': [{'name': f'object_{i}', 'confidence': 0.8} for i in range(num_objects)]}

    def simulate_navigation(self, route):
        """Simulate navigation"""
        time.sleep(0.2)  # Simulate planning time
        return {'success': True, 'path': [(0, 0), (route['goal'][0], route['goal'][1])]}

    def simulate_manipulation(self, obj):
        """Simulate manipulation"""
        time.sleep(0.3)  # Simulate planning and execution time
        return {'success': True, 'grasped': True}

    def simulate_complete_pipeline(self, scenario):
        """Simulate complete pipeline"""
        time.sleep(1.0)  # Simulate complete pipeline time
        return {'success': True, 'completed': True}

    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        if not self.test_results:
            return "No benchmark results available"

        report_lines = [
            "# VLA System Performance Benchmark Report",
            f"Generated: {time.ctime()}",
            ""
        ]

        for test_name, results in self.test_results.items():
            if results.get('success', False):
                report_lines.append(f"## {test_name.replace('_', ' ').title()}")
                report_lines.append(f"- Average Time: {results['average_time']:.3f}s")
                report_lines.append(f"- Min Time: {results['min_time']:.3f}s")
                report_lines.append(f"- Max Time: {results['max_time']:.3f}s")
                report_lines.append(f"- Std Dev: {results['std_deviation']:.3f}s")
                report_lines.append(f"- Throughput: {results['throughput']:.2f} ops/sec")
                report_lines.append(f"- Sample Size: {results['sample_size']}")
                report_lines.append("")
            else:
                report_lines.append(f"## {test_name.replace('_', ' ').title()}: FAILED")
                report_lines.append(f"- Error: {results.get('error', 'Unknown error')}")
                report_lines.append("")

        return "\n".join(report_lines)

    def check_performance_requirements(self, requirements: Dict) -> Dict:
        """Check if system meets performance requirements"""
        compliance_results = {}

        for req_name, req_value in requirements.items():
            if req_name in self.test_results:
                actual_value = self.test_results[req_name].get('average_time', float('inf'))

                # For time-based requirements, lower is better
                compliant = actual_value <= req_value.get('max', float('inf'))

                compliance_results[req_name] = {
                    'compliant': compliant,
                    'actual': actual_value,
                    'required': req_value,
                    'pass': compliant
                }

        return compliance_results
```

## Performance Optimization Best Practices

### 1. Profiling and Measurement

```python
import cProfile
import pstats
from functools import wraps
import io


def profile_function(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Save profile data
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions

        # Log profiling results
        profile_data = s.getvalue()
        print(f"Profiling results for {func.__name__}:")
        print(profile_data)

        return result

    return wrapper


class PerformanceProfiler:
    """Performance profiler for VLA system components"""
    def __init__(self, node):
        self.node = node
        self.profiles = {}

    def start_profiling(self, component_name):
        """Start profiling for component"""
        if component_name not in self.profiles:
            self.profiles[component_name] = cProfile.Profile()

        self.profiles[component_name].enable()

    def stop_profiling(self, component_name):
        """Stop profiling for component"""
        if component_name in self.profiles:
            self.profiles[component_name].disable()

            # Get stats
            s = io.StringIO()
            ps = pstats.Stats(self.profiles[component_name], stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)

            profile_data = s.getvalue()
            self.node.get_logger().info(f"Profile for {component_name}:\n{profile_data}")

    def get_profile_summary(self, component_name):
        """Get summary of profile data"""
        if component_name not in self.profiles:
            return None

        s = io.StringIO()
        ps = pstats.Stats(self.profiles[component_name], stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(5)  # Top 5 functions

        return s.getvalue()
```

### 2. Resource Management Best Practices

```python
class ResourceManager:
    """Resource manager for optimal resource utilization"""
    def __init__(self, node):
        self.node = node
        self.resource_limits = {
            'cpu': 0.8,  # 80% CPU limit
            'memory': 0.8,  # 80% memory limit
            'disk': 0.9,  # 90% disk limit
            'network': 100 * 1024 * 1024  # 100 MB/s network limit
        }
        self.current_allocation = {}
        self.reservation_system = ResourceReservationSystem()

    def allocate_resources(self, component, requirements):
        """Allocate resources to component"""
        # Check if sufficient resources are available
        if self.check_resource_availability(requirements):
            # Reserve resources
            reservation = self.reservation_system.reserve(component, requirements)

            if reservation['success']:
                self.current_allocation[component] = reservation
                return reservation
            else:
                return {'success': False, 'error': 'Resource reservation failed'}
        else:
            return {'success': False, 'error': 'Insufficient resources available'}

    def check_resource_availability(self, requirements):
        """Check if required resources are available"""
        import psutil

        # Check CPU availability
        if requirements.get('cpu', 0) > (1.0 - psutil.cpu_percent() / 100.0) * self.resource_limits['cpu']:
            return False

        # Check memory availability
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if requirements.get('memory_gb', 0) > available_memory * self.resource_limits['memory']:
            return False

        # Add other resource checks as needed

        return True

    def deallocate_resources(self, component):
        """Deallocate resources from component"""
        if component in self.current_allocation:
            self.reservation_system.release(self.current_allocation[component])
            del self.current_allocation[component]

    def optimize_resource_allocation(self):
        """Optimize resource allocation across components"""
        # Implementation to optimize resource allocation
        # This could involve adjusting priorities, releasing unused resources, etc.
        pass


class ResourceReservationSystem:
    """System for managing resource reservations"""
    def __init__(self):
        self.reservations = {}
        self.priority_queue = []

    def reserve(self, component, requirements):
        """Reserve resources for component"""
        reservation_id = f"{component}_{int(time.time())}"

        reservation = {
            'id': reservation_id,
            'component': component,
            'requirements': requirements,
            'allocated_at': time.time(),
            'expires_at': time.time() + 3600  # 1 hour default
        }

        self.reservations[reservation_id] = reservation
        return {'success': True, 'reservation': reservation}

    def release(self, reservation):
        """Release resource reservation"""
        if isinstance(reservation, dict):
            reservation_id = reservation.get('id')
        else:
            reservation_id = reservation

        if reservation_id in self.reservations:
            del self.reservations[reservation_id]
```

Performance optimization in VLA systems requires a multi-layered approach that addresses computational efficiency, resource management, real-time constraints, and system scalability. By implementing the strategies outlined in this chapter, VLA systems can achieve the responsiveness and reliability required for effective autonomous humanoid robot operation.