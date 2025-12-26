---
sidebar_position: 3
---

# Chapter 2: ROS 2 Control with Python Agents (rclpy)

## Learning Objectives

After completing this chapter, you will be able to:
- Explain the role of Python agents in robot control
- Create ROS 2 nodes using rclpy
- Publish and subscribe to topics for sensor and actuator data
- Use services for request-response robot behaviors
- Describe how to bridge AI decision logic to low-level ROS controllers

## The Role of Python Agents in Robot Control

Python agents play a crucial role in robot control by serving as the interface between high-level AI decision-making and low-level hardware control. In the context of ROS 2, Python agents are nodes that:

1. **Process Sensor Data**: Collect and interpret data from various sensors (cameras, lidars, IMUs, etc.)
2. **Implement AI Algorithms**: Execute path planning, decision-making, learning algorithms, and other AI functions
3. **Control Actuators**: Send commands to motors, grippers, and other actuators
4. **Coordinate Subsystems**: Manage communication between different robot subsystems
5. **Handle User Interaction**: Process user commands and provide feedback

Python is particularly well-suited for AI agents due to its rich ecosystem of machine learning and data processing libraries, rapid prototyping capabilities, and strong community support.

## Creating ROS 2 Nodes Using rclpy

rclpy is the Python client library for ROS 2. It provides the interface between Python programs and the ROS 2 middleware.

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        # Node initialization code here

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Key Components of a Node:

1. **Node Class**: Inherits from `rclpy.node.Node`
2. **Initialization**: Calls `super().__init__()` with a unique node name
3. **ROS 2 Context**: Initializes with `rclpy.init()` and spins with `rclpy.spin()`

## Publishing and Subscribing to Topics

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
```

## Using Services for Request-Response Robot Behaviors

Services provide synchronous request-response communication patterns, which are useful for operations that require confirmation or return results.

### Service Server Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
```

### Service Client Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Bridging AI Decision Logic to Low-Level ROS Controllers

The bridge between AI decision logic and low-level controllers is essential for autonomous robot behavior. This typically involves:

1. **Sensor Data Processing**: Converting raw sensor data into meaningful information for AI algorithms
2. **AI Decision Making**: Running path planning, behavior selection, or learning algorithms
3. **Command Generation**: Converting AI decisions into specific commands for low-level controllers
4. **Feedback Integration**: Using feedback from low-level controllers to adjust AI decisions

### Example: AI-Driven Navigation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class AINavigationNode(Node):
    def __init__(self):
        super().__init__('ai_navigation_node')

        # Subscribers for sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for AI decision making
        self.timer = self.create_timer(0.1, self.ai_decision_callback)

        # Internal state
        self.laser_data = None
        self.velocity_cmd = Twist()

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg.ranges

    def ai_decision_callback(self):
        """AI decision-making logic"""
        if self.laser_data is None:
            return

        # Simple obstacle avoidance AI
        min_distance = min(self.laser_data) if self.laser_data else float('inf')

        if min_distance < 1.0:  # Obstacle detected
            self.velocity_cmd.linear.x = 0.0
            self.velocity_cmd.angular.z = 0.5  # Turn
        else:
            self.velocity_cmd.linear.x = 0.5  # Move forward
            self.velocity_cmd.angular.z = 0.0

        # Publish command to low-level controller
        self.cmd_vel_pub.publish(self.velocity_cmd)

def main(args=None):
    rclpy.init(args=args)
    ai_nav_node = AINavigationNode()
    rclpy.spin(ai_nav_node)
    ai_nav_node.destroy_node()
    rclpy.shutdown()
```

## Summary

In this chapter, you've learned how to create Python agents for ROS 2 control using rclpy. You now understand how to create nodes, publish and subscribe to topics, use services for request-response communication, and bridge AI decision logic to low-level controllers.

## Exercises

1. Create a Python node that publishes sensor data to a topic and another node that subscribes to that topic.
2. Implement a service that performs a simple calculation based on robot state.
3. Design a simple AI agent that makes navigation decisions based on sensor input.

## References

- [rclpy API Documentation](https://docs.ros.org/en/rolling/p/rclpy/)
- [ROS 2 Tutorials](https://docs.ros.org/en/rolling/Tutorials.html)
- [Python Client Library for ROS 2](https://github.com/ros2/rclpy)