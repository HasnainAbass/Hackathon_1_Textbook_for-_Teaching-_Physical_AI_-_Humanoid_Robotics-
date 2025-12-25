# ROS 2 Integration: Perception and Control Systems

## Introduction

This chapter focuses on the integration of perception and control systems within the ROS 2 framework for Vision-Language-Action (VLA) systems. The integration of perception (sensing and understanding the environment) with control (executing robot actions) is crucial for creating autonomous humanoid systems that can effectively interact with their environment based on voice commands.

## ROS 2 Architecture for Perception-Control Integration

### System Overview

The perception-control integration in ROS 2 follows a distributed architecture with multiple nodes communicating through topics, services, and actions:

```
Perception Nodes → Data Processing → Control Nodes → Robot Hardware
       ↑                                           ↓
       └─────────── ROS 2 Communication ────────────┘
```

### Key Components

1. **Perception Nodes**: Process sensor data (cameras, LiDAR, etc.) to understand the environment
2. **Data Processing Nodes**: Integrate and interpret perception data
3. **Control Nodes**: Generate commands to control robot hardware
4. **Action Servers**: Handle long-running tasks like navigation and manipulation
5. **Parameter Servers**: Manage configuration and calibration parameters

## Perception System Implementation

### Camera Perception Node

```python
#!/usr/bin/env python3
"""
Camera Perception Node for VLA Systems
Processes camera data to detect objects and understand environment
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import json


class CameraPerceptionNode(Node):
    def __init__(self):
        super().__init__('camera_perception_node')

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'camera/detections',
            10
        )

        self.processed_image_pub = self.create_publisher(
            Image,
            'camera/processed_image',
            10
        )

        # Object detection configuration
        self.detection_config = self.load_detection_config()

        # Processing parameters
        self.processing_rate = 10  # Hz
        self.processing_timer = self.create_timer(1.0/self.processing_rate, self.process_timer_callback)

        # Internal state
        self.latest_image = None
        self.latest_detections = Detection2DArray()

        self.get_logger().info('Camera Perception Node initialized')

    def load_detection_config(self):
        """Load object detection configuration"""
        return {
            'object_classes': ['cup', 'ball', 'book', 'bottle', 'person'],
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'image_size': (640, 480)
        }

    def image_callback(self, msg):
        """Receive and store latest image"""
        try:
            # Store the latest image for processing
            self.latest_image = msg
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def process_timer_callback(self):
        """Process latest image at regular intervals"""
        if self.latest_image is not None:
            try:
                # Convert ROS Image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')

                # Perform object detection
                detections = self.detect_objects(cv_image)

                # Publish detections
                self.publish_detections(detections)

                # Publish processed image for visualization
                processed_cv_image = self.annotate_image(cv_image, detections)
                processed_image_msg = self.bridge.cv2_to_imgmsg(processed_cv_image, 'bgr8')
                processed_image_msg.header = self.latest_image.header
                self.processed_image_pub.publish(processed_image_msg)

                # Update internal state
                self.latest_detections = detections

            except Exception as e:
                self.get_logger().error(f'Error in processing: {e}')

    def detect_objects(self, cv_image):
        """Detect objects in the image (simplified implementation)"""
        # In a real system, this would use YOLO, SSD, or similar
        # For simulation, return mock detections
        height, width = cv_image.shape[:2]

        # Mock detection results
        mock_detections = [
            {
                'class': 'cup',
                'confidence': 0.8,
                'bbox': [width//2 - 25, height//2 - 25, 50, 50],  # [x, y, w, h]
                'center': [width//2, height//2]
            },
            {
                'class': 'ball',
                'confidence': 0.7,
                'bbox': [width//3, height//3, 40, 40],
                'center': [width//3 + 20, height//3 + 20]
            }
        ]

        # Create Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header = Header()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'

        for detection in mock_detections:
            if detection['confidence'] >= self.detection_config['confidence_threshold']:
                vision_detection = Detection2D()
                vision_detection.header = detection_array.header

                # Set bounding box
                vision_detection.bbox.center.x = float(detection['center'][0])
                vision_detection.bbox.center.y = float(detection['center'][1])
                vision_detection.bbox.size_x = float(detection['bbox'][2])
                vision_detection.bbox.size_y = float(detection['bbox'][3])

                # Set object hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = detection['class']
                hypothesis.score = detection['confidence']

                vision_detection.results.append(hypothesis)
                detection_array.detections.append(vision_detection)

        return detection_array

    def annotate_image(self, cv_image, detections):
        """Annotate image with detection results"""
        annotated_image = cv_image.copy()

        for detection in detections.detections:
            if detection.results:
                # Get the most confident result
                result = detection.results[0]
                class_name = result.id
                confidence = result.score

                # Draw bounding box
                center_x = int(detection.bbox.center.x)
                center_y = int(detection.bbox.center.y)
                size_x = int(detection.bbox.size_x)
                size_y = int(detection.bbox.size_y)

                top_left = (center_x - size_x//2, center_y - size_y//2)
                bottom_right = (center_x + size_x//2, center_y + size_y//2)

                cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(
                    annotated_image,
                    f'{class_name}: {confidence:.2f}',
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        return annotated_image

    def publish_detections(self, detections):
        """Publish detection results"""
        self.detection_pub.publish(detections)

    def get_latest_detections(self):
        """Get the latest detection results"""
        return self.latest_detections


def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down camera perception node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### LiDAR Perception Node

```python
#!/usr/bin/env python3
"""
LiDAR Perception Node for VLA Systems
Processes LiDAR data for obstacle detection and navigation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from std_msgs.msg import ColorRGBA


class LIDARPerceptionNode(Node):
    def __init__(self):
        super().__init__('lidar_perception_node')

        # Publishers and subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.obstacle_pub = self.create_publisher(
            MarkerArray,
            'obstacle_markers',
            10
        )

        self.processing_rate = 10  # Hz
        self.processing_timer = self.create_timer(1.0/self.processing_rate, self.process_timer_callback)

        # Internal state
        self.latest_scan = None
        self.obstacles = []

        self.get_logger().info('LiDAR Perception Node initialized')

    def scan_callback(self, msg):
        """Receive and store latest scan data"""
        self.latest_scan = msg

    def process_timer_callback(self):
        """Process latest scan data"""
        if self.latest_scan is not None:
            try:
                # Process scan data to detect obstacles
                self.obstacles = self.detect_obstacles(self.latest_scan)

                # Publish obstacle markers for visualization
                self.publish_obstacle_markers()

            except Exception as e:
                self.get_logger().error(f'Error in LiDAR processing: {e}')

    def detect_obstacles(self, scan_msg):
        """Detect obstacles from LiDAR scan data"""
        obstacles = []

        # Convert scan ranges to Cartesian coordinates
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)) and range_val < 2.0:  # Within 2m
                angle = angle_min + i * angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                # Simple clustering to group nearby points
                obstacles.append({'x': x, 'y': y, 'range': range_val, 'angle': angle})

        return obstacles

    def publish_obstacle_markers(self):
        """Publish obstacle markers for visualization"""
        marker_array = MarkerArray()

        for i, obstacle in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = 'laser_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position
            marker.pose.position.x = obstacle['x']
            marker.pose.position.y = obstacle['y']
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            # Scale
            marker.scale.x = 0.2  # 20cm diameter
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            # Color
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8  # 80% alpha

            marker_array.markers.append(marker)

        self.obstacle_pub.publish(marker_array)
```

## Control System Implementation

### Navigation Control Node

```python
#!/usr/bin/env python3
"""
Navigation Control Node for VLA Systems
Controls robot navigation based on perception data and goals
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
import math


class NavigationControlNode(Node):
    def __init__(self):
        super().__init__('navigation_control_node')

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, 'navigation_goal', self.goal_callback, 10)

        # Publishers for visualization
        self.goal_marker_pub = self.create_publisher(Marker, 'navigation_goal_marker', 10)

        # Internal state
        self.current_pose = None
        self.current_twist = None
        self.latest_scan = None
        self.safe_to_navigate = True

        self.get_logger().info('Navigation Control Node initialized')

    def odom_callback(self, msg):
        """Update current pose and twist from odometry"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        """Update latest scan and check for obstacles"""
        self.latest_scan = msg

        # Check if path is clear
        self.safe_to_navigate = self.is_path_clear(msg)

    def goal_callback(self, msg):
        """Receive navigation goal and execute"""
        if self.safe_to_navigate:
            self.execute_navigation_goal(msg)
        else:
            self.get_logger().warn('Path not clear, aborting navigation')

    def is_path_clear(self, scan_msg):
        """Check if path ahead is clear of obstacles"""
        if scan_msg is None:
            return True  # If no scan data, assume path is clear

        # Check forward sector (e.g., 60 degrees ahead)
        forward_start = len(scan_msg.ranges) // 2 - 30  # 30 degrees from center
        forward_end = len(scan_msg.ranges) // 2 + 30    # 30 degrees from center

        min_range = float('inf')
        for i in range(forward_start, forward_end):
            if 0 <= i < len(scan_msg.ranges):
                range_val = scan_msg.ranges[i]
                if not (math.isnan(range_val) or math.isinf(range_val)):
                    min_range = min(min_range, range_val)

        # If minimum range is less than safe distance, path is not clear
        return min_range > 0.5  # 50cm safety margin

    def execute_navigation_goal(self, goal_msg):
        """Execute navigation goal using navigation2"""
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return

        # Send navigation goal
        goal = NavigateToPose.Goal()
        goal.pose = goal_msg

        # Send async goal
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.navigation_result_callback)

        # Publish goal marker for visualization
        self.publish_goal_marker(goal_msg)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return

        self.get_logger().info('Navigation goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        """Handle navigation completion"""
        result = future.result().result
        self.get_logger().info(f'Navigation completed with result: {result}')

    def publish_goal_marker(self, pose_stamped):
        """Publish visualization marker for navigation goal"""
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = 'navigation_goals'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Position and orientation
        marker.pose = pose_stamped.pose

        # Scale
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.1  # Arrow width
        marker.scale.z = 0.1  # Arrow height

        # Color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.goal_marker_pub.publish(marker)

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
```

### Manipulation Control Node

```python
#!/usr/bin/env python3
"""
Manipulation Control Node for VLA Systems
Controls robot manipulation based on perception data and goals
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from vision_msgs.msg import Detection2DArray
from builtin_interfaces.msg import Duration
import math


class ManipulationControlNode(Node):
    def __init__(self):
        super().__init__('manipulation_control_node')

        # Publishers and subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            'camera/detections',
            self.detection_callback,
            10
        )

        self.joint_command_pub = self.create_publisher(
            JointTrajectory,
            'joint_trajectory_controller/joint_trajectory',
            10
        )

        self.manipulation_goal_sub = self.create_subscription(
            String,
            'manipulation_goal',
            self.manipulation_goal_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Internal state
        self.current_joint_states = {}
        self.latest_detections = None
        self.robot_arm_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

        self.get_logger().info('Manipulation Control Node initialized')

    def detection_callback(self, msg):
        """Update latest detections"""
        self.latest_detections = msg

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if name in self.robot_arm_joints:
                self.current_joint_states[name] = msg.position[i]

    def manipulation_goal_callback(self, msg):
        """Handle manipulation goal"""
        goal = msg.data.lower().strip()

        if 'grasp' in goal or 'pick' in goal:
            # Find object to grasp
            target_object = self.find_target_object(goal)
            if target_object:
                self.execute_grasp(target_object)
            else:
                self.get_logger().warn(f'No suitable object found for: {goal}')
        elif 'place' in goal or 'drop' in goal:
            self.execute_place()
        else:
            self.get_logger().warn(f'Unknown manipulation goal: {goal}')

    def find_target_object(self, goal):
        """Find target object based on goal description"""
        if not self.latest_detections:
            return None

        # Extract object type from goal
        object_type = None
        if 'cup' in goal:
            object_type = 'cup'
        elif 'ball' in goal:
            object_type = 'ball'
        elif 'book' in goal:
            object_type = 'book'

        if not object_type:
            return None

        # Find the closest object of the specified type
        closest_object = None
        min_distance = float('inf')

        for detection in self.latest_detections.detections:
            if detection.results and detection.results[0].id == object_type:
                # Calculate distance from center of image (simplified)
                distance = math.sqrt(
                    (detection.bbox.center.x - 320)**2 +  # Assuming 640x480 image
                    (detection.bbox.center.y - 240)**2
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_object = detection

        return closest_object

    def execute_grasp(self, target_object):
        """Execute grasp action for target object"""
        self.get_logger().info(f'Attempting to grasp {target_object.results[0].id}')

        # Calculate grasp pose based on object position
        grasp_pose = self.calculate_grasp_pose(target_object)

        # Plan and execute grasp trajectory
        grasp_trajectory = self.plan_grasp_trajectory(grasp_pose)

        # Publish trajectory
        self.joint_command_pub.publish(grasp_trajectory)

    def calculate_grasp_pose(self, target_object):
        """Calculate appropriate grasp pose for target object"""
        # Simplified grasp pose calculation
        grasp_pose = Pose()
        grasp_pose.position.x = target_object.bbox.center.x / 100.0  # Scale to meters
        grasp_pose.position.y = target_object.bbox.center.y / 100.0
        grasp_pose.position.z = 0.1  # 10cm above object

        # Simple orientation (gripper facing down)
        grasp_pose.orientation.w = 1.0  # No rotation
        grasp_pose.orientation.x = 0.0
        grasp_pose.orientation.y = 0.0
        grasp_pose.orientation.z = 0.0

        return grasp_pose

    def plan_grasp_trajectory(self, grasp_pose):
        """Plan trajectory for grasping action"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.robot_arm_joints

        # Create trajectory points
        points = []

        # Pre-grasp position (simplified)
        pre_grasp_point = JointTrajectoryPoint()
        pre_grasp_point.positions = [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]  # Example joint positions
        pre_grasp_point.velocities = [0.0] * len(self.robot_arm_joints)
        pre_grasp_point.time_from_start = Duration(sec=2, nanosec=0)
        points.append(pre_grasp_point)

        # Grasp position
        grasp_point = JointTrajectoryPoint()
        grasp_point.positions = [0.2, -0.3, 0.7, 0.1, 0.4, 0.1]  # Example joint positions
        grasp_point.velocities = [0.0] * len(self.robot_arm_joints)
        grasp_point.time_from_start = Duration(sec=4, nanosec=0)
        points.append(grasp_point)

        # Close gripper (simplified as joint movement)
        close_point = JointTrajectoryPoint()
        close_point.positions = [0.2, -0.3, 0.7, 0.1, 0.4, 0.3]  # Gripper closed
        close_point.velocities = [0.0] * len(self.robot_arm_joints)
        close_point.time_from_start = Duration(sec=5, nanosec=0)
        points.append(close_point)

        trajectory.points = points
        return trajectory

    def execute_place(self):
        """Execute place action"""
        self.get_logger().info('Executing place action')

        # Plan and execute place trajectory
        place_trajectory = self.plan_place_trajectory()

        # Publish trajectory
        self.joint_command_pub.publish(place_trajectory)

    def plan_place_trajectory(self):
        """Plan trajectory for placing action"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.robot_arm_joints

        # Create trajectory points
        points = []

        # Lift position (simplified)
        lift_point = JointTrajectoryPoint()
        lift_point.positions = [0.2, -0.3, 0.8, 0.1, 0.4, 0.3]  # Lifted position
        lift_point.velocities = [0.0] * len(self.robot_arm_joints)
        lift_point.time_from_start = Duration(sec=2, nanosec=0)
        points.append(lift_point)

        # Place position
        place_point = JointTrajectoryPoint()
        place_point.positions = [0.5, -0.2, 0.6, 0.2, 0.3, 0.3]  # Place position
        place_point.velocities = [0.0] * len(self.robot_arm_joints)
        place_point.time_from_start = Duration(sec=4, nanosec=0)
        points.append(place_point)

        # Open gripper
        open_point = JointTrajectoryPoint()
        open_point.positions = [0.5, -0.2, 0.6, 0.2, 0.3, 0.1]  # Gripper open
        open_point.velocities = [0.0] * len(self.robot_arm_joints)
        open_point.time_from_start = Duration(sec=5, nanosec=0)
        points.append(open_point)

        trajectory.points = points
        return trajectory
```

## Integration Patterns

### Perception-Control Coordination

```python
class PerceptionControlCoordinator:
    def __init__(self, node):
        self.node = node
        self.perception_nodes = []
        self.control_nodes = []
        self.integration_rules = self.define_integration_rules()

    def define_integration_rules(self):
        """Define rules for perception-control integration"""
        return {
            'navigation': {
                'perception_requirements': ['obstacle_detection', 'mapping'],
                'control_requirements': ['navigation_control'],
                'integration_logic': self.integrate_navigation_perception
            },
            'manipulation': {
                'perception_requirements': ['object_detection', 'pose_estimation'],
                'control_requirements': ['manipulation_control'],
                'integration_logic': self.integrate_manipulation_perception
            }
        }

    def integrate_navigation_perception(self, perception_data, control_context):
        """Integrate navigation perception with control"""
        # Update navigation map with new perception data
        updated_map = self.update_navigation_map(
            control_context.get('current_map', {}),
            perception_data.get('obstacles', [])
        )

        # Plan path considering updated map
        path = self.plan_path_with_obstacles(
            control_context['start_pose'],
            control_context['goal_pose'],
            updated_map
        )

        return {
            'updated_path': path,
            'safe_navigation': len(path) > 0,
            'obstacle_avoidance': True
        }

    def integrate_manipulation_perception(self, perception_data, control_context):
        """Integrate manipulation perception with control"""
        # Update object positions with new perception data
        updated_objects = self.update_object_positions(
            control_context.get('known_objects', {}),
            perception_data.get('detections', [])
        )

        # Calculate grasp poses for detected objects
        grasp_poses = self.calculate_grasp_poses(updated_objects)

        return {
            'grasp_poses': grasp_poses,
            'target_objects': list(updated_objects.keys()),
            'manipulation_feasible': len(grasp_poses) > 0
        }

    def update_navigation_map(self, current_map, new_obstacles):
        """Update navigation map with new obstacle information"""
        # Merge new obstacles with existing map
        updated_map = current_map.copy()
        updated_map['obstacles'] = current_map.get('obstacles', []) + new_obstacles
        return updated_map

    def update_object_positions(self, known_objects, new_detections):
        """Update known object positions with new detections"""
        updated_objects = known_objects.copy()

        for detection in new_detections:
            if detection.results:
                obj_id = detection.results[0].id
                updated_objects[obj_id] = {
                    'position': {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y,
                        'z': detection.bbox.center.z if hasattr(detection.bbox.center, 'z') else 0
                    },
                    'confidence': detection.results[0].score,
                    'timestamp': self.node.get_clock().now().to_msg()
                }

        return updated_objects

    def plan_path_with_obstacles(self, start_pose, goal_pose, map_with_obstacles):
        """Plan path considering obstacles in the map"""
        # In a real system, this would use path planning algorithms like A* or RRT
        # For simulation, return a simple path
        return [
            {'x': start_pose.position.x, 'y': start_pose.position.y},
            {'x': goal_pose.position.x, 'y': goal_pose.position.y}
        ]

    def calculate_grasp_poses(self, objects):
        """Calculate appropriate grasp poses for objects"""
        grasp_poses = []

        for obj_id, obj_data in objects.items():
            # Calculate grasp pose based on object properties
            grasp_pose = {
                'object_id': obj_id,
                'position': obj_data['position'],
                'approach_direction': 'top',  # Default approach
                'grasp_type': 'parallel'      # Default grasp type
            }
            grasp_poses.append(grasp_pose)

        return grasp_poses
```

## Safety and Validation

### Safety Validation Layer

```python
class SafetyValidator:
    def __init__(self, node):
        self.node = node
        self.safety_limits = {
            'max_linear_velocity': 0.5,
            'max_angular_velocity': 0.5,
            'max_manipulation_force': 30.0,
            'min_human_distance': 1.0,
            'max_payload': 5.0
        }

    def validate_perception_control_action(self, action, context):
        """Validate perception-control action for safety"""
        action_type = action.get('type', 'unknown')
        parameters = action.get('parameters', {})

        if action_type == 'navigation':
            return self.validate_navigation_safety(parameters, context)
        elif action_type == 'manipulation':
            return self.validate_manipulation_safety(parameters, context)
        elif action_type == 'perception':
            return self.validate_perception_safety(parameters, context)
        else:
            return self.validate_generic_action_safety(action, context)

    def validate_navigation_safety(self, parameters, context):
        """Validate navigation action for safety"""
        issues = []

        # Check destination safety
        destination = parameters.get('location', {})
        if 'x' in destination and 'y' in destination:
            # Check if destination is too close to humans
            humans = context.get('detected_humans', [])
            for human in humans:
                distance = self.calculate_distance(destination, human['position'])
                if distance < self.safety_limits['min_human_distance']:
                    issues.append(f'Destination too close to human: {distance:.2f}m')

            # Check if destination is in safe area
            if self.is_in_dangerous_area(destination):
                issues.append('Destination is in dangerous area')

        # Check path safety
        obstacles = context.get('obstacles', [])
        if len(obstacles) > 10:  # Too many obstacles
            issues.append('Too many obstacles in navigation path')

        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'risk_level': self.assess_risk_level(issues)
        }

    def validate_manipulation_safety(self, parameters, context):
        """Validate manipulation action for safety"""
        issues = []

        # Check payload safety
        payload = parameters.get('payload_weight', 0)
        if payload > self.safety_limits['max_payload']:
            issues.append(f'Payload {payload}kg exceeds limit {self.safety_limits["max_payload"]}kg')

        # Check object safety
        target_object = parameters.get('target_object', {})
        if target_object.get('is_fragile', False):
            issues.append('Target object is fragile, special handling required')

        # Check force limits
        grasp_force = parameters.get('grasp_force', 0)
        if grasp_force > self.safety_limits['max_manipulation_force']:
            issues.append(f'Grasp force {grasp_force}N exceeds limit {self.safety_limits["max_manipulation_force"]}N')

        # Check human safety
        humans = context.get('detected_humans', [])
        for human in humans:
            if self.is_in_workspace(target_object.get('position', {}), human['position']):
                issues.append('Manipulation workspace contains human')

        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'risk_level': self.assess_risk_level(issues)
        }

    def validate_perception_safety(self, parameters, context):
        """Validate perception action for safety"""
        # Perception actions are generally safe
        # But check if they involve potentially dangerous sensors
        sensor_type = parameters.get('sensor_type', 'camera')
        if sensor_type == 'laser_rangefinder' and parameters.get('power_level', 'low') == 'high':
            return {
                'safe': False,
                'issues': ['High-power laser rangefinder may be unsafe around humans'],
                'risk_level': 'high'
            }

        return {
            'safe': True,
            'issues': [],
            'risk_level': 'low'
        }

    def validate_generic_action_safety(self, action, context):
        """Validate generic action for safety"""
        return {
            'safe': True,
            'issues': [],
            'risk_level': 'low'
        }

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        import math
        return math.sqrt(
            (pos1.get('x', 0) - pos2.get('x', 0))**2 +
            (pos1.get('y', 0) - pos2.get('y', 0))**2 +
            (pos1.get('z', 0) - pos2.get('z', 0))**2
        )

    def is_in_dangerous_area(self, position):
        """Check if position is in dangerous area"""
        # In a real system, this would check against a map of dangerous areas
        # For simulation, return False
        return False

    def is_in_workspace(self, obj_position, human_position):
        """Check if object position is in human workspace"""
        if not obj_position or not human_position:
            return False

        distance = self.calculate_distance(obj_position, human_position)
        return distance < 1.0  # Within 1 meter

    def assess_risk_level(self, issues):
        """Assess risk level based on issues"""
        if not issues:
            return 'low'
        elif len(issues) == 1:
            return 'medium'
        else:
            return 'high'
```

## Performance Optimization

### Efficient Perception-Control Communication

```python
class OptimizedPerceptionControlInterface:
    def __init__(self, node):
        self.node = node
        self.data_cache = {}
        self.compression_enabled = True
        self.throttling_rate = 10  # Hz

    def compress_perception_data(self, data):
        """Compress perception data for efficient transmission"""
        if not self.compression_enabled:
            return data

        # Simple compression - remove low-confidence detections
        compressed_data = []
        for detection in data:
            if detection.get('confidence', 0) > 0.7:  # Only keep high-confidence
                compressed_data.append(detection)

        return compressed_data

    def throttle_data_transmission(self, data, topic_name):
        """Throttle data transmission to reduce network load"""
        import time

        current_time = time.time()
        if topic_name not in self.data_cache:
            self.data_cache[topic_name] = {'last_sent': 0, 'data': None}

        time_since_last = current_time - self.data_cache[topic_name]['last_sent']
        min_interval = 1.0 / self.throttling_rate

        if time_since_last >= min_interval:
            self.data_cache[topic_name]['last_sent'] = current_time
            self.data_cache[topic_name]['data'] = data
            return True, data
        else:
            # Return cached data instead
            cached_data = self.data_cache[topic_name]['data']
            return False, cached_data  # Throttled, don't send new data
```

The integration of perception and control systems in ROS 2 enables sophisticated autonomous behaviors in VLA systems. Through proper message passing, action coordination, and safety validation, we create robust systems that can perceive their environment and execute complex tasks safely and effectively.