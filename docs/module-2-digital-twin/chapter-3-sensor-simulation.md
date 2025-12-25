---
sidebar_position: 5
---

# Chapter 3: Sensor Simulation for Humanoid Robots

## Learning Objectives

After completing this chapter, you will be able to:
- Simulate sensors like LiDAR, depth cameras, and IMUs in digital twins
- Publish simulated sensor data to ROS 2 for perception testing
- Use synthetic sensor data for testing perception algorithms

## Introduction to Sensor Simulation

Sensor simulation is a critical component of digital twin technology for robotics. It enables the generation of synthetic sensor data that mimics real-world sensors, allowing for:
- Testing perception algorithms without physical hardware
- Training machine learning models with diverse scenarios
- Validating robot behavior under various environmental conditions
- Accelerating development cycles through rapid iteration

For humanoid robots, sensor simulation is particularly important as these robots often incorporate multiple sensor types to achieve human-like perception capabilities.

## Simulating LiDAR Sensors

### LiDAR Principles

LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time it takes for the light to return after reflecting off objects. This provides precise distance measurements for 3D mapping and navigation.

### LiDAR Simulation in Gazebo

Gazebo provides realistic LiDAR simulation through its sensor plugins:

```xml
<!-- LiDAR sensor definition in URDF/SDF -->
<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topicName>/laser_scan</topicName>
      <frameName>lidar_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Simulation Parameters

Key parameters for realistic LiDAR simulation:
- **Range Resolution**: Accuracy of distance measurements
- **Angular Resolution**: Precision of angle measurements
- **Field of View**: Horizontal and vertical coverage
- **Update Rate**: Frequency of sensor readings
- **Noise Modeling**: Addition of realistic measurement noise

### Unity LiDAR Simulation

In Unity, LiDAR simulation can be implemented using raycasting:

```csharp
// Example Unity LiDAR simulation
using UnityEngine;
using System.Collections.Generic;

public class UnityLidarSimulation : MonoBehaviour
{
    public int numRays = 720;
    public float fov = 180f;
    public float maxRange = 30f;
    public string topicName = "/laser_scan";

    private List<float> ranges;

    void Start()
    {
        ranges = new List<float>(new float[numRays]);
    }

    void Update()
    {
        SimulateLidar();
    }

    void SimulateLidar()
    {
        float angleStep = fov / numRays;

        for (int i = 0; i < numRays; i++)
        {
            float angle = transform.eulerAngles.y + (i * angleStep) - (fov / 2f);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = maxRange;
            }
        }

        // Publish to ROS topic (simplified)
        PublishLidarData();
    }

    void PublishLidarData()
    {
        // Implementation to send data to ROS
        // This would typically use the ROS-TCP-Connector
    }
}
```

## Simulating Depth Cameras

### Depth Camera Principles

Depth cameras provide both visual and depth information, making them valuable for 3D scene understanding, object recognition, and navigation. They typically provide RGB-D data (color + depth).

### Depth Camera Simulation in Gazebo

Gazebo supports realistic depth camera simulation:

```xml
<!-- Depth camera definition in URDF/SDF -->
<gazebo reference="camera_link">
  <sensor type="depth" name="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <cameraName>camera</cameraName>
      <imageTopicName>/rgb/image_raw</imageTopicName>
      <depthImageTopicName>/depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>/depth/points</pointCloudTopicName>
      <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
      <frameName>camera_depth_optical_frame</frameName>
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera Simulation Parameters

Key parameters for realistic depth camera simulation:
- **Resolution**: Image width and height in pixels
- **Field of View**: Horizontal and vertical viewing angles
- **Depth Range**: Minimum and maximum measurable distances
- **Frame Rate**: Number of frames per second
- **Noise Models**: Realistic noise for depth measurements
- **Distortion**: Lens distortion modeling

### Unity Depth Camera Simulation

In Unity, depth cameras can be simulated using multiple render textures:

```csharp
// Example Unity depth camera simulation
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class UnityDepthCamera : MonoBehaviour
{
    [Header("Depth Camera Settings")]
    public float minDepth = 0.1f;
    public float maxDepth = 10f;
    public Shader depthShader;

    private Camera cam;
    private RenderTexture depthTexture;

    void Start()
    {
        cam = GetComponent<Camera>();
        CreateDepthTexture();
    }

    void CreateDepthTexture()
    {
        depthTexture = new RenderTexture(640, 480, 24);
        depthTexture.format = RenderTextureFormat.RFloat;
        cam.SetTargetBuffers(depthTexture.colorBuffer, depthTexture.depthBuffer);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Apply depth shader and process depth data
        if (depthShader != null)
        {
            Graphics.Blit(source, destination, depthShader);
        }
        else
        {
            Graphics.Blit(source, destination);
        }

        // Process depth data and publish to ROS
        ProcessDepthData();
    }

    void ProcessDepthData()
    {
        // Extract depth information from texture
        // Convert to ROS sensor_msgs/PointCloud2 format
        // Publish to ROS topic
    }
}
```

## Simulating IMUs

### IMU Principles

Inertial Measurement Units (IMUs) measure linear acceleration and angular velocity. They typically include:
- Accelerometers: Measure linear acceleration
- Gyroscopes: Measure angular velocity
- Sometimes magnetometers: Measure magnetic field for orientation

### IMU Simulation in Gazebo

Gazebo provides realistic IMU simulation:

```xml
<!-- IMU sensor definition in URDF/SDF -->
<gazebo reference="imu_link">
  <sensor type="imu" name="imu_sensor">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topicName>/imu/data</topicName>
      <bodyName>imu_link</bodyName>
      <frameName>imu_link</frameName>
      <updateRateHZ>100.0</updateRateHZ>
      <gaussianNoise>0.0</gaussianNoise>
      <xyzOffset>0 0 0</xyzOffset>
      <rpyOffset>0 0 0</rpyOffset>
      <serviceName>/default_imu</serviceName>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation Parameters

Key parameters for realistic IMU simulation:
- **Update Rate**: Frequency of IMU readings
- **Noise Models**: Realistic noise for accelerometer and gyroscope readings
- **Bias Drift**: Slow drift in sensor readings over time
- **Scale Factor Error**: Deviations from ideal sensor response
- **Cross-Axis Sensitivity**: Interaction between different measurement axes

### Unity IMU Simulation

In Unity, IMU simulation can be achieved through physics calculations:

```csharp
// Example Unity IMU simulation
using UnityEngine;

public class UnityIMUSimulation : MonoBehaviour
{
    [Header("IMU Noise Parameters")]
    public float accelerometerNoise = 0.017f; // stddev in m/s^2
    public float gyroscopeNoise = 0.0002f;    // stddev in rad/s
    public float magnetometerNoise = 0.1f;    // stddev in microtesla

    private Rigidbody rb;
    private float lastUpdateTime;
    private Vector3 lastAngularVelocity;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        lastUpdateTime = Time.time;
    }

    void Update()
    {
        PublishIMUData();
    }

    void PublishIMUData()
    {
        // Calculate linear acceleration (remove gravity)
        Vector3 linearAcc = rb.velocity - Physics.gravity;

        // Calculate angular velocity
        Vector3 angularVel = rb.angularVelocity;

        // Add noise to measurements
        Vector3 noisyAcc = linearAcc + AddNoise(accelerometerNoise, 3);
        Vector3 noisyGyro = angularVel + AddNoise(gyroscopeNoise, 3);

        // Publish to ROS topic (simplified)
        PublishToROS(noisyAcc, noisyGyro);
    }

    Vector3 AddNoise(float noiseLevel, int dimensions)
    {
        Vector3 noise = Vector3.zero;
        for (int i = 0; i < dimensions; i++)
        {
            noise[i] = Random.Range(-noiseLevel, noiseLevel);
        }
        return noise;
    }

    void PublishToROS(Vector3 linearAcc, Vector3 angularVel)
    {
        // Implementation to send IMU data to ROS
        // This would typically use the ROS-TCP-Connector
    }
}
```

## Publishing Simulated Sensor Data to ROS 2

### ROS 2 Sensor Message Types

Common ROS 2 message types for sensor data:
- `sensor_msgs/LaserScan`: For LiDAR data
- `sensor_msgs/Image`: For camera images
- `sensor_msgs/PointCloud2`: For 3D point cloud data
- `sensor_msgs/Imu`: For IMU data
- `sensor_msgs/Range`: For sonar and other range sensors

### Publisher Implementation

```python
# Example ROS 2 publisher for simulated sensor data
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
import numpy as np

class SimulatedSensorPublisher(Node):
    def __init__(self):
        super().__init__('simulated_sensor_publisher')

        # Publishers for different sensor types
        self.lidar_pub = self.create_publisher(LaserScan, '/lidar_scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        # Timer for publishing at specific rate
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

    def publish_sensor_data(self):
        # Create and publish LiDAR data
        lidar_msg = LaserScan()
        lidar_msg.header.stamp = self.get_clock().now().to_msg()
        lidar_msg.header.frame_id = 'lidar_frame'
        lidar_msg.angle_min = -np.pi/2
        lidar_msg.angle_max = np.pi/2
        lidar_msg.angle_increment = np.pi/360  # 0.5 degree resolution
        lidar_msg.range_min = 0.1
        lidar_msg.range_max = 30.0
        lidar_msg.ranges = np.random.uniform(0.1, 30.0, 361).tolist()

        self.lidar_pub.publish(lidar_msg)

        # Similar implementation for IMU and camera data
        # ... (simplified for brevity)

def main(args=None):
    rclpy.init(args=args)
    publisher = SimulatedSensorPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Using Synthetic Sensor Data for Perception Testing

### Perception Pipeline Testing

Synthetic sensor data enables comprehensive testing of perception pipelines:

1. **Object Detection**: Testing detection algorithms with diverse scenarios
2. **SLAM**: Simultaneous localization and mapping with known ground truth
3. **Path Planning**: Testing navigation algorithms in complex environments
4. **Human Detection**: Testing social robotics applications

### Data Generation Strategies

1. **Domain Randomization**: Varying environmental conditions to improve robustness
2. **Synthetic-to-Real Transfer**: Training on synthetic data and testing on real data
3. **Active Learning**: Using simulation to identify critical training scenarios
4. **Edge Case Testing**: Creating rare but critical scenarios for safety validation

### Quality Assurance

To ensure synthetic data quality:
- Compare synthetic and real sensor characteristics
- Validate that synthetic data covers real-world distributions
- Test perception algorithms on both synthetic and real data
- Monitor for simulation artifacts that don't exist in reality

## Best Practices for Sensor Simulation

1. **Realistic Noise Models**: Include appropriate noise models for each sensor type
2. **Calibration Validation**: Ensure simulated sensors match real sensor calibration
3. **Cross-Sensor Validation**: Verify consistency between different sensor types
4. **Performance Optimization**: Balance simulation quality with computational requirements
5. **Ground Truth Access**: Maintain access to ground truth for validation

## Summary

In this chapter, you've learned about simulating various sensor types for humanoid robots, including LiDAR, depth cameras, and IMUs. You now understand how to publish this simulated sensor data to ROS 2 and use it for perception testing.

## Exercises

1. Implement a simple LiDAR simulation in either Gazebo or Unity
2. Create a perception pipeline that processes synthetic sensor data
3. Compare synthetic and real sensor data characteristics

## References

- [Gazebo Sensor Documentation](http://classic.gazebosim.org/tutorials?tut=ros2_lidar&cat=connect_ros)
- [ROS 2 Sensor Message Types](https://docs.ros.org/en/rolling/p/sensor_msgs/)
- [Unity Robotics Sensors](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [Perception in Robotics with Synthetic Data](https://arxiv.org/abs/1804.06500)