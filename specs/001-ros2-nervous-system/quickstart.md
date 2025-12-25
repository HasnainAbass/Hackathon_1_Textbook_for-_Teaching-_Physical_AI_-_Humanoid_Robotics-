# Quickstart Guide: Module 1 - The Robotic Nervous System (ROS 2)

**Date**: 2025-12-22
**Feature**: 001-ros2-nervous-system
**Status**: Complete

## Overview

This quickstart guide provides the essential steps to set up and begin working with Module 1: The Robotic Nervous System (ROS 2) educational content. This module introduces AI and robotics students to ROS 2 as a middleware for humanoid robots.

## Prerequisites

- Basic Python knowledge
- Understanding of fundamental programming concepts
- Web browser for viewing documentation
- Git for version control (optional, for local development)

## Setup Instructions

### 1. Environment Setup

1. **Install Node.js and npm** (for local Docusaurus development):
   ```bash
   # Check if Node.js is installed
   node --version

   # If not installed, download from nodejs.org
   # Or use a package manager like nvm
   ```

2. **Clone or access the documentation repository**:
   ```bash
   # If working locally:
   git clone <repository-url>
   cd <repository-name>
   ```

### 2. Docusaurus Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start local development server**:
   ```bash
   npm start
   ```
   This will start a local server at http://localhost:3000

### 3. Accessing Module Content

The Module 1 content will be accessible through the documentation navigation:
- Main navigation → "Module 1: The Robotic Nervous System (ROS 2)"
- Or directly at `/docs/module-1-ros2-nervous-system/`

## Content Structure

### Chapter 1: ROS 2 Foundations
- Introduction to ROS 2 and its importance for Physical AI
- Core architecture: nodes, topics, services, and actions
- DDS-based communication and real-time considerations
- Modular, distributed robot systems

### Chapter 2: Python AI Agents with rclpy
- Role of Python agents in robot control
- Creating ROS 2 nodes using rclpy
- Publishing/subscribing to topics for sensor and actuator data
- Using services for request-response robot behaviors
- Bridging AI decision logic to low-level ROS controllers

### Chapter 3: Humanoid Representation with URDF
- Purpose of URDF in humanoid robotics
- Defining links, joints, and kinematic chains
- Modeling humanoid structures and constraints
- Connecting physical body to ROS 2 control and simulation

## Learning Path

1. **Start with Chapter 1** to understand the foundational concepts
2. **Proceed to Chapter 2** to learn practical implementation with Python
3. **Complete Chapter 3** to understand robot representation
4. **Complete exercises** in each chapter to reinforce learning

## Key Resources

- [Official ROS 2 Documentation](https://docs.ros.org/)
- [rclpy API Documentation](https://docs.ros.org/en/rolling/p/rclpy/)
- [URDF Documentation](http://wiki.ros.org/urdf)

## Troubleshooting

**Issue**: Cannot access documentation locally
- **Solution**: Ensure Node.js and npm are properly installed
- **Solution**: Run `npm install` to install dependencies

**Issue**: Content appears outdated
- **Solution**: Pull latest changes from repository
- **Solution**: Clear browser cache or try incognito/private mode

## Next Steps

After completing Module 1, continue with:
- Module 2: Simulation and Control (covers simulation and Gazebo)
- Module 3: Perception and AI (covers advanced perception and AI training)