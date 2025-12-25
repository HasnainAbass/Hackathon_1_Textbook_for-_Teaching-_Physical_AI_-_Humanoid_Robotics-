# Quickstart Guide: Module 2 - The Digital Twin (Gazebo & Unity)

**Date**: 2025-12-22
**Feature**: 002-digital-twin-sim
**Status**: Complete

## Overview

This quickstart guide provides the essential steps to set up and begin working with Module 2: The Digital Twin (Gazebo & Unity) educational content. This module introduces AI and robotics students to digital twin concepts using Gazebo for physics simulation and Unity for high-fidelity environments.

## Prerequisites

- Basic ROS 2 knowledge
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

The Module 2 content will be accessible through the documentation navigation:
- Main navigation → "Module 2: The Digital Twin (Gazebo & Unity)"
- Or directly at `/docs/module-2-digital-twin/`

## Content Structure

### Chapter 1: Gazebo Physics Simulation
- Role of digital twins in Physical AI
- Simulating gravity, collisions, and dynamics
- Integrating URDF with Gazebo
- Validating robot behavior in simulation

### Chapter 2: Unity Digital Environments
- Purpose of Unity in robotics
- Visual realism and human-robot interaction
- Synchronizing Unity with ROS 2
- Sim-to-real considerations

### Chapter 3: Sensor Simulation for Humanoid Robots
- Simulating LiDAR, depth cameras, and IMUs
- Publishing simulated sensor data to ROS 2
- Using synthetic sensor data for testing perception

## Learning Path

1. **Start with Chapter 1** to understand Gazebo physics simulation
2. **Proceed to Chapter 2** to learn about Unity digital environments
3. **Complete Chapter 3** to understand sensor simulation
4. **Complete exercises** in each chapter to reinforce learning

## Key Resources

- [Gazebo Classic Documentation](http://classic.gazebosim.org/)
- [Gazebo Garden Documentation](https://gazebosim.org/)
- [Unity Robotics Hub](https://unity.com/solutions/industrial/robotics)
- [ROS 2 Gazebo Integration](https://github.com/ros-simulation/gazebo_ros_pkgs)

## Troubleshooting

**Issue**: Cannot access documentation locally
- **Solution**: Ensure Node.js and npm are properly installed
- **Solution**: Run `npm install` to install dependencies

**Issue**: Content appears outdated
- **Solution**: Pull latest changes from repository
- **Solution**: Clear browser cache or try incognito/private mode

## Next Steps

After completing Module 2, continue with:
- Module 3: Perception and AI (covers advanced perception and AI training)
- Module 4: Voice-to-Action and LLM Cognitive Planning (covers AI interaction)