// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2-nervous-system/index',
        'module-1-ros2-nervous-system/chapter-1-ros2-foundations',
        'module-1-ros2-nervous-system/chapter-2-python-ai-agents',
        'module-1-ros2-nervous-system/chapter-3-humanoid-representation',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/index',
        'module-2-digital-twin/chapter-1-gazebo-physics-simulation',
        'module-2-digital-twin/chapter-2-unity-digital-environments',
        'module-2-digital-twin/chapter-3-sensor-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3-isaac-robot-brain/index',
        'module-3-isaac-robot-brain/chapter-1-isaac-sim-synthetic-data',
        'module-3-isaac-robot-brain/chapter-2-isaac-ros-perception',
        'module-3-isaac-robot-brain/chapter-3-nav2-humanoid-navigation',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/intro',
        'module-4-vla/glossary',
        'module-4-vla/code-conventions',
        'module-4-vla/prerequisites',
        {
          type: 'category',
          label: 'Voice-to-Action Interfaces',
          items: [
            'module-4-vla/voice-to-action/index',
            'module-4-vla/voice-to-action/voice-commands',
            'module-4-vla/voice-to-action/speech-to-text',
            'module-4-vla/voice-to-action/intent-extraction',
            'module-4-vla/voice-to-action/ros2-integration',
          ],
        },
        {
          type: 'category',
          label: 'LLM-Based Cognitive Planning',
          items: [
            'module-4-vla/llm-planning/index',
            'module-4-vla/llm-planning/task-decomposition',
            'module-4-vla/llm-planning/language-goals',
            'module-4-vla/llm-planning/constraint-aware-planning',
            'module-4-vla/llm-planning/human-in-loop',
          ],
        },
        {
          type: 'category',
          label: 'Capstone: The Autonomous Humanoid',
          items: [
            'module-4-vla/capstone/index',
            'module-4-vla/capstone/end-to-end-pipeline',
            'module-4-vla/capstone/ros2-perception-control',
            'module-4-vla/capstone/simulation-validation',
          ],
        },
      ],
    },
  ],
};

module.exports = sidebars;