# Intent Extraction from Transcribed Text

## Introduction

Intent extraction is the process of identifying the user's intended action from transcribed text. This critical step in the voice-to-action pipeline transforms natural language into structured commands that can be executed by the robot. This section covers techniques for extracting intent from transcribed voice commands.

## Understanding Intent Extraction

Intent extraction involves analyzing transcribed text to determine:

- **Action**: What the user wants the robot to do
- **Objects**: What entities are involved in the action
- **Parameters**: Specific details about how to perform the action
- **Constraints**: Limitations or conditions for the action

### Example Intent Extraction

Input: "Move forward 2 meters and then turn left"

Output:
- Action: Navigation
- Sub-actions: Move forward, Turn left
- Parameters: Distance = 2 meters
- Sequence: Execute move first, then turn

## Approaches to Intent Extraction

### 1. Rule-Based Approach

Rule-based systems use predefined patterns to identify intents:

```python
import re

class RuleBasedIntentExtractor:
    def __init__(self):
        # Define patterns for different command types
        self.patterns = {
            'navigation': [
                r'move\s+(?P<direction>forward|backward|left|right)\s+(?P<distance>\d+)\s+meters?',
                r'go\s+(?P<direction>forward|backward|left|right)',
                r'go\s+to\s+(?P<location>\w+)'
            ],
            'manipulation': [
                r'pick\s+up\s+(?P<object>[\w\s]+)',
                r'place\s+(?P<object>[\w\s]+)\s+on\s+(?P<location>[\w\s]+)',
                r'hand\s+me\s+(?P<object>[\w\s]+)'
            ]
        }

    def extract_intent(self, text):
        """Extract intent using rule-based patterns"""
        text = text.lower().strip()

        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return {
                        'intent': intent_type,
                        'action': intent_type,
                        'parameters': match.groupdict(),
                        'confidence': 0.9  # High confidence for rule-based match
                    }

        # If no pattern matches, return unknown intent
        return {
            'intent': 'unknown',
            'action': 'unknown',
            'parameters': {},
            'confidence': 0.0
        }
```

### 2. Machine Learning Approach

Using machine learning models for intent classification:

```python
from transformers import pipeline

class MLIIntentExtractor:
    def __init__(self):
        # Use a pre-trained model for intent classification
        self.classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium"
        )

    def extract_intent(self, text):
        """Extract intent using machine learning"""
        result = self.classifier(text)

        return {
            'intent': result['label'],
            'confidence': result['score']
        }
```

### 3. Hybrid Approach

Combining rule-based and ML approaches for better accuracy:

```python
class HybridIntentExtractor:
    def __init__(self):
        self.rule_extractor = RuleBasedIntentExtractor()
        self.ml_extractor = MLIIntentExtractor()

    def extract_intent(self, text):
        """Extract intent using hybrid approach"""
        # Try rule-based extraction first
        rule_result = self.rule_extractor.extract_intent(text)

        if rule_result['confidence'] > 0.8:
            return rule_result

        # Fall back to ML extraction for ambiguous cases
        ml_result = self.ml_extractor.extract_intent(text)

        # Combine results with confidence scoring
        if ml_result['confidence'] > 0.7:
            return {
                'intent': ml_result['intent'],
                'confidence': ml_result['confidence']
            }

        return rule_result  # Return rule result even if low confidence
```

## Intent Extraction in ROS 2

### ROS 2 Intent Extraction Node

```python
#!/usr/bin/env python3
# ROS 2 node for intent extraction

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import Intent  # Custom message type

class IntentExtractionNode(Node):
    def __init__(self):
        super().__init__('intent_extraction_node')

        # Initialize intent extractor
        self.extractor = HybridIntentExtractor()

        # Subscribe to transcribed text
        self.text_sub = self.create_subscription(
            String,
            'transcribed_text',
            self.text_callback,
            10
        )

        # Publish extracted intents
        self.intent_pub = self.create_publisher(
            Intent,
            'extracted_intent',
            10
        )

        self.get_logger().info('Intent Extraction Node Initialized')

    def text_callback(self, msg):
        """Process transcribed text and extract intent"""
        try:
            # Extract intent from text
            intent_data = self.extractor.extract_intent(msg.data)

            # Create intent message
            intent_msg = Intent()
            intent_msg.action = intent_data['intent']
            intent_msg.confidence = intent_data['confidence']
            intent_msg.original_text = msg.data

            # Extract and format parameters
            if 'parameters' in intent_data:
                for key, value in intent_data['parameters'].items():
                    param = Intent.Parameter()
                    param.name = key
                    param.value = str(value)
                    intent_msg.parameters.append(param)

            # Publish the extracted intent
            self.intent_pub.publish(intent_msg)
            self.get_logger().info(f'Extracted intent: {intent_data["intent"]}')

        except Exception as e:
            self.get_logger().error(f'Error in intent extraction: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = IntentExtractionNode()

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

## Custom Intent Message Definition

Create a custom message type for intents:

```text
# vla_interfaces/msg/Intent.msg
string action
float32 confidence
string original_text
Parameter[] parameters

# Nested message for parameters
# vla_interfaces/msg/Parameter.msg
string name
string value

# Define common action types
# vla_interfaces/msg/ActionType.msg
string NAVIGATION = "navigation"
string MANIPULATION = "manipulation"
string INFORMATION = "information"
string SYSTEM = "system"
```

## Advanced Intent Extraction Techniques

### Named Entity Recognition (NER)

Using NER to identify objects and locations:

```python
import spacy

class NERIntentExtractor:
    def __init__(self):
        # Load spaCy model for NER
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        """Extract named entities from text"""
        doc = self.nlp(text)

        entities = {
            'locations': [],
            'objects': [],
            'people': [],
            'quantities': []
        }

        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical entities, locations, facilities
                entities['locations'].append(ent.text)
            elif ent.label_ in ['OBJECT', 'PRODUCT']:  # Objects and products
                entities['objects'].append(ent.text)
            elif ent.label_ == 'PERSON':  # People
                entities['people'].append(ent.text)
            elif ent.label_ in ['QUANTITY', 'CARDINAL', 'MONEY']:  # Quantities
                entities['quantities'].append(ent.text)

        return entities
```

### Context-Aware Intent Extraction

Consider the robot's current state and environment:

```python
class ContextAwareExtractor:
    def __init__(self):
        self.current_location = None
        self.visible_objects = []
        self.robot_state = 'idle'

    def extract_intent_with_context(self, text, context):
        """Extract intent considering current context"""
        # Get basic intent
        basic_intent = self.extract_intent(text)

        # Enhance with context
        enhanced_intent = {
            'intent': basic_intent['intent'],
            'confidence': basic_intent['confidence'],
            'context': context,
            'resolved_entities': self.resolve_entities(
                basic_intent.get('parameters', {}),
                context
            )
        }

        return enhanced_intent

    def resolve_entities(self, parameters, context):
        """Resolve ambiguous entities using context"""
        resolved = parameters.copy()

        # Resolve relative directions based on robot orientation
        if 'direction' in resolved and resolved['direction'] in ['left', 'right']:
            resolved['direction'] = self.absolute_direction(
                resolved['direction'],
                context.get('robot_orientation', 0)
            )

        # Resolve ambiguous object references
        if 'object' in resolved:
            resolved['object'] = self.disambiguate_object(
                resolved['object'],
                context.get('visible_objects', [])
            )

        return resolved
```

## Intent Confidence and Validation

### Confidence Scoring

```python
def calculate_intent_confidence(self, text, extracted_intent):
    """Calculate confidence score for extracted intent"""
    confidence = extracted_intent['confidence']

    # Adjust confidence based on various factors
    if len(text) < 3:
        confidence *= 0.5  # Lower confidence for very short text

    # Check for ambiguous words
    ambiguous_words = ['maybe', 'perhaps', 'kind of', 'sort of']
    if any(word in text.lower() for word in ambiguous_words):
        confidence *= 0.7

    # Check for multiple possible interpretations
    possible_intents = self.get_possible_intents(text)
    if len(possible_intents) > 1:
        confidence *= 0.8  # Lower confidence for ambiguous text

    return min(confidence, 1.0)  # Ensure confidence doesn't exceed 1.0
```

### Intent Validation

```python
def validate_intent(self, intent):
    """Validate extracted intent before execution"""
    # Check if intent is supported
    supported_intents = ['navigation', 'manipulation', 'information', 'system']
    if intent['intent'] not in supported_intents:
        return False, f"Unsupported intent: {intent['intent']}"

    # Check parameter validity
    if intent['intent'] == 'navigation':
        if 'distance' in intent.get('parameters', {}):
            distance = float(intent['parameters']['distance'])
            if distance <= 0 or distance > 100:  # Reasonable distance limits
                return False, f"Invalid distance: {distance}"

    # Additional validation rules can be added here
    return True, "Intent is valid"
```

## Error Handling and Fallback Strategies

### Handling Unknown Intents

```python
def handle_unknown_intent(self, text):
    """Handle cases where intent cannot be determined"""
    # Ask for clarification
    clarification_request = f"I didn't understand '{text}'. Could you rephrase that?"

    # Publish clarification request
    clarification_msg = String()
    clarification_msg.data = clarification_request
    self.clarification_pub.publish(clarification_msg)

    return {
        'intent': 'clarification_needed',
        'message': clarification_request,
        'confidence': 0.0
    }
```

### Fallback Mechanisms

```python
class RobustIntentExtractor:
    def __init__(self):
        self.primary_extractor = HybridIntentExtractor()
        self.fallback_extractors = [
            RuleBasedIntentExtractor(),
            SimpleKeywordExtractor()
        ]

    def extract_intent_robust(self, text):
        """Extract intent with fallback mechanisms"""
        # Try primary extractor
        result = self.primary_extractor.extract_intent(text)

        if result['confidence'] > 0.7:
            return result

        # Try fallback extractors
        for fallback_extractor in self.fallback_extractors:
            fallback_result = fallback_extractor.extract_intent(text)
            if fallback_result['confidence'] > result['confidence']:
                result = fallback_result
                if result['confidence'] > 0.7:
                    break

        return result
```

## Performance Considerations

### Caching Common Intents

```python
from functools import lru_cache

class CachedIntentExtractor:
    def __init__(self):
        self.extractor = HybridIntentExtractor()
        self.cache_size = 100

    @lru_cache(maxsize=100)
    def cached_extract_intent(self, text):
        """Extract intent with caching for common phrases"""
        return self.extractor.extract_intent(text)
```

## Integration with ROS 2 Actions

Intent extraction results can be used to trigger ROS 2 actions:

```python
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class IntentToActionMapper:
    def __init__(self, node):
        self.node = node
        self.nav_client = ActionClient(node, NavigateToPose, 'navigate_to_pose')

    def map_intent_to_action(self, intent_msg):
        """Map extracted intent to ROS 2 action"""
        if intent_msg.action == 'navigation':
            self.execute_navigation_action(intent_msg)
        elif intent_msg.action == 'manipulation':
            self.execute_manipulation_action(intent_msg)
        # Add more action mappings as needed
```

## Testing Intent Extraction

### Unit Testing

```python
import unittest

class TestIntentExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = HybridIntentExtractor()

    def test_navigation_intent(self):
        text = "Move forward 2 meters"
        result = self.extractor.extract_intent(text)

        self.assertEqual(result['intent'], 'navigation')
        self.assertGreater(result['confidence'], 0.7)

    def test_manipulation_intent(self):
        text = "Pick up the red ball"
        result = self.extractor.extract_intent(text)

        self.assertEqual(result['intent'], 'manipulation')
        self.assertIn('red ball', result['parameters'].get('object', ''))
```

The next section will cover ROS 2 integration for publishing actions from extracted intents.