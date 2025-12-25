# Speech-to-Text Integration with OpenAI Whisper

## Introduction

Speech-to-text (STT) conversion is a critical component of voice-to-action interfaces, transforming spoken commands into written text that can be processed by intent extraction systems. This section covers the integration of OpenAI Whisper for speech-to-text processing in ROS 2 environments.

## OpenAI Whisper Overview

OpenAI Whisper is a state-of-the-art automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data. It offers several advantages for robotics applications:

- **Multilingual Support**: Supports multiple languages out of the box
- **Robustness**: Performs well in various acoustic conditions
- **Accuracy**: High transcription accuracy across different accents
- **Open Source**: Available under MIT license for research and commercial use

### Whisper Model Variants

Whisper is available in several model sizes with different performance characteristics:

| Model | Size | Required VRAM | Relative Speed | English-only | Multilingual |
|-------|------|---------------|----------------|--------------|--------------|
| tiny  | 75MB | ~1GB | ~32x | ✅ | ✅ |
| base  | 145MB | ~1GB | ~16x | ✅ | ✅ |
| small | 444MB | ~2GB | ~6x | ✅ | ✅ |
| medium | 1.5GB | ~5GB | ~2x | ✅ | ✅ |
| large | 3.0GB | ~10GB | 1x | ❌ | ✅ |

For robotics applications, the `small` or `medium` models typically provide the best balance of accuracy and resource requirements.

## Integration with ROS 2

### Installation Requirements

```bash
pip install openai-whisper
pip install faster-whisper  # Optional: faster implementation
```

### Basic Whisper Node Implementation

```python
#!/usr/bin/env python3
# ROS 2 node for Whisper-based speech-to-text

import rclpy
from rclpy.node import Node
import whisper
import numpy as np
import io
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import wave

class WhisperSTTNode(Node):
    def __init__(self):
        super().__init__('whisper_stt_node')

        # Load Whisper model
        self.model = whisper.load_model("small")  # or "medium", "base", etc.

        # Create subscriber for audio data
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Create publisher for transcribed text
        self.text_pub = self.create_publisher(
            String,
            'transcribed_text',
            10
        )

        self.get_logger().info('Whisper STT Node Initialized')

    def audio_callback(self, msg):
        """Process incoming audio data and convert to text"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16)

            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Transcribe using Whisper
            result = self.model.transcribe(audio_float)
            transcribed_text = result['text']

            # Publish the transcribed text
            text_msg = String()
            text_msg.data = transcribed_text
            self.text_pub.publish(text_msg)

            self.get_logger().info(f'Transcribed: {transcribed_text}')

        except Exception as e:
            self.get_logger().error(f'Error in audio processing: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = WhisperSTTNode()

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

## Advanced Whisper Configuration

### Language-Specific Transcription

```python
def transcribe_with_language(self, audio_data, language='en'):
    """Transcribe with specific language"""
    result = self.model.transcribe(
        audio_data,
        language=language,
        task='transcribe'  # or 'translate' for translation
    )
    return result['text']
```

### Real-Time Processing Considerations

For real-time applications, consider these optimizations:

```python
def setup_real_time_processing(self):
    """Configure Whisper for real-time processing"""
    # Use faster-whisper for better performance
    from faster_whisper import WhisperImpl

    self.model = WhisperImpl("small")

    # Process audio in chunks
    self.audio_buffer = []
    self.chunk_size = 16000 * 2  # 2 seconds of audio at 16kHz
```

## Performance Optimization

### Model Loading and Caching

```python
def load_model_efficiently(self):
    """Load model with memory optimization"""
    # Load model to GPU if available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    self.model = whisper.load_model("small").to(device)
```

### Accuracy Considerations

Whisper's accuracy can be affected by:

- **Audio Quality**: Clean, high-quality audio improves accuracy
- **Background Noise**: Noisy environments may reduce accuracy
- **Microphone Distance**: Proximity to microphone affects clarity
- **Speech Rate**: Normal speech rate is optimal for recognition

## Integration Patterns

### Streaming Audio Processing

```python
class StreamingWhisperNode(Node):
    def __init__(self):
        super().__init__('streaming_whisper_node')

        # Buffer for audio chunks
        self.audio_buffer = np.array([])
        self.buffer_size = 16000 * 3  # 3 seconds at 16kHz

        # Timer for periodic transcription
        self.transcription_timer = self.create_timer(
            3.0,  # Transcribe every 3 seconds
            self.transcribe_buffer
        )
```

### Error Handling and Validation

```python
def validate_transcription(self, text):
    """Validate transcription quality"""
    # Check for empty results
    if not text.strip():
        return False, "Empty transcription"

    # Check for common error patterns
    if len(text) < 3:
        return False, "Transcription too short"

    # Additional validation rules can be added here
    return True, text
```

## ROS 2 Message Integration

### Audio Data Format

Whisper expects audio data in specific formats:

```python
def convert_audio_format(self, audio_msg):
    """Convert ROS audio message to Whisper-compatible format"""
    # Assuming audio_msg contains 16-bit integer samples
    audio_array = np.frombuffer(audio_msg.data, dtype=np.int16)

    # Convert to float32 (Whisper expects float32)
    audio_float = audio_array.astype(np.float32) / 32768.0

    return audio_float
```

## Configuration Parameters

### Common Whisper Parameters

```python
def transcribe_with_options(self, audio_data):
    """Transcribe with various options"""
    result = self.model.transcribe(
        audio_data,
        # Language options
        language='en',

        # Task options
        task='transcribe',  # 'transcribe' or 'translate'

        # Performance options
        temperature=0.0,  # Deterministic output
        best_of=1,        # Number of candidates to return
        beam_size=5,      # Beam search size
        patience=1.0,     # Patience for beam search
    )

    return result
```

## Accuracy and Performance Metrics

### Measuring STT Performance

```python
def evaluate_stt_performance(self, reference_text, transcribed_text):
    """Evaluate STT accuracy"""
    from jiwer import wer, cer  # Install with: pip install jiwer

    word_error_rate = wer(reference_text, transcribed_text)
    char_error_rate = cer(reference_text, transcribed_text)

    return {
        'word_error_rate': word_error_rate,
        'character_error_rate': char_error_rate,
        'accuracy': 1.0 - word_error_rate
    }
```

## Troubleshooting Common Issues

### Audio Format Issues
- Ensure audio data is in the correct format (16kHz, mono recommended)
- Verify audio levels are appropriate (not too quiet or too loud)

### Memory Issues
- Use smaller model variants if running on resource-constrained systems
- Consider using faster-whisper for better memory efficiency

### Accuracy Issues
- Ensure good audio quality and minimal background noise
- Consider preprocessing audio with noise reduction techniques
- Verify the language model matches the spoken language

## Security Considerations

When implementing Whisper-based STT:

- **Privacy**: Audio data may contain sensitive information
- **Data Handling**: Consider local processing to avoid sending audio to external services
- **Access Control**: Secure audio input and transcription outputs

The next section will cover intent extraction from the transcribed text.