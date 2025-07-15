# Fire Detection System

A real-time fire detection system built with Python and OpenCV, using Vision Transformer (ViT) deep learning model for accurate fire detection in video streams.

## Features

- 🔥 Real-time fire detection using Vision Transformer (ViT) model
- 📹 Support for both webcam and video file input
- 🎛️ Adjustable detection threshold
- 📊 Real-time probability display
- 📝 Detection event logging
- 🎯 GPU acceleration support (when available)
- 🖥️ Modern Tkinter-based GUI

## Requirements

- Python 3.7+
- OpenCV
- PyTorch
- Transformers (Hugging Face)
- Tkinter
- Pillow (PIL)
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/not-ekalabya/security-cam.git
cd security-cam
```

2. Install the required packages:
```bash
pip install opencv-python torch torchvision transformers pillow
```

## Usage

1. Run the application:
```bash
python main.py
```

2. In the application:
   - Click "Load Fire Model" to initialize the detection model
   - Choose your input source:
     - "Start Webcam" for live camera feed
     - "Load Video File" to process a video file
   - Adjust the detection threshold as needed
   - Use the "Stop" button to end detection

## Controls

- **Fire Detection Toggle**: Enable/disable fire detection
- **Threshold Adjustment**: Fine-tune detection sensitivity (0.01-0.99)
- **Input Controls**: Switch between webcam and video file sources
- **Model Management**: Load/unload detection models
- **Stop Button**: Halt current detection process

## Model Information

The system uses a Vision Transformer (ViT) model fine-tuned for fire detection:
- Model: `EdBianchi/vit-fire-detection`
- Supports both CPU and GPU inference
- Real-time detection with probability scores

## Sample Data

The repository includes sample video files in the `data/` directory for testing:
- `fire.mp4`
- `lithium_fires.mp4`

## Contributing

Feel free to open issues or submit pull requests for improvements and bug fixes.

## License

[MIT License](LICENSE)

## Author

not-ekalabya
