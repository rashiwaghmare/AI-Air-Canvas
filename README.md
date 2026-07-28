# AI-Air-Canvas
AI-powered virtual drawing application using OpenCV and MediaPipe with real-time hand tracking, contour-based shape recognition, and undo/redo functionality.
# AI Air Canvas

An AI-powered virtual drawing application that enables users to draw in the air using hand gestures detected through a webcam. The project uses OpenCV and MediaPipe for real-time hand tracking and contour-based shape recognition.

## Features

- Real-time hand tracking
- Air drawing using index finger
- Automatic shape recognition
- Circle detection
- Rectangle detection
- Triangle detection
- Undo and Redo functionality
- Canvas clearing
- Full-screen drawing interface
- Real-time webcam interaction

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy

## Shape Recognition Technique

The application analyses completed strokes using contour detection.

The recognition pipeline includes:

- Convex Hull
- Contour Detection
- Polygon Approximation (approxPolyDP)
- Circularity Calculation
- Bounding Rectangle Analysis

Recognised Shapes:

- Circle
- Triangle
- Rectangle

## Controls

| Key | Function |
|------|----------|
| Index Finger | Draw |
| Index + Middle Finger | Move Cursor |
| C | Clear Canvas |
| U | Undo |
| R | Redo |
| Q | Quit |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/AI-Air-Canvas.git

cd AI-Air-Canvas

pip install -r requirements.txt

python main.py
```

## Future Improvements

- Multiple brush sizes
- Gesture-based colour selection
- Save drawings as images
- AI handwriting recognition
- Multi-hand support

## Author

Rashi Waghmare
