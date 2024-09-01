# Avatarify User Manual

Welcome to Avatarify, an advanced AI-powered video reproduction platform. This user manual will guide you through the setup process and explain how to use the various features of Avatarify.

## Table of Contents

1. [Installation](#installation)
2. [Starting Avatarify](#starting-avatarify)
3. [Using the Web Interface](#using-the-web-interface)
4. [Available Effects](#available-effects)
5. [Troubleshooting](#troubleshooting)

## 1. Installation

Before using Avatarify, make sure you have the following prerequisites installed:

- Python 3.7 or higher
- Node.js 14 or higher
- CUDA-compatible GPU (recommended for optimal performance)

Follow these steps to install Avatarify:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/avatarify.git
   cd avatarify
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies:
   ```
   cd nextjs_interface
   npm install
   ```

## 2. Starting Avatarify

To start Avatarify, follow these steps:

1. Open a terminal and navigate to the Avatarify directory.

2. Start the Python backend:
   ```
   python main.py
   ```

3. Open a new terminal window, navigate to the Avatarify directory, and start the Next.js frontend:
   ```
   cd nextjs_interface
   npm run dev
   ```

4. Open a web browser and go to `http://localhost:3000` to access the Avatarify web interface.

## 3. Using the Web Interface

The Avatarify web interface consists of the following elements:

- Video display: Shows the processed video stream in real-time.
- Effect selector: A dropdown menu to choose from available video effects.
- Apply Effect button: Applies the selected effect to the video stream.

To use the interface:

1. Select an effect from the dropdown menu.
2. Click the "Apply Effect" button to apply the chosen effect to the video stream.
3. The effect will be applied in real-time, and you'll see the results in the video display.

## 4. Available Effects

Avatarify offers the following effects:

- Face Swap: Replaces faces in the video with a chosen face.
- Style Transfer: Applies the style of a chosen image to the video.
- Cartoon: Gives the video a cartoon-like appearance.
- Deep Dream: Applies a psychedelic, dream-like effect to the video.
- Glitch: Adds a glitch effect to the video.
- VHS: Simulates the look of old VHS tapes.
- Rainbow: Applies a colorful, rainbow effect to the video.

## 5. Troubleshooting

If you encounter any issues while using Avatarify, try the following:

- Ensure all dependencies are correctly installed.
- Check that your GPU drivers are up to date.
- Verify that the paths to the face swap, style transfer, and voice cloning models in `main.py` are correct.
- If the video stream is slow or laggy, try reducing the video resolution or frame rate in the `config.py` file.

If problems persist, please check the project's GitHub repository for known issues or to report a new one.

For additional help or to contribute to the project, please refer to the README.md file in the project root directory.