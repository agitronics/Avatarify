# Avatarify - Advanced AI-Powered Video Reproduction

Avatarify is a powerful platform for real-time video manipulation using advanced AI techniques. This project combines cutting-edge face swapping, style transfer, voice cloning, and various video effects to create a unique and interactive video processing experience.

## Features

- Advanced face swapping using a ResNet50-based model
- Improved style transfer with adaptive instance normalization
- Text-to-speech voice cloning
- Real-time video processing with various effects
- Web-based user interface for easy control

## Prerequisites

- Python 3.7 or higher
- Node.js 14 or higher
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/AGITRONICS/avatarify.git
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

## Usage

1. Start the Python backend:
   ```
   cd avatarify
   python main.py
   ```

2. In a new terminal, start the Next.js frontend:
   ```
   cd avatarify/nextjs_interface
   npm run dev
   ```

3. Open a web browser and navigate to `http://localhost:3000` to access the Avatarify web interface.

4. Use the interface to apply various effects and manipulate the video stream in real-time.

## Configuration

You may need to adjust the paths for the face swap model, style transfer model, and voice cloning model in the `main.py` file to match your specific setup.

## Contributing

If you like this work, or use it to make money, a tip would be most helpful in further developmemnt of this or other things. 
btc: 3Kz2rfM7E3nN8ovbMcggWMW7maQar7zhdW
eth: 0x7b6Df61215C3DE2138Ee52Cc22cFa7eBbc9c7789


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [OpenCV](https://opencv.org/) for computer vision tasks
- [Next.js](https://nextjs.org/) for the web interface
