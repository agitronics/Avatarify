import torch

class Config:
    def __init__(self):
        self.config = {
            'api_url': 'https://api.example.com',
            'api_key': 'your_api_key_here',
            'facial_gesture_model_path': 'path/to/facial_gesture_model.pth',
            'num_gestures': 10,
            'video_width': 640,
            'video_height': 480,
            'fps': 30,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

    def get(self, key):
        return self.config.get(key)