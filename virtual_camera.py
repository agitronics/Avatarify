import pyvirtualcam
import numpy as np

class VirtualCamera:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None

    def start(self):
        self.cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)

    def add_frame(self, frame):
        if self.cam:
            frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
            self.cam.send(frame_rgb)
            self.cam.sleep_until_next_frame()

    def stop(self):
        if self.cam:
            self.cam.close()
            self.cam = None