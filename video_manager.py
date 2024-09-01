import cv2
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class VideoManager(QObject):
    playback_frame_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.recorder = None
        self.player = None

    def start_recording(self, output_path, fps, resolution):
        if self.recorder:
            self.stop_recording()
        self.recorder = VideoRecorder(output_path, fps, resolution)
        self.recorder.start()

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    def record_frame(self, frame):
        if self.recorder:
            self.recorder.add_frame(frame)

    def start_playback(self, video_path):
        if self.player:
            self.stop_playback()
        self.player = VideoPlayer(video_path)
        self.player.frame_signal.connect(self.playback_frame_signal)
        self.player.start()

    def stop_playback(self):
        if self.player:
            self.player.stop()
            self.player = None

class VideoRecorder(QThread):
    def __init__(self, output_path, fps, resolution):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.active = False

    def run(self):
        self.active = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)
        while self.active or self.frames:
            if self.frames:
                frame = self.frames.pop(0)
                out.write(frame)
        out.release()

    def add_frame(self, frame):
        self.frames.append(frame)

    def stop(self):
        self.active = False
        self.wait()

class VideoPlayer(QThread):
    frame_signal = pyqtSignal(object)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.active = False

    def run(self):
        self.active = True
        cap = cv2.VideoCapture(self.video_path)
        while self.active:
            ret, frame = cap.read()
            if ret:
                self.frame_signal.emit(frame)
            else:
                break
        cap.release()

    def stop(self):
        self.active = False
        self.wait()