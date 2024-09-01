import cv2
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class CameraManager(QObject):
    frame_signal = pyqtSignal(object, int)

    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.active = False

    def add_camera(self, camera_id, source):
        if camera_id not in self.cameras:
            camera = Camera(camera_id, source)
            camera.frame_signal.connect(self.handle_frame)
            self.cameras[camera_id] = camera
            if self.active:
                camera.start()

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]

    def start(self):
        self.active = True
        for camera in self.cameras.values():
            camera.start()

    def stop(self):
        self.active = False
        for camera in self.cameras.values():
            camera.stop()

    def handle_frame(self, frame, camera_id):
        self.frame_signal.emit(frame, camera_id)

    def get_available_cameras(self):
        available_cameras = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                available_cameras.append(index)
            cap.release()
            index += 1
        return available_cameras

    def get_active_cameras(self):
        return list(self.cameras.keys())

class Camera(QThread):
    frame_signal = pyqtSignal(object, int)

    def __init__(self, camera_id, source):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.active = False

    def run(self):
        self.active = True
        cap = cv2.VideoCapture(self.source)
        while self.active:
            ret, frame = cap.read()
            if ret:
                self.frame_signal.emit(frame, self.camera_id)
        cap.release()

    def stop(self):
        self.active = False
        self.wait()