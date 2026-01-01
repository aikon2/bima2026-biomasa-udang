
import cv2

class Camera:
    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(cfg["device"])
        self.cap.set(3, cfg["width"])
        self.cap.set(4, cfg["height"])
        self.cap.set(5, cfg["fps"])

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
