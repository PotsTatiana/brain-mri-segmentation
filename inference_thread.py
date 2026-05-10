# inference_thread.py
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import numpy as np
import cv2

class InferenceThread(QThread):
    finished = pyqtSignal(np.ndarray)   # маска для одного среза

    def __init__(self, model, image_2d, threshold, device, kernel_size=5):
        """
        image_2d: 2D numpy (H, W) - один канал MRI
        model: модель, принимающая (1,2,H,W) - дублируем канал
        """
        super().__init__()
        self.model = model
        self.image = image_2d
        self.threshold = threshold
        self.device = device
        self.kernel_size = kernel_size

    def run(self):
        # Подготовка входа: (1,2,H,W) дублируем канал, так как модель обучена на 2 каналах
        img_tensor = torch.from_numpy(self.image).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        img_tensor = img_tensor.repeat(1, 2, 1, 1)  # (1,2,H,W)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]  # (H,W)

        binary = (pred > self.threshold).astype(np.uint8)
        # Морфологическая очистка
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        self.finished.emit(binary)