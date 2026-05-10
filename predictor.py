import torch
import numpy as np
import cv2


class Predictor:

    def __init__(self, model, device):

        self.model = model
        self.device = device

    def predict(self, img):

        img = np.expand_dims(img,0)
        img = img.astype("float32")
        img = img / img.max()

        tensor = torch.tensor(img).float().to(self.device)

        with torch.no_grad():

            out = self.model(tensor)

            pred = torch.sigmoid(out)

            pred = (pred>0.5).float()

        pred = pred.cpu().numpy()[0,0]

        # -------- очистка шума --------

        kernel = np.ones((5,5), np.uint8)

        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)

        return pred