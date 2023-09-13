import numpy as np
from Detection.retinaface import RetinaFace
from Detection.warped import preprocess
from config import retina_weight
import os


class DetectFace:
    def __init__(self):
        self.thresh = 0.8
        self.scales = [1024, 1980]
        self.gpu = -1
        self.flip = False
        self.detector = RetinaFace(os.path.join(retina_weight, 'R50'),
                                   0,
                                   self.gpu, 'net3')

    def detect(self, img):
        list_img = []
        im_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [im_scale]
        faces, landmarks = self.detector.detect(img,
                                                self.thresh,
                                                scales=scales,
                                                do_flip=self.flip)

        for i, face in enumerate(faces):
            img_warped = preprocess(img, face[0:4], landmarks[i])
            list_img.append(img_warped)
        return list_img


