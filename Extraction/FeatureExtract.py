import cv2
import numpy as np
import torch
from Extraction.backbones import get_model
from sklearn.preprocessing import normalize
import os
import glob


class Extractor:
    def __init__(self, weight, name):
        self.net = get_model(name, fp16=False)
        self.net.load_state_dict(torch.load(weight, map_location='cuda:0'))
        self.net.eval()

    def extract(self, img):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        with torch.no_grad():
            feat = self.net(img).numpy().astype(np.float16)
            feat = normalize(feat).flatten()
        return feat

    def load_feature_init(self, folders):
        list_feature = []
        list_name = []
        folders = [folder for folder in glob.glob(folders + '/*')]
        for folder in folders:
            print(folder)
            folder_name = os.path.basename(folder)
            images = [image for image in glob.glob(folder + '/*')]
            for image in images:
                img = cv2.imread(image)
                feature = self.extract(img)
                for i in range(20):
                    list_feature.append(feature)
                    list_name.append(folder_name)
        list_feature = np.array(list_feature)
        list_name = np.array(list_name)
        return list_feature, list_name


