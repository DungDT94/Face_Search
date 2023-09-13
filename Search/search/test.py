import pynndescent
import numpy as np
import h5py
from urllib.request import urlretrieve
import os
import pickle
from scipy.spatial import distance
from ..FeatureExtract import *

data_feature = np.load("/home/dungdinh/Documents/insightface/recognition/arcface_torch/feature/feature16.npy",
                       allow_pickle=True)
data_index = np.load("/home/dungdinh/Documents/insightface/recognition/arcface_torch/feature/name16.npy", allow_pickle=True)


def distance_(feat1, feat2):
    d = distance.euclidean(feat1, feat2)
    return d


index = pynndescent.NNDescent(data_feature, n_neighbors=50)
index.prepare()
# with open("index_class.pkl", "wb") as file:
# pickle.dump(index, file)
# with open("index_class.pkl", "rb") as file:
# index_class = pickle.load(file)
# index_class.prepare()
model = Extractor('/home/dungdinh/Documents/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_fp16'
                      '/backbone.pth', 'r100')
#test = np.load("/home/dungdinh/Documents/insightface/recognition/arcface_torch/feature_test.npy", allow_pickle=True)
test = model.extract('/home/dungdinh/Documents/FaceNet-Infer/data_test/4/2.png')
neighbors = index.query([test], k=15)
print(neighbors)
print(neighbors[1])
print(data_index[2306])
distance1 = 999
for index in neighbors[0][0]:
    d = distance_(data_feature[index], test)
    if d < distance1:
        distance1 = d
print(distance1)
