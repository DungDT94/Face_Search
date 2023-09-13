import hnswlib
import numpy as np
from Extraction.FeatureExtract import Extractor
import cv2
from Detection.DetectFace import DetectFace
from scipy.spatial import distance


class Distance:
    def __init__(self):
        self.max_element = 40000
        self.thresh = 1.1
        self.data = []
        self.ids = np.array([])
        self.p = hnswlib.Index(space='l2', dim=512)
        self.p.init_index(max_elements=self.max_element*2, ef_construction=50, M=16)
        self.p.set_ef(10)

    def calculate(self, feature):
        index, distances = self.p.knn_query(feature, k=10)
        names = self.ids[index]
        if distances[0][0] > self.thresh:
            return None, distances
        return names, distances

    def add_user(self, data_add, data_name):
        if isinstance(self.data, list):
            self.data.append(data_add)
            self.data = np.squeeze(np.array(self.data))
        else:
            self.data = np.vstack((self.data, data_add))
        self.p.add_items(data_add)
        self.ids = np.hstack((self.ids, data_name))
        return self.data, self.ids

    def delete_user(self, user_name):
        for i, name in enumerate(self.ids):
            if name == user_name:
                #print(name)
                self.p.mark_deleted(i)

    def show(self):
        return self.p.get_current_count()


def distance_(feat1, feat2):
    d = distance.euclidean(feat1, feat2)
    return d


if __name__ == "__main__":
    model_detect = DetectFace()
    model_feature = Extractor('../Extraction/ms1mv3_arcface_r100_fp16/backbone.pth', 'r100')
    model_distance = Distance()

    data = np.load('/home/dungdinh/Documents/insightface (copy)/Extraction/feature/feature16.npy')
    ids = np.load('/home/dungdinh/Documents/insightface (copy)/Extraction/feature/name16.npy')

    data_1, ids_1 = model_distance.add_user(data, ids)
    print(ids.shape)
    print(data_1.shape)
    print(model_distance.show())

    img = cv2.imread('/home/dungdinh/Downloads/test_face/mai-phuong-thuy-4015.jpeg')
    list_img = model_detect.detect(img)
    for img in list_img:
        feature = model_feature.extract(img)
        names, distances = model_distance.calculate(feature)
        print('names:', names)
        print('distances hsnw:', distances)




    data_add, data_name = model_feature.load_feature_init('/home/dungdinh/Documents/FaceNet-Infer/data_add')
    data_2, ids_2 = model_distance.add_user(data_add, data_name)

    print(ids_2.shape)
    print(data_2.shape)
    print(model_distance.show())

    img = cv2.imread('/home/dungdinh/Downloads/test_face/mai-phuong-thuy-4015.jpeg')
    list_img = model_detect.detect(img)
    for img in list_img:
        feature = model_feature.extract(img)
        names, distances = model_distance.calculate(feature)
        print('names:', names)
        print('distances hsnw:', distances)


    model_distance.delete_user('18')
    print(model_distance.show())

    model_distance.delete_user('15')
    print(model_distance.show())


    img = cv2.imread('/home/dungdinh/Downloads/test_face/mai-phuong-thuy-4015.jpeg')
    list_img = model_detect.detect(img)
    for img in list_img:
        feature = model_feature.extract(img)
        names, distances = model_distance.calculate(feature)
        print('names:', names)
        print('distances hsnw:', distances)

    '''
    img = cv2.imread('/home/dungdinh/Documents/FaceNet-Infer/data_test/15/9.png')
    list_img = model_detect.detect(img)
    for img in list_img:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        feature = model_feature.extract(img)
        names, distances = model_distance.calculate(feature)
        print('names:', names)
        print('distances hsnw:', distances)'''

    '''
    print('before', model_distance.show())
    print('before', model_distance.ids.shape)
    data_add, data_name = model_feature.load_feature_init('/home/dungdinh/Documents/FaceNet-Infer/data_add')
    model_distance.add_user(data_add, data_name)
    print('after', model_distance.show())
    print('after', model_distance.ids.shape)'''




    '''
    for i in range(data.shape[0]):
        #print(data[i].shape)
        if (data[i] == feature).all():
            print(ids[i])'''

    '''
    names, distances = model_distance.calculate(feature)

    print(names, distances)'''
