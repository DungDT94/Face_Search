import hnswlib
import numpy as np
from scipy.spatial import distance
from pymongo import MongoClient
client = MongoClient()
db = client.test_database
collection = db.test_collection


class Search:
    def __init__(self, data, ids, thresh):
        self.thresh = thresh
        self.data = np.load(data)
        self.ids = np.load(ids)
        self.max_element = self.data.shape[0]
        self.p = hnswlib.Index(space='l2', dim=self.data.shape[1])
        self.p.init_index(max_elements=self.max_element*2, ef_construction=100, M=20)
        self.p.set_ef(25)
        self.p.add_items(self.data)

    def calculate(self, feature):
        index, distances = self.p.knn_query(feature, k=10)
        names = self.ids[index]
        if distances[0][0] > self.thresh:
            return None, distances
        return names, distances

    def show(self):
        return self.p.get_current_count()

    def add_user(self, data_add, data_name):
        self.data = np.vstack((self.data, data_add))
        self.p.add_items(data_add)
        self.ids = np.hstack((self.ids, data_name))
        num_element = self.p.get_current_count()
        data_mongo = {'feature': data_add.tolist(), 'id_feature': num_element, 'id_person': data_name.tolist()}
        collection.insert_one(data_mongo)
        return self.p, self.data, self.ids

    def delete_user(self, user_name):
        delete_list = []
        for i, name in enumerate(self.ids):
            if name == user_name:
                self.p.mark_deleted(i)
                delete_list.append(i)
        collection.delete_many({"id_person": user_name})
        self.data = np.delete(self.data, delete_list, axis=0)
        self.ids = np.delete(self.ids, delete_list)
        return self.p, self.data, self.ids


def distance_(feat1, feat2):
    d = distance.euclidean(feat1, feat2)
    return d


