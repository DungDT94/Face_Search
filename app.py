from Search.distance import *
from Detection.DetectFace import *
from Extraction.FeatureExtract import *
from config import feature, name
from config import extract_weight


def load_p(data):
    max_element = data.shape[0]
    p = hnswlib.Index(space='l2', dim=data.shape[1])
    p.init_index(max_elements=max_element, ef_construction=100, M=20)
    p.set_ef(25)
    p.add_items(data)
    return p


class FaceSearch(Search):
    def __init__(self, data, ids, thresh):
        super().__init__(data, ids, thresh)
        self.model_detect = DetectFace()
        self.model_extract = Extractor(extract_weight, 'r100')
        self.model_search = Search(data, ids, thresh)

    def process(self, img):
        list_img = self.model_detect.detect(img)
        if len(list_img) != 0:
            list_names = []
            list_distances = []
            for img in list_img:
                feature = self.model_extract.extract(img)
                names, distances = self.model_search.calculate(feature)
                if names is not None:
                    list_names.append(names[0].tolist())
                    list_distances.append(distances[0].tolist())
                else:
                    list_names.append([names])
                    list_distances.append(distances[0].tolist())
            dict_info = {'id': list_names, 'distances': list_distances}
            return dict_info
        else:
            return None

    def add_1user(self, img, name):
        list_img = self.model_detect.detect(img)
        for img in list_img:
            feature = self.model_extract.extract(img)
            self.model_search.p, self.model_search.data, self.model_search.ids = self.model_search.add_user(feature, np.array(name))

    def add_many_user(self, folders):  # add user ban dau
        list_feature = []
        list_name = []
        folders = [folder for folder in glob.glob(folders + '/*')]
        for folder in folders:
            folder_name = os.path.basename(folder)
            images = [image for image in glob.glob(folder + '/*')]
            for image in images:
                img = cv2.imread(image)
                feature = self.model_extract.extract(img)
                for i in range(20):
                    list_feature.append(feature)
                    list_name.append(folder_name)
        list_feature = np.array(list_feature)
        list_name = np.array(list_name)
        self.model_search.p, self.model_search.data, self.model_search.ids = self.model_search.add_user(list_feature, list_name)

    def delete(self, name):
        self.p, self.data, self.ids = self.model_search.delete_user(name)
        self.model_search.p = load_p(self.data)

    def show(self):
        return {'shape data': self.model_search.data.shape, 'shape id': self.model_search.ids.shape, 'count number p': self.model_search.p.get_current_count()}


if __name__ == "__main__":
    model_face = FaceSearch(feature, name, 1.1)
    img = cv2.imread('/home/dungdinh/Documents/FaceNet-Infer/data_add_2/3/4.png')
    print(model_face.process(img))

