import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from helpers import *
import pickle


class HOGClassifier():

    def __init__(self, dataset):
        self.color_space = 'HLS'
        self.spatial_size = (16, 16)
        self.hist_bins = 16
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 'ALL'
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True

        self.dataset = dataset

    def extract_data_features(self):
        t=time.time()

        self.car_features = extract_features(self.dataset.cars,
                color_space=self.color_space,
                orient=self.orient,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hog_channel=self.hog_channel,
                spatial_feat=self.spatial_feat,
                hist_feat=self.hist_feat,
                hog_feat=self.hog_feat)

        self.non_car_features = extract_features(self.dataset.non_cars,
                color_space=self.color_space,
                orient=self.orient,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hog_channel=self.hog_channel,
                spatial_feat=self.spatial_feat,
                hist_feat=self.hist_feat,
                hog_feat=self.hog_feat)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract HOG features...')
        self.X = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        print('Feature vectors shape:',self.X.shape)

    def scale_features(self):
        self.X_scaler = StandardScaler().fit(self.X)
        self.scaled_X = self.X_scaler.transform(self.X)

    def get_labels(self):
        self.Y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))

    def split_up_data(self):
        # rand_state = np.random.randint(0, 100)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            # self.scaled_X, self.Y, test_size=0.2, random_state=rand_state)

        car_split    = len(self.car_features) * 0.2
        notcar_split = len(self.non_car_features) * 0.2

        self.X_test = np.vstack((self.scaled_X[:int(car_split)],
            self.scaled_X[len(car_features):(len(car_features) + int(notcar_split))]))

        self.y_test = np.hstack((self.Y[:int(car_split)],
            self.Y[len(self.car_features):(len(self.car_features) + int(notcar_split))]))

        self.X_train = np.vstack((self.scaled_X[int(car_split):len(self.car_features)],
            self.scaled_X[(len(self.car_features) + int(notcar_split)):]))

        self.y_train = np.hstack((self.Y[int(car_split):len(self.car_features)],
            self.Y[(len(self.car_features) + int(notcar_split)):]))

        print('Using:',self.orient,'orientations',self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(self.X_train[0]))

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')

        print(self.X_train.shape, 'X_train.shape')
        print(self.y_train.shape, 'y_train.shape')
        print(self.X_test.shape, 'X_test')
        print(self.y_test.shape, 'y_test shape')
        print(self.scaled_X.shape, 'scaled_X shape')
        print(self.Y.shape, 'y  shape')

    def train(self):
        svc = LinearSVC()
        t=time.time()

        svc.fit(self.X_train, self.y_train)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')

        self.svc = svc

    def accuracy(self):
        print('Test Accuracy of SVC = ', round(self.svc.score(self.X_test, self.y_test), 4))

    def predict(self, img):
        print('SVC predicts: ', self.svc.predict(img))

    def save_data(self):
        print('Saving data to pickle file...')
        try:
            with open('processed_svc.p', 'wb') as pfile:
                pickle.dump(
                    {   'svc':self.svc,
                        'X_scaler': self.X_scaler
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', FILE, ':', e)
            raise

        print('Data cached in pickle file.')

    def load_model(self):
        self.svc = pickle.load(open(FILE,"rb"))

