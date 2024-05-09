import os
import time
import datetime
import numpy as np
from itertools import chain
from collections import namedtuple
from decision_stumps import decision_stump


PATH_OBJECT_TRAINING = './facesDB_ai_MIT/train/face/'
PATH_NON_OBJECT_TRAINING = './facesDB_ai_MIT/train/non-face/'
PATH_OBJECT_TESTING = './facesDB_ai_MIT/test/face/'
PATH_NON_OBJECT_TESTING = './facesDB_ai_MIT/test/non-face/'

#PATH_OBJECT_TESTING = './facesDB_ai_MIT/train/face/'
#PATH_NON_OBJECT_TESTING = './facesDB_ai_MIT/train/non-face/'
#PATH_OBJECT_TRAINING = './facesDB_ai_MIT/test/face/'
#PATH_NON_OBJECT_TRAINING = './facesDB_ai_MIT/test/non-face/'


class ViolaJones:
    Region = namedtuple('Region', 'x y width height')

    def __init__(self, cycles=50):
        self.cycles = cycles
        self.weights = None
        self.features = None
        self.stumps = None
        self.polarities = None

    @staticmethod
    def integral_intensity(image: np.ndarray):
    #    row, col = image.shape
    #    ii = np.zeros((row, col), dtype=int)
    #    for c in range(col):
    #        s = 0
    #        ii[:, c] = ii[:, c-1] if c else ii[:, 0]
    #        for r in range(row):
    #            s += image[r, c]
    #            ii[r, c] += s
        """
            Get image of integral intensity from original image.
        """

        integral_image = image.copy()
        integral_image = np.cumsum(integral_image, 0)
        integral_image = np.cumsum(integral_image, 1)
        return integral_image

    def make_features(self, height, width):
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                x = 0
                while x + w < width:
                    y = 0
                    while y + h < height:
                        # make all possible regions
                        current = self.Region(x, y, w, h)
                        right = self.Region(x+w, y, w, h)
                        right_2 = self.Region(x+2*w, y, w, h)
                        bottom = self.Region(x, y+h, w, h)
                        bottom_2 = self.Region(x, y+2*h, w, h)
                        bottom_right = self.Region(x+w, y+h, w, h)
                        
                        # save valid regions
                        if x + 2*w < width: # horizontal window
                            features.append(([right], [current]))
                            if x + 3*w < width:
                                features.append(([right], [current, right_2]))

                        if y + 2*h < height: # vertical window
                            features.append(([current], [bottom]))
                            if y + 3*h < height:
                                features.append(([bottom], [current, bottom_2]))

                        if (x + 2*w < width) and (y + 2*h < height): # 2x2 window
                            features.append(([right, bottom], [current, bottom_right]))
                        y += 1
                    x += 1
        return features

    @staticmethod
    def get_intensity(region: Region, integral_image):
        """
            Get the sum of intensity of a region in integral image.
        """

        x, y, w, h = region.x, region.y, region.width, region.height
        point_a = integral_image[y+h, x+w]
        point_b = integral_image[y, x]
        point_c = integral_image[y+h, x]
        point_d = integral_image[y, x+w]
        
        intensity = point_a + point_b - point_c - point_d
        return intensity

    def apply_features(self, integral_images, features, filename='mit_face_X'): 
        X = np.zeros((len(integral_images), len(features)))
        for r, ii in enumerate(integral_images):
            for c, (regions_p, regions_n) in enumerate(features):
                value_p = sum(self.get_intensity(g, ii) for g in regions_p)
                value_n = sum(self.get_intensity(g, ii) for g in regions_n)
                X[r, c] = value_p - value_n
        np.save(filename, X)
        return X

    def adaboost(self, X, y, features):
        # use [] instead of np.empty is to avoid loop stop before reaching cycles
        self.weights = []
        self.features = []
        self.stumps = []
        self.polarities = []

        sample_weights = np.full(X.shape[0], 1/X.shape[0])
        for i in range(self.cycles):
            print(f'(In adaboost) Cycle: {i}. Get into decision stump.')
            index_f, stump, polarity = decision_stump(X, y, sample_weights)
#            print(f'(In adaboost) sample_weights: {sample_weights}')
            print(f'(In adaboost) No. {index_f} feature is chosen')
            print(f'(In adaboost) Finish {i} round decision stump')
            hypothesis = np.where(X[:, index_f] < stump, polarity, -polarity)
            error = sum(sample_weights[hypothesis != y])
#            print(f'(In adaboost) y: {y}')
#            print(f'(In adaboost) y_hat: {hypothesis}')

            weight = np.log(1/(error+10**(-10)) - 1) * 0.5
            sample_weights *= np.exp(-weight * y * hypothesis)
            sample_weights /= sum(sample_weights)

            # append the best feature each loop
            self.weights.append(weight)
            self.features.append(features[index_f])
            self.stumps.append(stump)
            self.polarities.append(polarity)

            if not error:
                print(f'---adaboost stops at round {i}')
                break

    def fit(self, images, y, filename='mit_face_X'):
        filename += '.npy'
        print('Computing integral images')
        iis = [self.integral_intensity(i) for i in images]
        print(f'Total training data: {len(iis)}')
        print('Complete integral images')
        print('Making features')
        features = self.make_features(*(iis[0].shape))
        print(f'Total features: {len(features)}')
        print('Complete features')
        print(f'Applying features to integral images to make X. Current time: {datetime.datetime.now().time()}')
        # load X from file if exist, else calculate it from scratch
        if os.path.isfile(filename):
            print('Loading X from file')
            X = np.fromfile(filename).reshape(len(iis), -1)
            print(f'---X shape: {X.shape}')
        else:
            X = self.apply_features(iis, features, filename)
            X.tofile(filename)
            print(f'Save X to file {filename}')
        print(f'Complete X. Current time: {datetime.datetime.now().time()}')
        print('Ready for adaboost')
        self.adaboost(X, y, features)

    def predict(self, images):
        predict = np.zeros(len(images))
        for i, image in enumerate(images):
            for w, f, s, p in zip(self.weights, self.features, self.stumps, self.polarities):
                ii = self.integral_intensity(image)
                value_p = sum(self.get_intensity(g, ii) for g in f[0])
                value_n = sum(self.get_intensity(g, ii) for g in f[1])
                value = value_p - value_n
                predict[i] += w * np.where(value < s, p, -p)
        return np.where(np.sign(predict) > 0, 1, -1)

def load_pgm(object_path=PATH_OBJECT_TRAINING, non_object_path=PATH_NON_OBJECT_TRAINING):
    """
        Load pgm extension files from which the Viola-Jones algorithm learns.
        
        returns:
          images: list of np.ndarray, each np.ndarray is the image's raw data.
          y: np.ndarray, the label of each image in images.
    """

    # file paths from object folder and non-object folder
    obj_files = [object_path + p for p in os.listdir(object_path)]
    non_obj_files = [non_object_path + p for p in os.listdir(non_object_path)]

    # build labels in {1, -1}
    y = np.ones(len(obj_files) + len(non_obj_files), dtype=int)
    y[len(obj_files):] = -1

    # save images from raw data to list of np.ndarray
    images = []
    print('Loading training files')
    for file in chain(obj_files, non_obj_files):
        with open(file, 'rb') as f:
            f.readline()
            width, height = map(lambda x: int(x), f.readline().split())
            f.readline()

            pic = np.zeros((height, width), dtype=int)
            for r in range(height):
                for c in range(width):
                    pic[r, c] = ord(f.read(1))
        images.append(pic)
    print('Files loading finish')
    return images, y


if __name__ == '__main__':
    start_time = time.time()

    # training set
    images_training, y_training = load_pgm()
    vj = ViolaJones(10)
    vj.fit(images_training, y_training, filename='mit_face_X')
    y_train_hat = vj.predict(images_training)

    # testing set
    images_testing, y_testing = load_pgm(PATH_OBJECT_TESTING, PATH_NON_OBJECT_TESTING)
    y_test_hat = vj.predict(images_testing)

    print('error of trainging set:', sum(y_train_hat != y_training) / len(y_training) * 100, '%')
    print('error of testing set:', sum(y_test_hat != y_testing) / len(y_testing) * 100, '%')

    print(f'End time: {datetime.datetime.now().time()}')
    print(datetime.timedelta(seconds=time.time()-start_time))
