import os
import time
import datetime
import pickle
import cv2
import numpy as np
from itertools import chain
from collections import namedtuple, deque
from decision_stumps import decision_stump


PATH_OBJECT_TRAINING = '../facesDB_ai_MIT/train/face/'
PATH_NON_OBJECT_TRAINING = '../facesDB_ai_MIT/train/non-face/'
PATH_OBJECT_TESTING = '../facesDB_ai_MIT/test/face/'
PATH_NON_OBJECT_TESTING = '../facesDB_ai_MIT/test/non-face/'


Region = namedtuple('Region', 'x y width height')


class ViolaJonesCascade:
    def __init__(self, samples=[1, 5, 10, 20], scaling=1.25):
        self.samples = samples
        self.scaling = scaling
        self.vjs = []

    def save_parameters(self, filename='mit_face_X'):
        with open(filename+'_vjs.pickle', 'wb') as f:
            pickle.dump(self.vjs, f)

    def load_parameters(self, filename='mit_face_X'):
        if os.path.isfile(filename + '_vjs.pickle'):
            with open(filename + '_vjs.pickle', 'rb') as f:
                self.vjs = pickle.load(f)

    def fit(self, images, y, filename='mit_face_X'):
        for i, cycle in enumerate(self.samples):
            vj = ViolaJones(cycle, self.scaling)
            vj.fit(images, y, filename + str(i))
            y_hat = np.array([vj.predict(image) for image in images])
            y_hat = np.sign(y_hat).astype(int)
            self.vjs.append(vj) # append latest classifier

            # break loop if the remaining sample are positive
            if (y[y_hat == 1] == 1).all():
                print('Fitting ends without non-object left')
                break

            # selecting data for next round
            images = images[y_hat == 1]
            y = y[y_hat == 1]

    def predict(self, image, is_integral_image=False):
        for i, vj in enumerate(self.vjs):
            if vj.predict(image, is_integral_image) < 0:
                return -1
        return 1

    def _predict(self, image, is_integral_image=False, filename='mit_face_X'):
        value = 0 
        for i, vj in enumerate(self.vjs):
            temp = vj.predict(image, is_integral_image)
            if temp < 0:
                return -1
            value += temp
        return value

    def nms(self, regions, weights, threshold=0.4):
        selected_frames = []

        frames = np.array(regions)
        weights = np.array(weights)
        area_frames = frames[:, 2] * frames[:, 3] # element-wise width * height
    
        # sort indices
        indices = np.array([(w, a) for w, a in zip(weights, area_frames)],
                           dtype=[('w', weights.dtype), ('a', area_frames.dtype)])
        indices = indices.argsort(order=['w', 'a'])

        while len(indices):
            index = indices[-1]
            main = regions[index]
            selected_frames.append(main)

            # intersection coordinates
            left = np.maximum(main.x, frames[indices, 0]) 
            right = np.minimum(main.x + main.width, 
                               frames[indices, 0] + frames[indices, 2]) 
            top = np.maximum(main.y, frames[indices, 1]) 
            bottom = np.minimum(main.y + main.height, 
                                frames[indices, 1] + frames[indices, 3]) 

            # intersections
            widths = np.maximum(0, right - left)
            heights = np.maximum(0, bottom - top)
            intersections = widths * heights

            # intersection over union (iou)
            area_unions = area_frames[index] + area_frames[indices] - intersections
            ious = intersections / area_unions

            # renew indices, add some additional criteria
            indices = indices[(ious < threshold) &\
                       (intersections < area_frames[indices] * threshold) &\
                       (intersections < area_frames[index] * threshold)]
        return selected_frames

    def hot_zones(self, regions, weights, height, width, threshold=0.5):
        # detect hot zones
        zones = np.zeros((height, width))
        for r, w in zip(regions, weights):
            zones[r.y: r.y+r.height+1, r.x: r.x+r.width+1] += w
        zones /= np.max(zones)

        # find boundary and make region
        hot_zones = []
        map_1s = zones >= threshold
        ones = {(r, c) for r, c in zip(*np.where(map_1s))}
        while ones:
            stack = {ones.pop()}
            left, right, top, bottom = width + 1, -1, height + 1, -1
            while stack:
                r, c = stack.pop()
                left, right, top, bottom = min(left, c), max(right, c),\
                                           min(top, r), max(bottom, r)
                if (r-1, c) in ones:
                    ones.discard((r-1, c))
                    stack.add((r-1, c))
                if (r+1, c) in ones:
                    ones.discard((r+1, c))
                    stack.add((r+1, c))
                if (r, c-1) in ones:
                    ones.discard((r, c-1))
                    stack.add((r, c-1))
                if (r, c+1) in ones:
                    ones.discard((r, c+1))
                    stack.add((r, c+1))
            hot_zones.append(Region(left, top, right-left, bottom-top))

        # zones on picture
        zones = (zones * 255).astype(np.uint8)

        return hot_zones, zones

    def face_detection(self, image_path, threshold=0.5, filename='mit_face_X'):
        # if parameter exist, load it
        if not self.vjs and os.path.isfile(filename + '_vjs.pickle'):
            with open(filename + '_vjs.pickle', 'rb') as f:
                self.vjs = pickle.load(f)

        # load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        vj = self.vjs[0]
        width, height, scaling = vj.width_training, vj.height_training, self.scaling
        img_h, img_w = image.shape

        scale = 1
        regions = []
        weights = []

        ii_whole = vj.integral_intensity(image)

        while width <= img_w and height <= img_h:
            for x in range(img_w - width + 1):
                for y in range(img_h - height + 1):
                    ii = ii_whole[y:y+height, x:x+width]
                    value = self._predict(ii, True)
                    if value > 0:
                        regions.append(Region(x, y, width, height))
                        weights.append(value)

            width = int(width * scaling)
            height = int(height * scaling)
            scale = int(scale * scaling)

        # nms, frame, and show framed picture
#        frames = self.nms(regions, weights, threshold)
        frames, zones = self.hot_zones(regions, weights, *image.shape, threshold)
        for f in frames:
            cv2.rectangle(image, (f.x, f.y), (f.x+f.width, f.y+f.height), (255, 255, 255), 1)

        cv2.imshow(' ', image)
        cv2.imshow('hot zone', zones)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return frames


class ViolaJones:
    def __init__(self, cycles=50, scaling=1.25):
        self.cycles = cycles
        self.scaling = scaling
        self.weights = None
        self.features = None
        self.stumps = None
        self.polarities = None
        self.width_training = None
        self.height_training = None

    def save_parameters(self):
        with open(filename+'.pickle', 'wb') as f:
            pickle.dump([self.weights, self.features, self.stumps, 
                         self.polarities, self.scaling, self.cycles,
                         self.height_training, self.width_training], f)
        print('---Viola-Jones parameters saved---')

    def load_parameters(self, filename='mit_face_X'):
        if os.path.isfile(filename + '.pickle'):
            with open(filename + '.pickle', 'rb') as f:
                self.weights, self.features, self.stumps, self.polarities,\
                self.scaling, self.cycles,  self.height_training,\
                self.width_training = pickle.load(f)
        print('---Viola-Jones parameters loaded---')

    @staticmethod
    def integral_intensity(image: np.ndarray):
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
                        current = Region(x, y, w, h)
                        right = Region(x+w, y, w, h)
                        right_2 = Region(x+2*w, y, w, h)
                        bottom = Region(x, y+h, w, h)
                        bottom_2 = Region(x, y+2*h, w, h)
                        bottom_right = Region(x+w, y+h, w, h)
                        
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

    def region_difference(self, integral_image, p_regions, n_regions):
        value_p = sum(self.get_intensity(g, integral_image) for g in p_regions)
        value_n = sum(self.get_intensity(g, integral_image) for g in n_regions)
#        value = value_p - value_n
        value = (value_p - value_n) / (value_p + value_n + 10**(-12))
        return value

    def apply_features(self, integral_images, features, filename='mit_face_X'): 
        # this is key to help find difference
        X = np.zeros((len(integral_images), len(features)))
        for r, ii in enumerate(integral_images):
            for c, (p_regions, n_regions) in enumerate(features):
                X[r, c] = self.region_difference(ii, p_regions, n_regions)
        return X

    def adaboost(self, X, y, features):
        # use [] instead of np.empty is to avoid loop stop before reaching cycles
        self.weights = []
        self.features = []
        self.stumps = []
        self.polarities = []

#        sample_weights = np.full(X.shape[0], 1/X.shape[0])
        n_positives = sum(y > 0)
        n_negatives = len(y) - n_positives
        sample_weights = np.empty(len(y))
        sample_weights[y > 0] = 1 / n_positives
        sample_weights[y < 0] = 1 / n_negatives
        for i in range(self.cycles):
            print(f'(In adaboost) Cycle: {i}. Get into decision stump.')
            t_start = datetime.datetime.now()
            index_f, stump, polarity = decision_stump(X, y, sample_weights)
            print(f'(In adaboost) Finish {i} round decision stump. Lapse: {datetime.datetime.now() - t_start}')
            hypothesis = np.where(X[:, index_f] < stump, polarity, -polarity)
            error = sum(sample_weights[hypothesis != y])

            weight = np.log(1/(error+10**(-10)) - 1) * 0.5
            sample_weights *= np.exp(-weight * y * hypothesis)
            sample_weights /= sum(sample_weights)

            # append the best feature each loop
            self.weights.append(weight)
            self.features.append(features[index_f])
            self.stumps.append(stump)
            self.polarities.append(polarity)

            if not error:
                print(f'   Perfect match. Adaboost stops at round {i}')
                break

    def fit(self, images, y, filename='mit_face_X', save_dataset=True):
        print('Computing integral images')
        iis = [self.integral_intensity(i) for i in images]
        self.height_training, self.width_training = iis[0].shape
        print('Complete integral images')
        print('Making features')
        features = self.make_features(*(iis[0].shape))
        print(f'Complete features. Total features: {len(features)}')
        # load X from file if exist, or calculate it from scratch
        save_X_name = filename + '_data.npy'
        if os.path.isfile(save_X_name):
            print('Loading X from file')
            X = np.fromfile(save_X_name).reshape(len(iis), -1)
        else:
            print(f'Applying features to integral images to make X. Current time: {datetime.datetime.now().time()}')
            X = self.apply_features(iis, features, save_X_name)
            print(f'Complete X. Current time: {datetime.datetime.now().time()}')
            if save_dataset:
                X.tofile(save_X_name)
                print(f'Save X to file {filename}')
        print('Ready for adaboost')
        self.adaboost(X, y, features)

    def predict(self, image, is_integral_image=False):
        predict = 0
        for w, f, s, p in zip(self.weights, self.features, self.stumps, self.polarities):
            ii = image if is_integral_image else self.integral_intensity(image)
            value = self.region_difference(ii, f[0], f[1])
            predict += w * np.where(value < s, p, -p)
        return predict
#        return np.sign(predict)


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
    print('Loading pgm files')
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
    print('pgm files loading finish')
    images = np.array(images)
    return images, y


if __name__ == '__main__':
    # dataset
#    images_training, y_training = load_pgm()
#    images_testing, y_testing = load_pgm(PATH_OBJECT_TESTING, PATH_NON_OBJECT_TESTING)

    # test of Viola-Jones cascade
    vj_cascade = ViolaJonesCascade([1, 2, 8, 16, 30])
#    vj_cascade.fit(images_training, y_training)
#    vj_cascade.save_parameters()
    vj_cascade.load_parameters()
#    y_hat_cascade = np.array([vj_cascade.predict(i) for i in images_training])
#    error_cascade = sum(y_hat_cascade != y_training) / len(y_training)
#    print(f'Error of cascade: {error_cascade * 100:.2f}%')
#    y_hat_cascade = np.array([vj_cascade.predict(i) for i in images_testing])
#    error_cascade = sum(y_hat_cascade != y_testing) / len(y_testing)
#    print(f'Error of cascade: {error_cascade * 100:.2f}%')
    face = '09'
    threshold = 0.4
#    vj_cascade.face_detection(f'./face{face}.jpg', threshold)
    vj_cascade.face_detection(f'./face{face}.png', threshold)
