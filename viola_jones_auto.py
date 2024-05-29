import os
import time
import datetime
import pickle
import cv2
import numpy as np
from itertools import chain
from collections import namedtuple, deque
from decision_stumps import decision_stump


PATH_OBJECT_TRAINING = './facesDB_ai_MIT/train/face/'
PATH_NON_OBJECT_TRAINING = './facesDB_ai_MIT/train/non-face/'
PATH_OBJECT_TESTING = './facesDB_ai_MIT/test/face/'
PATH_NON_OBJECT_TESTING = './facesDB_ai_MIT/test/non-face/'


Region = namedtuple('Region', 'x y width height')


class ViolaJonesGeneral:
    """
        Controlling coefficient 'a' in exp(-ax) to determine target false-
        positive rate(FP rate) and false-negative rate(FN rate) in each stage.
        Notice that the FP rate and FN rate are not the overall counterparts,
        so one may need to set lower target in order to get the accurate result.
    """

    def __init__(self, false_positive_factor=2, false_negative_factor=2,
                 false_positive_rate=0.000001, false_negative_rate=0.000001, 
                 maximum_stages=10, scaling=1.25):
        # false positive or negative related
        self.fp_factor = false_positive_factor # coefficient in exponential
        self.fn_factor = false_negative_factor # coefficient in exponential
        self.fp_rate = false_positive_rate # FP/FN rate at the last stage 
        self.fn_rate = false_negative_rate # not the overall FP/FN rate

        # fitting parameters from AdaBoost
        self.weights = []
        self.features = []
        self.stumps = []
        self.polarities = []

        # training images' shape
        self.width_training = None
        self.height_training = None

        self.max_stages = maximum_stages # max levels if fp/fn rate don't reach
        self.scaling = scaling

    @staticmethod
    def rate_target(factor, stage):
        return np.exp(-factor * (stage + 1))

    def adaboostForVJ(self, X, y, sample_weights=None):
        if sample_weights is None:
#             sample_weights = np.ones(len(y)) / len(y)
            n_positives = sum(y > 0)
            n_negatives = len(y) - n_positives
            sample_weights = np.empty(len(y))
            sample_weights[y > 0] = 1 / n_positives
            sample_weights[y < 0] = 1 / n_negatives

        index_f, stump, polarity = decision_stump(X, y, sample_weights)
        hypothesis = np.where(X[:, index_f] < stump, polarity, -polarity)

        error = sum(sample_weights[hypothesis != y])
        weight = np.log(1 / (error + 10**(-10)) - 1) * 0.5
        sample_weights *= np.exp(-hypothesis * weight * y)
        sample_weights /= sum(sample_weights)

        return index_f, stump, polarity, hypothesis, weight, sample_weights

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
    def get_intensity(region: Region, integral_image, scale=1):
        """
            Get the sum of intensity of a region in integral image.
        """

        x, y, w, h = int(region.x * scale), int(region.y * scale),\
                     int(region.width * scale), int(region.height * scale)

        point_a = integral_image[y+h, x+w]
        point_b = integral_image[y, x]
        point_c = integral_image[y+h, x]
        point_d = integral_image[y, x+w]
        
        intensity = point_a + point_b - point_c - point_d
        return intensity

    def region_difference(self, integral_image, p_regions, n_regions, scale=1):
        value_p = sum(self.get_intensity(g, integral_image, scale)\
                      for g in p_regions)
        value_n = sum(self.get_intensity(g, integral_image, scale)\
                      for g in n_regions)
        value = (value_p - value_n) / (value_p + value_n + 10**(-12))
        return value

    def apply_features(self, integral_images, features, filename='mit_face_X'): 
        # this is key to help find difference
        X = np.zeros((len(integral_images), len(features)))
        for r, ii in enumerate(integral_images):
            for c, (p_regions, n_regions) in enumerate(features):
                X[r, c] = self.region_difference(ii, p_regions, n_regions)
        return X

    def fit(self, images, y, filename='mit_face_X', save_dataset=True):
        t_start_overall = datetime.datetime.now()

        print('Computing integral images')
        iis = [self.integral_intensity(i) for i in images]
        self.height_training, self.width_training = iis[0].shape
        print('Making features')
        features = self.make_features(*(iis[0].shape))
        print(f'Complete features. Total features: {len(features)}')

        # load X from file if exist, or calculate it from scratch
        save_X_name = filename + '_data.npy'
        if os.path.isfile(save_X_name):
            print('Loading X from file')
            X = np.fromfile(save_X_name).reshape(len(iis), -1) 
        else:
            print(f'Applying features to integral images to make X.')
            X = self.apply_features(iis, features, save_X_name)
            print(f'X accomplished')
            if save_dataset:
                X.tofile(save_X_name)
                print(f'Save X to file {filename}')

        # fitting
        fp_loop, fn_loop = 1, 1 # fp/fn rate in each loop. 1 is 100%
        for i in range(self.max_stages):
            if fp_loop <= self.fp_rate and fn_loop <= self.fn_rate:
                break # meet rate criteria, stop fitting
    
            # target rates at each stage
            fp_stage = max(self.rate_target(self.fp_factor, i), self.fp_rate)
            fn_stage = max(self.rate_target(self.fn_factor, i), self.fn_rate)
            print(f'Final target FP rate / FN rate: {self.fp_rate * 100:.6f}%' +\
                  f' / {self.fn_rate * 100:.6f}%. Current stage target rates: ' +\
                  f'{fp_stage * 100:.6f}% / {fn_stage * 100:.6f}%')

            # sometimes last time fp_loop / fn_loop < current fp_stage / fn_stage
            if fp_loop <= fp_stage and fn_loop <= fn_stage:
                continue

            # for new stage
            self.weights.append([])
            self.features.append([])
            self.stumps.append([])
            self.polarities.append([])

            j = 0
            y_hat = np.zeros(len(y))
            sample_weights = None
            while fp_loop > fp_stage or fn_loop > fn_stage:
                print(f'Stage {i} loop {j} fitting.')
                t_start = datetime.datetime.now()

                index_f, stump, polarity, hypothesis, weight, sample_weights =\
                self.adaboostForVJ(X, y, sample_weights)

                # append fitting parameters
                self.weights[-1].append(weight)
                self.features[-1].append(features[index_f])
                self.stumps[-1].append(stump)
                self.polarities[-1].append(polarity)

                # predict and renew fp_loop and fn_loop
                y_hat += weight * hypothesis
                fp = len(y_hat[(y_hat > 0) & (y < 0)])
                fn = len(y_hat[(y_hat < 0) & (y > 0)])
                fp_loop = fp / len(y)
                fn_loop = fp / len(y)

                t_end = datetime.datetime.now()
                print(f'Current FP rate/ FN rate: {fp_loop * 100:.6f}% / ' +\
                      f'{fn_loop * 100:.6f}% (lapse {t_end - t_start})')

                j += 1

            # renew training X and y
            X = X[(y_hat > 0)]
            y = y[(y_hat > 0)]

            print(f'Stage {i} has {len(self.weights[-1])} loops')

        t_end_overall = datetime.datetime.now()

        print(f'Final FP/FN rate: {fp_loop} / {fn_loop}. Lapse {t_end_overall - t_start_overall}. Loops: {len(self.weights[-1])}.')

    def _predict(self, image, is_integral_image=False, scale=1):
        predict = 0 
        ii = image if is_integral_image else self.integral_intensity(image)
        for weight, feature, stump, polarity in zip(self.weights, self.features,\
            self.stumps, self.polarities):
            value = 0
            for w, f, s, p in zip(weight, feature, stump, polarity):
                temp = self.region_difference(ii, f[0], f[1], scale)
                value += w * np.where(temp < s, p, -p)
            if value < 0:
                return -1
            predict += value
        return predict

    def predict(self, image, is_integral_image=False, scale=1):
        return int(np.sign(self._predict(image, is_integral_image, scale)))

    def save_parameters(self, filename='mit_face_X'):
        parameters = self.fp_factor, self.fn_factor, self.fp_rate, self.fn_rate,\
                     self.weights, self.features, self.stumps, self.polarities,\
                     self.width_training, self.height_training,\
                     self.max_stages, self.scaling

        with open(filename+'_parameters.pickle', 'wb') as f:
            pickle.dump(parameters, f)

        print('Viola-Jones parameters saved')

    def load_parameters(self, filename='mit_face_X'):
        if os.path.isfile(filename + '_parameters.pickle'):
            with open(filename + '_parameters.pickle', 'rb') as f:
                self.fp_factor, self.fn_factor, self.fp_rate, self.fn_rate,\
                self.weights, self.features, self.stumps, self.polarities,\
                self.width_training, self.height_training,\
                self.max_stages, self.scaling = pickle.load(f)

        print('Viola-Jones parameters loaded')

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
            zones[r.y: r.y + r.height + 1, r.x: r.x + r.width + 1] += w
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
        if not self.weights:
            self.load_parameters(filename)

        # load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        width, height, scaling = self.width_training, self.height_training, self.scaling
        img_h, img_w = image.shape

        scale = 1
        regions = []
        weights = []

        ii_whole = self.integral_intensity(image)

        while width <= img_w and height <= img_h:
            for x in range(img_w - width + 1):
                for y in range(img_h - height + 1):
                    ii = ii_whole[y: y + height, x: x + width]
                    value = self._predict(ii, True, scale)
                    if value > 0:
                        regions.append(Region(x, y, width, height))
                        weights.append(value)

            width = int(width * scaling + 1)
            height = int(height * scaling + 1)
            scale = scale * scaling

        # nms, frame, and show framed picture
#        frames = self.nms(regions, weights, threshold)
        frames, zones = self.hot_zones(regions, weights, *image.shape, threshold)
        for f in frames:
            cv2.rectangle(image, (f.x, f.y), (f.x+f.width, f.y+f.height), (255, 255, 255), 1)

        cv2.imshow(' ', image)
#        cv2.imshow('hot zone', zones)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return frames


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
    vj = ViolaJonesGeneral()
#    vj.fit(images_training, y_training)
#    vj.save_parameters()
    vj.load_parameters()
#    y_hat_cascade = np.array([vj.predict(i) for i in images_training])
#    error_cascade = sum(y_hat_cascade != y_training) / len(y_training)
#    print(f'Error of cascade: {error_cascade * 100:.2f}%')
#    y_hat_cascade = np.array([vj.predict(i) for i in images_testing])
#    error_cascade = sum(y_hat_cascade != y_testing) / len(y_testing)
#    print(f'Error of cascade: {error_cascade * 100:.2f}%')

    face = '01'
    threshold = 0.4
    vj.face_detection(f'./face{face}.jpeg', threshold)
#    vj.face_detection(f'./face{face}.png', threshold)
