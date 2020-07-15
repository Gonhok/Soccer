import numpy as np
import torch
from collections import deque
import torchvision
from scipy.optimize import linear_sum_assignment

class Tracklet(object):
    last_ID = 0

    def __init__(self, bounding_box, feature):
        self.ID = Tracklet.last_ID
        Tracklet.last_ID += 1

        self.states = np.zeros((0, 4))
        state = np.asarray(bounding_box).reshape((1, 4))
        self.states = np.concatenate((self.states, state), axis=0)

        self.cur_feature = feature
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.no_tracking_count = 0

    def update(self, bounding_box, feature):

        state = np.asarray(bounding_box).reshape((1, 4))
        self.states = np.concatenate((self.states, state), axis=0)
        self.cur_feature = feature

        if len(self.states) > 60:
            self.states = np.delete(self.states, obj=0, axis=0)

        self.estimation_model()

    def estimation_model(self):
        self.vel_x = (self.states[-1, 0] + self.states[-1, 2]/2.0) - (self.states[-2, 0] + self.states[-2, 2]/2.0)
        self.vel_y = (self.states[-1, 1] + self.states[-1, 3]/2.0) - (self.states[-2, 1] + self.states[-2, 3]/2.0)


class Tracker(object):
    def __init__(self):
        self.active = False
        self.tracklets = []

    def forward(self, detector_bb, feature_map):

        if not self.active:
            features = self.extract_features(bounding_box=detector_bb, feature_map=feature_map)
            for i in range(detector_bb):
                tracklet = Tracklet(detector_bb[i, :4], features[i, :])
                self.tracklets.append(tracklet)

        else:
            candidates = self.generate_candidates()
            union_bb= np.concatenate((detector_bb, candidates), axis=0)
            features = self.extract_features(bounding_box=union_bb, feature_map=feature_map)

            score_matrix = self.construct_score_matrix(features, bounding_box=union_bb)
            self.matching(score_matrix, bounding_box=union_bb, detector_bb=detector_bb, features=features)

    def generate_candidates(self, num=50):
        candidates = np.zeros((1, 4))

        for tracklet in self.tracklets:
            x_ = (tracklet.states[-1, 0] + tracklet.vel_x + np.random.normal(0.0, 1.0, num)).reshape((num, 1))
            y_ = (tracklet.states[-1, 1] + tracklet.vel_y + np.random.normal(0.0, 1.0, num)).reshape((num, 1))
            w_ = np.mean(tracklet.states[-4:, 2]) + np.zeros((num, 1))
            h_ = np.mean(tracklet.states[-4:, 3]) + np.zeros((num, 1))

            bb = np.concatenate((x_, y_, w_, h_), axis=1)
            candidates = np.concatenate((candidates, bb), axis=0)

        return candidates

    def construct_score_matrix(self, features, bounding_box):

        m = len(self.tracklets)
        n = bounding_box.shape[0]

        cost_a = np.zeros((m, n))
        cost_s = np.zeros((m, n))
        cost_m = np.zeros((m, n))

        for i in range(m):
            cost_a[i, :] = np.sum(self.tracklets[i].cur_feature[None, :] * features, axis=1)
            cost_s[i, :] = np.exp(-((self.tracklets[i].states[2] - bounding_box[:, 2])/(self.tracklets[i].states[2] + bounding_box[:, 2])
                                    + (self.tracklets[i].states[3] - bounding_box[:, 3])/(self.tracklets[i].states[3] + bounding_box[:, 3])) ** 2)
            dist = (bounding_box[:, 0] - self.tracklets[i].states[0] - self.tracklets[i].vel_x) ** 2 \
                   + (bounding_box[:, 1] - self.tracklets[i].states[1] - self.tracklets[i].vel_y) ** 2
            cost_m[i, :] = np.exp(-(dist))

        return (cost_a + cost_s + cost_m)/3.0

    def matching(self, score_matrix, bounding_box, detector_bb, features):
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        mask = score_matrix[row_ind, col_ind] < 0.3

        for i in range(len(col_ind)):
            if (mask):
                self.tracklets[row_ind[i]].update(bounding_box[col_ind[i], :4], features[col_ind[i]])
                self.tracklets[row_ind[i]].no_tracking_count = 0
            else:
                self.tracklets[row_ind[i]].no_tracking_coung += 1

        ### If detector bounding boxes are not machted, new tracklet will be initizliazed.
        no_detector_matching = set(range(len(detector_bb)))
        no_detector_matching = no_detector_matching - set(col_ind[~mask])

        for idx in no_detector_matching:
            tracklet = Tracklet(detector_bb[idx, :4], features[idx, :])
            self.tracklets.append(tracklet)

    def extract_features(self, bounding_box, feature_map, img_size=(1080, 1920)):
        n, c, h, w = feature_map.shape
        spatial_ratio = img_size / h

        features = torchvision.ops.roi_align(input=feature_map, boxes=bounding_box, output_size=(16, 16), spatial_scale=spatial_ratio,
                                             sampling_ratio=-1)
        K = features.shape[0]
        features = features.reshape((K, -1))
        features /= (np.linalg.norm(features, axis=1)).reshape((K, 1))
        return features