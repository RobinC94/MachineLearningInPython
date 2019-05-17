#!/usr/bin/python
# -*-coding:utf-8-*-

import os, sys
import numpy as np
from collections import Counter


class KNN(object):
    def __init__(self, k):
        assert k >= 1, "k must be valid"
        self.k = k
        self.datas = None
        self.labels = None
        self.range = 1.0
        self.min_vals = 0.0

    def fit(self, datas, labels):
        self.labels = labels
        self.datas, self.range, self.min_vals = self.norm_dataset(datas)

    def __repr__(self):
        return 'knn(k=%d):' % self.k

    def predict(self, in_x):
        in_x = (in_x - self.min_vals)/self.range
        distance = [np.sqrt(np.sum((in_x - x) ** 2)) for x in self.datas]
        nearest = np.argsort(distance)
        top_k = [self.labels[i] for i in nearest[:self.k]]
        votes = Counter(top_k)
        return votes.most_common(1)[0][0]

    @staticmethod
    def norm_dataset(dataset):
        min_vals = dataset.min(0)
        max_vals = dataset.max(0)
        ranges = max_vals - min_vals
        m = dataset.shape[0]
        norm_data = dataset - np.tile(min_vals, (m,1))
        norm_data = norm_data/np.tile(ranges, (m,1))
        return norm_data, ranges, min_vals

    def evaluate(self, eval_datas, eval_labels):
        eval_predict = [self.predict(x) for x in eval_datas]
        eval_predict = np.array(eval_predict)
        eval_labels = np.array(eval_labels)
        return sum(eval_predict == eval_labels)/eval_datas.shape[0]
