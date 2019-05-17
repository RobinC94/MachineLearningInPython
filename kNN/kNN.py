#!/usr/bin/python
# -*-coding:utf-8-*-

import os, sys
import operator
import numpy as np
import matplotlib.pyplot as plt


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
        data_size = self.datas.shape[0]
        diff_mat = np.tile(in_x, (data_size, 1)) - self.datas
        distance = (diff_mat**2).sum(axis=1)
        sorted_dis_indicies = distance.argsort()
        class_count = {}
        for i in range(self.k):
            vote_label = self.labels[sorted_dis_indicies[i]]
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
            print(class_count)
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    @staticmethod
    def norm_dataset(dataset):
        min_vals = dataset.min(0)
        max_vals = dataset.max(0)
        ranges = max_vals - min_vals
        m = dataset.shape[0]
        norm_data = dataset - np.tile(min_vals, (m,1))
        norm_data = norm_data/np.tile(ranges, (m,1))
        return norm_data, ranges, min_vals


def file2matrix(filename):
    with open(filename, 'r') as fr:
        array_of_lines = fr.readlines()
        num_of_lines = len(array_of_lines)
        return_mat = np.zeros((num_of_lines, 3))
        class_label_list = []
        index = 0
        for line in array_of_lines:
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:-1]
            class_label_list.append(int(list_from_line[-1]))
            index += 1
        return return_mat, class_label_list


if __name__ == '__main__':
    dating_data, dating_label = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data[:, 1], dating_data[:, 2], 15.0*np.array(dating_label), 15.0*np.array(dating_label))
    plt.show()
    date_knn = KNN(30)
    date_knn.fit(dating_data, dating_label)
    print(date_knn.predict([30000,5,1]))
