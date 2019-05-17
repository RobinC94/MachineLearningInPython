#!/usr/bin/python
# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
from kNN import KNN


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
            class_label = list_from_line[-1]
            if class_label == 'largeDoses':
                class_label = 3
            elif class_label == 'smallDoses':
                class_label = 2
            elif class_label == 'didntLike':
                class_label = 1
            else:
                class_label = int(class_label)
            class_label_list.append(class_label)
            index += 1
        return return_mat, class_label_list


if __name__ == '__main__':
    dating_data, dating_label = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data[:, 1], dating_data[:, 2], 15.0*np.array(dating_label), 15.0*np.array(dating_label))
    plt.show()
    date_knn = KNN(3)
    date_knn.fit(dating_data, dating_label)
    y = date_knn.predict([30000, 5, 1])
    print("Classification result: ", y)
    eval_data, eval_label = file2matrix('datingTestSet.txt')
    score = date_knn.evaluate(eval_data, eval_label)
    print("Evaluation result: %.2f%%" % (score * 100))
