# @Time : 2021/4/10 10:23
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : MILTool.py
# @Time : 2020/12/4 11:01
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : MILT.py
import os
import sys
import time

import numpy as np
import scipy.io as scio
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MILTool:
    def __init__(self, para_file_name, bags=None):
        self.file_name = para_file_name
        if bags is not None:
            self.bags = bags
        else:
            self.bags = self.__load_file(self.file_name)
        self.bags_label = self.__bag_label(self.bags)
        self.instance_space = ins_space(self.bags)

    def get_dis_data(self, para_file_name, para_data):
        path = 'D:/data/distance/' + para_file_name[:-4] + '_distance.csv'
        if os.path.isfile(path) != 1:
            dis = cdist(para_data, para_data)
            np.savetxt(path, dis, delimiter=',')
        else:
            dis = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
        return dis

    def __load_file(self, para_path):
        """
        Load file.
        @param:
        ------------
            para_file_name: the path of the given file.
        ------------
        @return:
        ------------
            The data.
        ------------
        """
        temp_type = para_path.split('.')[-1]

        if temp_type == 'mat':
            ret_data = scio.loadmat(para_path)
            return ret_data['data']
        else:
            with open(para_path) as temp_fd:
                ret_data = temp_fd.readlines()

            return ret_data

    def get_classifier(self, classifier_model):
        '''
        Get the classifier model you want.
        :param classifier_model: The classifier Model.
        :return:
        '''
        if classifier_model == 'knn':
            return KNeighborsClassifier(n_neighbors=3)
        elif classifier_model == 'DTree':
            return DecisionTreeClassifier()
        elif classifier_model == 'linear_svm':
            return SVC(kernel='linear')
        elif classifier_model == 'rbf_svm':
            return SVC(kernel='rbf')
        else:
            return None

    def __bag_label(self, bags):
        """
        Get the bags label.
        :param para_bags:
        :return:
        """
        temp_bag_lab = bags[:, -1]
        return np.array([list(val)[0][0] for val in temp_bag_lab])


def ins_space(para_bags):
    """
    Get the original instance space.
    :param para_bags:the original bags.
    :return: The instance space.
    """
    ins_space = []

    if para_bags.shape[0] == 1:
        for ins in para_bags[0][:, :-1]:
            ins_space.append(ins)
    else:
        for i in range(para_bags.shape[0]):
            for ins in para_bags[i, 0][:, :-1]:
                ins_space.append(ins)
    return np.array(ins_space)


def get_bar(i, j):
    k = i * 10 + j + 1
    str = '>' * ((j + 10 * i) // 2) + ' ' * ((100 - k) // 2)
    sys.stdout.write('\r' + str + '[%s%%]' % (i * 10 + j + 1))
    sys.stdout.flush()
    time.sleep(0.0001)


def get_ten_fold_index(bags):
    """
    Get the training set index and test set index.
    @param
        para_k:
            The number of k-th fold.
    :return
        ret_tr_idx:
            The training set index, and its type is dict.
        ret_te_idx:
            The test set index, and its type is dict.
    """
    num_bags = bags.shape[0]
    temp_rand_idx = np.random.permutation(num_bags)
    temp_fold = int(num_bags / 10)
    ret_tr_idx = {}
    ret_te_idx = {}
    for i in range(10):
        temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx[i] = temp_tr_idx
        ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
    return ret_tr_idx, ret_te_idx


def dis_euclidean(ins1, ins2):
    """
    Calculate the distance between two instances
    :param ins1: the first instance
    :param ins2: the second instance
    :return: the distance between two instances
    """
    dis_instances = np.sqrt(np.sum((ins1 - ins2) ** 2))
    return dis_instances


def cosine_similarity(instance1, instance2):
    """
    Cosine similarity with two instances.
    :param instance1: The first instance.
    :param instance2: The two instance.
    :return:
    """
    num = instance1.dot(instance2.T)
    de_nom = np.linalg.norm(instance1) * np.linalg.norm(instance2)
    return num / de_nom