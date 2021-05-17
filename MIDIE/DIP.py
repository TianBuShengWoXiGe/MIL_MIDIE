# @Time : 2021/5/15 9:09
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : DIP.py
import math

import numpy as np
from scipy.spatial.distance import cdist

from MILFrame.MILTool import MILTool


class DIP:
    def __init__(self, bags, scale_num):
        self.bags = bags
        self.scale_num = scale_num
        self.inner_bag_distance = []
        self.discriminative_instance = self.__get_discriminative_ins()

    def __get_discriminative_ins(self):
        """
        Get the discriminative instance of the each bags to build the discriminative instance pool.
        :return:
        """
        for bag_i in range(self.bags.shape[0]):
            self.__get_instance_in_bag(self.bags[bag_i])
        return np.array(self.inner_bag_distance)

    def __get_instance_in_bag(self, bag):
        """
        Extract instance from bag as discriminative instance.
            Input: The single bag with the label.
            process:
                First：Calculate the density and affinity of instance.
                Second: lambda = density * affinity.
                Third：  Take the first k instances with larger lambda values as discriminative instance, where k=num_bag*scale_num.
            Output: The discriminative instance set.
        :param bag: The single bag.
        :return:
        """
        # Step 1. Calculate the distances of each instance.
        # print(bag[0].shape[0])
        ins_space_to_bag = []
        for ins in bag[0][:, :-1]:
            ins_space_to_bag.append(ins)
        ins_space_to_bag = np.array(ins_space_to_bag)
        distance_ins_to_ins = cdist(ins_space_to_bag, ins_space_to_bag)
        # Step 2. Calculate the affinity of each instance.
        affinity_ins = np.zeros((len(distance_ins_to_ins), len(distance_ins_to_ins))).astype("int32")
        affinity_ins_score = []
        density_ins_score = []
        ave_dis_ins = distance_ins_to_ins.mean()
        for i in range(len(distance_ins_to_ins)):
            for j in range(len(distance_ins_to_ins)):
                if distance_ins_to_ins[i, j] <= ave_dis_ins:
                    affinity_ins[i, j] = 1
            affinity_ins_score.append(sum(affinity_ins[i]))
        # Step 3. Calculate the density of each instance.
        dis_cut = 0.4 * distance_ins_to_ins.max()
        density_ins_score = np.zeros(len(distance_ins_to_ins)).astype("float64")
        for i in range(len(distance_ins_to_ins)):
            if dis_cut == 0:
                density_ins_score[i] = 1
            else:
                density_ins_score[i] = sum(np.exp(-(distance_ins_to_ins[i] / dis_cut) ** 2))
        # Step 4. Calculate the lambda of each instance.
        lambda_ins_score = np.multiply(affinity_ins_score, density_ins_score).tolist()
        # Step 5. Get the discriminative instance.
        for i in range(math.ceil(self.scale_num * bag[0].shape[0])):
            self.inner_bag_distance.append(ins_space_to_bag[lambda_ins_score.index(max(lambda_ins_score))])
            lambda_ins_score[lambda_ins_score.index(max((lambda_ins_score)))] = -1

        # return lambda_ins


if __name__ == '__main__':
    file_path = "D:/Data/data_zero/benchmark/musk1.mat"
    mil = MILTool(file_path)
    bags = mil.bags
    test = DIP(bags, 0.001)
    dis = test.discriminative_instance
    # test.get_discriminative_ins()
    # dis=test.inner_bags_distance
    print(dis)
    print(dis.shape)
    # print(dis.mean())
