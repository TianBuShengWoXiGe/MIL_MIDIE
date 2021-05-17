# @Time : 2021/5/11 22:01
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : SDI.py
import numpy as np
from scipy.spatial.distance import cdist

import MIDIE.DIP as DIP
from MILFrame.MILTool import MILTool


class SDI:
    def __init__(self, bags, ratio_instance_to_bag, num_SDI):
        self.bags = bags
        self.ratio_ins = ratio_instance_to_bag
        self.num_SDI = num_SDI
        self.discriminative_instance = self.__print_SDI()
        self.final_discriminative_instance = self.__select_discriminative_instance()

    def __print_SDI(self):
        DIP_demo = DIP.DIP(self.bags, self.ratio_ins)
        return DIP_demo.discriminative_instance

    def __select_discriminative_instance(self):
        """
        Select discriminative instance through the density peak.
        :return:
        """
        # Step 1. Calculate the distance of discriminative instances.
        discriminative_instance_distance = cdist(self.discriminative_instance, self.discriminative_instance)
        # Step 2. The cutoff distance.
        dis_cut = 0.4 * discriminative_instance_distance.max()
        # The density of  discriminative instances.
        density_discriminative_ins = np.zeros(len(discriminative_instance_distance)).astype("float64")
        for i in range(len(discriminative_instance_distance)):
            if dis_cut == 0:
                density_discriminative_ins[i] = 1
            else:
                density_discriminative_ins[i] = sum(np.exp(-(discriminative_instance_distance[i] / dis_cut) ** 2))
        # Step 3. The distance of closest instance that is denser than itself.
        distance_closest = []
        for i in range(len(density_discriminative_ins)):
            more_density_instance_index = []
            temp_density_instance = density_discriminative_ins[i]
            for j in range(len(density_discriminative_ins)):
                if density_discriminative_ins[j] > temp_density_instance:
                    more_density_instance_index.append(j)
            temp_distance_more_instance = []
            for index in range(0, len(more_density_instance_index)):
                index_k = more_density_instance_index[index]
                temp_distance_more_instance.append(discriminative_instance_distance[i][index_k])
            if temp_distance_more_instance:
                temp_distance_more_instance.sort()
                distance_closest.append(temp_distance_more_instance[0])
            else:
                distance_closest.append(float('inf'))
        # Step 4. The lambda of discriminative instance.
        lambda_discriminative_instance = np.multiply(distance_closest, density_discriminative_ins).tolist()
        final_discriminative_instance = []
        for i in range(self.num_SDI):
            index_most = lambda_discriminative_instance.index(max(lambda_discriminative_instance))
            final_discriminative_instance.append(self.discriminative_instance[index_most])
            lambda_discriminative_instance[index_most] = -1

        return np.array(final_discriminative_instance)


if __name__ == '__main__':
    file_path = "D:/Data/data_zero/benchmark/musk1.mat"
    mil = MILTool(file_path)
    bags = mil.bags
    SDI_demo = SDI(bags, 0.01, 2)
    ins = SDI_demo.final_discriminative_instance
    print(type(ins))
