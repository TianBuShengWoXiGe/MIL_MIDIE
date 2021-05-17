# @Time : 2021/5/15 21:13
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : DIE.py
import warnings

import numpy as np

from MIDIE.SDI import SDI
from MILFrame.MILTool import MILTool, get_ten_fold_index, dis_euclidean

warnings.filterwarnings('ignore')


class DIE:
    def __init__(self, all_bag, tr_index, bags_status, embed_status, ratio_instance_to_bag,
                 num_discriminative_instance):
        self.bags = all_bag
        self.bags_status = bags_status
        self.embed_status = embed_status
        self.tr_index = tr_index
        self.ra_ins = ratio_instance_to_bag
        self.num_dis_ins = num_discriminative_instance
        self.train_final_bag = self.__get_bags()
        self.embedding_vector = self.__embedding()

    def __embedding(self):
        """
        Let a bag embedding into a single vector.
        :return:
        """
        # Step 1. Get the discriminative instance through the select discriminative instance method.
        discriminative_instance = SDI(self.train_final_bag, self.ra_ins, self.num_dis_ins).final_discriminative_instance
        # Step 2. Embedding the bags as the single vector.
        bag_to_vector = []
        for bag_i in range(self.bags.shape[0]):
            # Step 2.1 Find the embedding model.
            temp_single_vector = []
            if self.embed_status == 'add':
                temp_single_vector = np.zeros(self.bags[bag_i][0].shape[1] - 1).astype("float64")
            elif self.embed_status == 'con':
                temp_single_vector = np.zeros(self.num_dis_ins * (self.bags[bag_i][0].shape[1] - 1)).astype("float64")
            else:
                print('Your input model is not exist!\n')
                break
            # Step 2.2 Specific steps of embedding.
            for ins_i in self.bags[bag_i][0][:, :-1]:
                temp_distance_dis_to_ins = []
                # Step 2.2.1 Find  the nearest discriminative instance.
                for dis_ins_i in range(self.num_dis_ins):
                    temp_distance_dis_to_ins.append(dis_euclidean(ins_i, discriminative_instance[dis_ins_i]))
                temp_index = temp_distance_dis_to_ins.index(min(temp_distance_dis_to_ins))
                # Subtract the i-th instance and the nearest discriminative instance.
                temp_dis_to_ins_vector = ins_i - discriminative_instance[temp_index]
                if self.embed_status == 'add':
                    temp_single_vector += temp_dis_to_ins_vector
                elif self.embed_status == 'con':
                    start_index = 0
                    end_index = self.bags[bag_i][0].shape[1] - 1
                    temp_single_vector[start_index + temp_index * end_index:(temp_index + 1) * end_index] += (
                        temp_dis_to_ins_vector)
            # if self.embed_status == 'add':
            #     temp_single_vector[-1] = self.bags[bag_i][-1]
            # elif self.embed_status == 'con':
            #     temp_single_vector.extend(self.bags[bag_i][-1])

            # Step 3. Normalize the vector
            temp_single_vector = np.sign(temp_single_vector) * np.sqrt(np.abs(temp_single_vector))
            temp_norm = np.linalg.norm(temp_single_vector)
            temp_single_vector = temp_single_vector / temp_norm

            bag_to_vector.append(temp_single_vector)

        return np.array(bag_to_vector)

    def __get_bags(self):
        """
        Get train bags through three types:
            g: The source of bags is the global bags.
            p: The source of bags is the positive bags.
            n：The source of bags is the negative bags
        :return: The bags.
        """
        if self.bags_status == 'g':
            return self.bags[self.tr_index]
        elif self.bags_status == 'p':
            positive_bags_index = []
            for i in range(len(self.tr_index)):
                if self.bags[self.tr_index[i], -1] == 1:
                    positive_bags_index.append(self.tr_index[i])
            return self.bags[positive_bags_index]
        elif self.bags_status == 'n':
            negative_bags_index = []
            for i in range(len(self.tr_index)):
                if not self.bags[self.tr_index[i], -1] == 1:
                    negative_bags_index.append(self.tr_index[i])
            return self.bags[negative_bags_index]


if __name__ == '__main__':
    file_path = "D:/Data/data_zero/benchmark/musk1.mat"
    mil = MILTool(file_path)
    bags = mil.bags
    train_index, te_index = get_ten_fold_index(bags)
    miDie_demo = DIE(bags, train_index[1], 'g', 'con', 0.01, 2).embedding_vector
    print(miDie_demo)
    miDie_demo = np.array(miDie_demo)

    # print(miDie_demo[tr_index[1]])
