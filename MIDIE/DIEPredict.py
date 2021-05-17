# @Time : 2021/5/12 10:20
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : DIEPredict.py
# @Time : 2021/4/10 15:52
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : DIEPredict.py

import warnings

# import warnings
#
# warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from MIDIE.DIE import DIE
from MILFrame.MILTool import MILTool, get_ten_fold_index

# from MIL_MIDIE.SMDP.smdp import SmDp
# from MIL_MIDIE.miVLAD.VLAD import DpAm
warnings.filterwarnings('ignore')


class DIEPredict:
    def __init__(self, file_path, bags_status, embed_status, ratio_instance_to_bag,
                 num_discriminative_instance, bags=None):
        self.file_path = file_path
        self.bags_status = bags_status
        self.embed_status = embed_status
        self.ra_ins = ratio_instance_to_bag
        self.num_dis_ins = num_discriminative_instance
        self.bags = bags
        self.__DIEPredict()

    def __DIEPredict(self):
        MIL = MILTool(para_file_name=self.file_path, bags=self.bags)
        bags = MIL.bags
        bags_label = MIL.bags_label
        num_fold = 10

        accuracy_res_score = np.zeros(4).astype('float64')
        f1_res_score = np.zeros(4).astype('float64')
        roc_aoc_res_score = np.zeros(4).astype('float64')
        accuracy_std = [[], [], [], []]
        f1_res_std = [[], [], [], []]
        roc_aoc_res_std = [[], [], [], []]
        for i in range(num_fold):
            # The four Classifiers.
            knn_estimator = MIL.get_classifier('knn')
            DTree_estimator = MIL.get_classifier('DTree')
            LSVM_estimator = MIL.get_classifier('linear_svm')
            RSVM_estimator = MIL.get_classifier('rbf_svm')

            accuracy_measure_score = np.zeros(4).astype('float64')
            fl_measure_score = np.zeros(4).astype('float64')
            roc_measure_score = np.zeros(4).astype('float64')

            tr_index, te_index = get_ten_fold_index(bags)
            temp_f1_score = 10
            temp_acc_score = 10
            temp_roc_score = 10
            for j in range(10):
                # get_bar(i, j)
                tr_bags = bags[tr_index[j]]
                tr_label_true = bags_label[tr_index[j]]
                te_bags = bags[te_index[j]]
                te_label_true = bags_label[te_index[j]]

                # Step 1. Call the algorithm to get the mapping vector.
                test_Demo = DIE(bags, tr_index[j], self.bags_status, self.embed_status, self.ra_ins,
                                self.num_dis_ins).embedding_vector

                tr_embedding_vector = test_Demo[tr_index[j]]
                te_embedding_vector = test_Demo[te_index[j]]

                # Step 2. This is a knn classification model.
                # Step 2.1.1 This is a knn classification model.
                knn_model = knn_estimator.fit(tr_embedding_vector, tr_label_true)
                te_label_predict_knn = knn_model.predict(te_embedding_vector)
                # Step 2.1.2 The three predict score.
                fl_measure_score[0] += f1_score(te_label_true, te_label_predict_knn)
                accuracy_measure_score[0] += accuracy_score(te_label_true, te_label_predict_knn)
                try:
                    temp_aoc = roc_auc_score(te_label_true, te_label_predict_knn)
                    roc_measure_score[0] += temp_aoc
                except ValueError:
                    pass
                # roc_measure_score[0] += roc_auc_score(te_label_true, te_label_predict_knn)

                # Step 2.2.1 This is a DTree classification model.
                DTree_model = DTree_estimator.fit(tr_embedding_vector, tr_label_true)
                te_label_predict_tree = DTree_model.predict(te_embedding_vector)

                # Step 2.2.2 The three predict score.
                fl_measure_score[1] += f1_score(te_label_true, te_label_predict_tree)
                accuracy_measure_score[1] += accuracy_score(te_label_true, te_label_predict_tree)
                try:
                    temp_aoc = roc_auc_score(te_label_true, te_label_predict_knn)
                    roc_measure_score[1] += temp_aoc
                except ValueError:
                    pass
                # roc_measure_score[1] += roc_auc_score(te_label_true, te_label_predict_tree)

                # Step 2.3.1 This is a DTree classification model.
                LSVM_model = LSVM_estimator.fit(tr_embedding_vector, tr_label_true)
                te_label_predict_LSVM = LSVM_model.predict(te_embedding_vector)

                # Step 2.3.2 The three predict score.
                fl_measure_score[2] += f1_score(te_label_true, te_label_predict_LSVM)
                accuracy_measure_score[2] += accuracy_score(te_label_true, te_label_predict_LSVM)
                try:
                    temp_aoc = roc_auc_score(te_label_true, te_label_predict_knn)
                    roc_measure_score[2] += temp_aoc
                except ValueError:
                    pass
                # roc_measure_score[2] += roc_auc_score(te_label_true, te_label_predict_LSVM)

                # Step 2.3.1 This is a DTree classification model.
                RSVM_model = RSVM_estimator.fit(tr_embedding_vector, tr_label_true)
                te_label_predict_RSVM = RSVM_model.predict(te_embedding_vector)

                # Step 2.3.2 The three predict score.
                fl_measure_score[3] += f1_score(te_label_true, te_label_predict_RSVM)
                accuracy_measure_score[3] += accuracy_score(te_label_true, te_label_predict_RSVM)
                try:
                    temp_aoc = roc_auc_score(te_label_true, te_label_predict_knn)
                    roc_measure_score[3] += temp_aoc
                except ValueError:
                    pass
                # roc_measure_score[3] += roc_auc_score(te_label_true, te_label_predict_RSVM)

            accuracy_res_score[0] += accuracy_measure_score[0] / temp_acc_score  # Knn   acc
            accuracy_res_score[1] += accuracy_measure_score[1] / temp_acc_score  # DTree acc
            accuracy_res_score[2] += accuracy_measure_score[2] / temp_acc_score  # LSVM  acc
            accuracy_res_score[3] += accuracy_measure_score[3] / temp_acc_score  # RSVM  acc

            accuracy_std[0].append(accuracy_measure_score[0] * temp_acc_score)  # Knn   std
            accuracy_std[1].append(accuracy_measure_score[1] * temp_acc_score)  # DTree std
            accuracy_std[2].append(accuracy_measure_score[2] * temp_acc_score)  # LSVM  std
            accuracy_std[3].append(accuracy_measure_score[3] * temp_acc_score)  # RSVM  std

            f1_res_score[0] += fl_measure_score[0] / temp_f1_score  # Knn   f1
            f1_res_score[1] += fl_measure_score[1] / temp_f1_score  # DTree f1
            f1_res_score[2] += fl_measure_score[2] / temp_f1_score  # LSVM  f1
            f1_res_score[3] += fl_measure_score[3] / temp_f1_score  # RSVM  f1

            f1_res_std[0].append(fl_measure_score[0] * temp_f1_score)  # Knn   std
            f1_res_std[1].append(fl_measure_score[1] * temp_f1_score)  # DTree std
            f1_res_std[2].append(fl_measure_score[2] * temp_f1_score)  # LSVM  std
            f1_res_std[3].append(fl_measure_score[3] * temp_f1_score)  # RSVM  std

            roc_aoc_res_score[0] += roc_measure_score[0] / temp_roc_score  # Knn   roc
            roc_aoc_res_score[1] += roc_measure_score[1] / temp_roc_score  # DTree roc
            roc_aoc_res_score[2] += roc_measure_score[2] / temp_roc_score  # LSVM  roc
            roc_aoc_res_score[3] += roc_measure_score[3] / temp_roc_score  # RSVM  roc

            roc_aoc_res_std[0].append(roc_measure_score[0] * temp_roc_score)  # Knn   std
            roc_aoc_res_std[1].append(roc_measure_score[1] * temp_roc_score)  # DTree std
            roc_aoc_res_std[2].append(roc_measure_score[2] * temp_roc_score)  # LSVM  std
            roc_aoc_res_std[3].append(roc_measure_score[3] * temp_roc_score)  # RSVM  std

        knn_acc_res = "&$%.1f" % (accuracy_res_score[0] * 10) + "_{\pm%.2f" % (np.std(accuracy_std[0])) + "}$"
        knn_f1_res = "&$%.1f" % (f1_res_score[0] * 10) + "_{\pm%.2f" % (np.std(f1_res_std[0])) + "}$"
        knn_roc_res = "&$%.1f" % (roc_aoc_res_score[0] * 10) + "_{\pm%.2f" % (np.std(roc_aoc_res_std[0])) + "}$"

        DTree_acc_res = "&$%.1f" % (accuracy_res_score[1] * 10) + "_{\pm%.2f" % (np.std(accuracy_std[1])) + "}$"
        DTree_f1_res = "&$%.1f" % (f1_res_score[1] * 10) + "_{\pm%.2f" % (np.std(f1_res_std[1])) + "}$"
        DTree_roc_res = "&$%.1f" % (roc_aoc_res_score[1] * 10) + "_{\pm%.2f" % (np.std(roc_aoc_res_std[1])) + "}$"

        LSVM_acc_res = "&$%.1f" % (accuracy_res_score[2] * 10) + "_{\pm%.2f" % (np.std(accuracy_std[2])) + "}$"
        LSVM_f1_res = "&$%.1f" % (f1_res_score[2] * 10) + "_{\pm%.2f" % (np.std(f1_res_std[2])) + "}$"
        LSVM_roc_res = "&$%.1f" % (roc_aoc_res_score[2] * 10) + "_{\pm%.2f" % (np.std(roc_aoc_res_std[2])) + "}$"

        RSVM_acc_res = "&$%.1f" % (accuracy_res_score[3] * 10) + "_{\pm%.2f" % (np.std(accuracy_std[3])) + "}$"
        RSVM_f1_res = "&$%.1f" % (f1_res_score[3] * 10) + "_{\pm%.2f" % (np.std(f1_res_std[3])) + "}$"
        RSVM_roc_res = "&$%.1f" % (roc_aoc_res_score[3] * 10) + "_{\pm%.2f" % (np.std(roc_aoc_res_std[3])) + "}$"

        # print("\nknn_acc & std   \t knn_f1 & std   \t knn_roc & std\t\t"
        #       "DTree_acc & std \t DTree_f1 & std \t DTree_roc & std\t"
        #       "LSVM_acc & std  \t LSVM_f1 & std  \t LSVM_roc & std\t\t"
        #       "RSVM_acc & std  \t RSVM_f1 & std  \t RSVM_roc & std")
        print('\t\t\t\t\t', knn_acc_res, '\t', knn_f1_res, '\t', knn_roc_res, '\t',
              DTree_acc_res, '\t', DTree_f1_res, '\t', DTree_roc_res, '\t',
              LSVM_acc_res, '\t', LSVM_f1_res, '\t', LSVM_roc_res, '\t',
              RSVM_acc_res, '\t', RSVM_f1_res, '\t', RSVM_roc_res)


if __name__ == '__main__':
    file_name = "D:/Data/data_zero/Mutagenesis/mutagenesis1.mat"
    para_name = file_name[30:]
    # DIEPredict(file_name=file_name, para_name=para_name)
