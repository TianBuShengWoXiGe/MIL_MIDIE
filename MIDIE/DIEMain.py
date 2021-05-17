# @Time : 2021/5/16 10:09
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : DIEMain.py
from MIDIE.DIEPredict import DIEPredict
from MILFrame.MnistLoadTool import MnistLoader

if __name__ == '__main__':
    file_path = "D:/Data/data_zero/Web/web1+.mat"
    print(file_path)
    # data_path = "../Data"
    # print(data_path)
    # po_label = 0
    bag_status = ['g', 'p', 'n']
    embed_status = ['add', 'con']
    # bags = MnistLoader(po_label=po_label, seed=1, data_path=data_path, bag_resize=(75, 75)).bag_space
    for embed_status_i in range(2):
        for bag_status_i in range(3):
            print(bag_status[bag_status_i], embed_status[embed_status_i])
            for i in range(1, 6, 1):
                DIEPredict(file_path, bag_status[bag_status_i], embed_status[embed_status_i], 0.1, i)
                # a=0
