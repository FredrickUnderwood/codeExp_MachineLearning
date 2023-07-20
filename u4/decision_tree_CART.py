import numpy as np

import decision_tree_ID3


def calc_Gini(data_list):
    ans = 0
    pList = decision_tree_ID3.calc_probabilty(data_list, -1)
    for index in pList:
        ans += (pList[index] ** 2)
    Gini = 1 - ans
    return Gini


def calc_Gini_index(data_list, col_no):
    Gini_index = 0
    data_list_attr = data_list[:, [col_no, -1]]
    pList = decision_tree_ID3.calc_probabilty(data_list, col_no)
    for index in pList:
        new_array = data_list_attr[np.where(data_list_attr[:, 0] == index)]
        Gini_index += pList[index] * calc_Gini(new_array)
    return Gini_index # 越小越好


if __name__ == '__main__':
    data_list = []
    with open('watermelon_dataset_2.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data_list.append(line.strip().split(','))
    data_list = np.array(data_list)
    for i in range(1,7):
        print(calc_Gini_index(data_list,i))
