import math

import decision_tree_ID3


def calc_IV(data_list, col_no):
    iv = 0
    pList = {}
    pList = decision_tree_ID3.calc_probabilty(data_list, col_no)
    for index in pList:
        iv -= (pList[index] * math.log(pList[index], 2))
    return iv


def calc_Gain_ratio(data_list, col_no):
    return decision_tree_ID3.calc_gain(data_list, col_no) / calc_IV(data_list, col_no) # 越大越好


if __name__ == '__main__':
    pass
