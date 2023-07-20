import math
from collections import Counter

import numpy as np


class tree:
    def __init__(self, node_name, value):
        self.node_name = node_name
        self.value = value  # 判断是第n号属性，从1-5
        self.children = []  # 存取子节点和到子节点边的权值

    # def __repr__(self):
    #     features = ['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    #     if str(self.value).isdigit():
    #         return str(self.node_name) + ':' + str(features[self.value]) + str(self.children) + '\n'
    #     else:
    #         return str(self.node_name) + ':' + str(self.value) + str(self.children) + '\n'
    @property
    def get_children(self):
        return self.children

    @property
    def get_info(self):
        return self.node_name, self.value


def get_data_set():  # 数据预处理，处理成array的形式
    data_list = []
    with open('watermelon_dataset_2.txt', 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        for line in lines:
            data_list.append(line.strip().split(','))
    return np.array(data_list)


def calc_probabilty(data_list, col_no):  # 返回每一列中的每一个取值所占的比例
    pList = {}
    label, count = np.unique(data_list[:, col_no], return_counts=True)  # 可以返回取值和取值的个数
    prob = count / data_list.shape[0]  # 返回数组的行数
    pList = dict(zip(label, prob))
    return pList


def calc_entropy(data_list):  # 计算信息熵
    entropy = 0
    pList = calc_probabilty(data_list=data_list, col_no=-1)
    for key in pList:
        p = float(pList[key])
        entropy -= p * math.log(p, 2)
    return entropy


def calc_gain(data_list, col_no):  # 计算信息增益
    mutual_information = 0
    data_list_attr = data_list[:, [col_no, -1]]
    pList = calc_probabilty(data_list, col_no)
    values = np.unique(data_list_attr[:, 0])
    for value in values:
        new_array = data_list_attr[np.where(data_list_attr[:, 0] == value)]
        mutual_information += pList[value] * calc_entropy(new_array)
    return calc_entropy(data_list) - mutual_information


def generate_train_set(data_list, train_list):  # 生成训练集
    return data_list[train_list]


def generate_test_list(data_list, test_list):  # 生成测试集合
    return data_list[test_list]


def all_attr_equal(data_list, feature_list):  # 用于确认一个集合的 每一个样本 是否 所有的属性 都 相等
    for feature in feature_list:
        data_list2 = data_list[:, feature]
        if len(set(data_list2)) != 1:
            return False
    return True


def find_max(data_list):  # 寻扎出现次数最多的取值
    count = Counter(data_list)
    return count.most_common(1)[0][0]


def generate_decision_tree(train_set, features_list, current_node):  # 生成决策树
    # 判断是否所有样本都是正例或者反例
    values = np.unique(train_set[:, -1])
    if len(values) == 1:
        if values[0] == '1':  # value是numpy的array，但是value[0]是str
            current_node.value = '好瓜'
            node_list.append(current_node)
            return
        else:
            current_node.value = '坏瓜'
            node_list.append(current_node)
            return
    # 如果属性集为空或所有剩余样本的属性全部相等
    if len(features_list) == 0 or all_attr_equal(train_set, features_list) == True:
        if find_max(train_set[:, -1]) == 1:
            current_node.value = '好瓜'
            node_list.append(current_node)
            return
        else:
            current_node.value = '坏瓜'
            node_list.append(current_node)
            return
    # 寻找最优划分属性
    largest_gain = -0x3f3f3f3f
    best_feature = None
    for feature in features_list:
        Gain = calc_gain(train_set, feature)
        if Gain >= largest_gain:
            largest_gain = Gain
            best_feature = feature
    # 遍历该最优划分属性的所有取值
    list1 = np.unique(train_set_all[:, best_feature])
    for val in list1:
        # print(best_feature,':',np.unique(train_set[:,best_feature]),':',val)
        flag = False
        train_set_V = []
        # print(train_set,val,best_feature)
        for iter in train_set:
            if iter[best_feature] == val:
                flag = True
                train_set_V.append(iter)
        train_set_V = np.array(train_set_V)
        global node_count
        if not flag:
            if find_max(train_set[:, -1]) == '1':
                node_count += 1
                new_node_name = f'node{node_count}'
                current_node.children.append((new_node_name, val))
                new_node_name = tree(new_node_name, '好瓜')
                node_list.append(new_node_name)
            else:
                node_count += 1
                new_node_name = f'node{node_count}'
                current_node.children.append((new_node_name, val))
                new_node_name = tree(new_node_name, '坏瓜')
                node_list.append(new_node_name)
            continue
        else:
            current_node.value = best_feature
            node_count += 1
            new_node_name = f'node{node_count}'
            current_node.children.append((new_node_name, val))
            new_node_name = tree(new_node_name, None)
            new_features_list = features_list.copy()
            new_features_list.remove(best_feature)
            generate_decision_tree(train_set_V, new_features_list, new_node_name)
    node_list.append(current_node)


def test(test_node, tree_list, current_node):
    if len(current_node.get_children) == 0:
        return current_node.get_info[1]
    for node in tree_list:
        if current_node.get_info[0] == node.get_info[0]:
            col_no = node.get_info[1]
            for child in node.get_children:
                if test_node[col_no] == child[1]:
                    new_node_name = child[0]
                    for new_node in tree_list:
                        if new_node.get_info[0] == new_node_name:
                            return test(test_node, tree_list, new_node)


if __name__ == '__main__':
    node_count = 0
    features = ['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    features_list = [1, 2, 3, 4, 5, 6]
    all_list = [x for x in range(17)]
    train_list = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
    test_list = [3, 4, 7, 8, 10, 11, 12]
    node0 = tree('node0', '好瓜')
    node_list = []
    train_set_all = np.array(generate_train_set(get_data_set(), all_list))
    train_set = np.array(generate_train_set(get_data_set(), train_list))
    test_set = np.array(generate_test_list(get_data_set(), test_list))
    generate_decision_tree(train_set, features_list, node0)
    for node in node_list:
        if type(node.get_info[1]) == str:
            print(f'({node.get_info[0]},{node.get_info[1]})', end=':')
        else:
            print(f'({node.get_info[0]},{features[int(node.get_info[1])]})', end=':')
        print(node.get_children)
    right_count = 0
    for test_sample in test_set:
        if test(test_sample, node_list, node0) == '好瓜':
            if test_sample[7] == '1':
                right_count += 1
            else:
                pass
        else:
            if test_sample[7] == '0':
                right_count += 1
            else:
                pass
    print('accuracy:', right_count / len(test_set) * 100, '%')
