import numpy as np
import pandas as pd
import math
import treePlotter


# load data from txt
def load_data(data_path='./data/traindata.txt'):
    data_list = []
    with open(data_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data_list.append(line.strip().split('\t'))
    return pd.DataFrame(data_list)


# prepare the functions
def calc_info_ent(label_list: pd.DataFrame):
    info_ent = 0
    example_num = label_list.shape[0]
    labels = label_list.unique()
    for label in labels:
        label_num = label_list[label_list == label].count()
        info_ent -= (label_num / example_num) * math.log2(label_num / example_num)
    return info_ent


def calc_cond_ent(feature_list: pd.DataFrame, label_list: pd.DataFrame):
    cond_ent = 0
    example_num = feature_list.shape[0]
    features = feature_list.unique()
    for feature in features:
        sub_label_col = []
        for i in range(example_num):
            if feature_list.iloc[i] == feature:
                sub_label_col.append(i)
        info_ent = calc_info_ent(label_list.iloc[sub_label_col])
        cond_ent += (len(sub_label_col) / example_num) * info_ent
    return cond_ent


def calc_gain(feature_list: pd.DataFrame, label_list: pd.DataFrame):
    best_gain = 0
    best_feature = 0
    feature_num = feature_list.shape[1]
    for i in range(feature_num):
        info_ent = calc_info_ent(label_list)
        cond_ent = calc_cond_ent(feature_list.iloc[:, i], label_list)
        gain = info_ent - cond_ent
        if gain > best_gain:
            best_gain = gain
            best_feature = i
    return best_feature


# transform to discrete data
def data_preprocess(feature_list: pd.DataFrame, label_list: pd.DataFrame):
    example_num = feature_list.shape[0]
    feature_num = feature_list.shape[1]
    divide_val = []
    for i in range(feature_num):
        feature_values = feature_list.iloc[:, i]
        feature_values = list(feature_values.unique())
        feature_values = sorted(feature_values)
        df_pre_feature = pd.DataFrame(np.nan, index=range(example_num), columns=range(len(feature_values)))
        k = 0
        for feature_value in feature_values:
            for j in range(example_num):
                if feature_list.iloc[j, i] <= feature_value:
                    df_pre_feature.iloc[j, k] = 0
                else:
                    df_pre_feature.iloc[j, k] = 1
            k += 1
        divide_value = feature_values[calc_gain(df_pre_feature, label_list)]
        divide_val.append(divide_value)
    return divide_val


# generate the feature_list with discrete value
def generate_feature_list(feature_list: pd.DataFrame, divide_values: list):
    example_num = feature_list.shape[0]
    feature_num = feature_list.shape[1]
    discrete_feature_list = pd.DataFrame(np.nan, index=range(example_num), columns=range(feature_num))
    for i in range(feature_num):
        for j in range(example_num):
            if feature_list.iloc[j, i] <= divide_values[i]:
                discrete_feature_list.iloc[j, i] = 0
            else:
                discrete_feature_list.iloc[j, i] = 1
    return discrete_feature_list


# build the tree
def decision_tree(feature_list: pd.DataFrame, label_list: pd.DataFrame):
    if len(label_list.unique()) == 1:
        return int(label_list.iloc[0])
    divide_vals = data_preprocess(feature_list, label_list)
    discrete_feature_list = generate_feature_list(feature_list, divide_vals)
    best_feature = calc_gain(discrete_feature_list, label_list)
    best_feature_name = feature_list.columns[best_feature]
    tree_node = {best_feature_name: {}}
    example_num = feature_list.shape[0]
    example_larger = []
    example_not_larger = []
    for i in range(example_num):
        if discrete_feature_list.iloc[i, best_feature] == 1:
            example_larger.append(i)
        else:
            example_not_larger.append(i)
    sub_label0 = label_list.iloc[example_not_larger]
    sub_feature0 = feature_list.iloc[example_not_larger, :]
    sub_label1 = label_list.iloc[example_larger]
    sub_feature1 = feature_list.iloc[example_larger, :]
    for i in range(2):
        if i == 0:
            tree_node[best_feature_name][f'<={divide_vals[best_feature]}'] = decision_tree(sub_feature0, sub_label0)
        if i == 1:
            tree_node[best_feature_name][f'>{divide_vals[best_feature]}'] = decision_tree(sub_feature1, sub_label1)
    return tree_node


# inference
def inference(tree, feature_name, test_features):
    if isinstance(tree, int):
        return tree
    for key in tree:
        if key[0] == '<' or key[0] == '>':
            if key[0] == '<':
                divide_value = float(key[2:])
                if float(test_features[feature_name]) <= divide_value:
                    return inference(tree[key], feature_name, test_features)
            else:
                divide_value = float(key[1:])
                if float(test_features[feature_name]) > divide_value:
                    return inference(tree[key], feature_name, test_features)
        else:
            feature_name = key
            return inference(tree[key], feature_name, test_features)


if __name__ == '__main__':
    df_data = load_data()
    df_data.columns = ['1', '2', '3', '4', 'label']
    df_label = df_data.iloc[:, -1]
    df_feature = df_data.iloc[:, :-1]
    df_feature_f1 = df_feature.iloc[0]

    df_test_data = load_data(data_path='./data/testdata.txt')
    df_test_data.columns = ['1', '2', '3', '4', 'label']
    df_test_label = df_test_data.iloc[:, -1]
    df_test_feature = df_test_data.iloc[:, :-1]

    my_tree = decision_tree(df_feature, df_label)
    treePlotter.createPlot(my_tree)
    correct = 0
    for i in range(df_test_feature.shape[0]):
        if inference(my_tree, '0', df_test_feature.iloc[i]) == int(df_test_label.iloc[i]):
            correct += 1
        else:
            inference_ans = inference(my_tree, '0', df_test_feature.iloc[i])
            print(f'No.{i} is wrong, inference answer is {inference_ans}, right answer is {df_test_label.iloc[i]}')
    print('accuracy:', correct / df_test_feature.shape[0])
