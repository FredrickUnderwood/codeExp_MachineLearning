import numpy as np
import pandas as pd


# load data
def load_data(feature_list: pd.DataFrame, label_list: pd.DataFrame):
    example_num = feature_list.shape[0]
    feature_num = feature_list.shape[1]
    new_feature_list = pd.DataFrame(np.nan, index=range(example_num), columns=range(feature_num))
    new_label_list = pd.DataFrame(0, index=range(example_num), columns=range(3))
    for i in range(feature_num):
        feature_num = list(feature_list.iloc[:, i])
        feature_num = sorted(feature_num)
        feature_max = float(feature_num[-1])
        feature_min = float(feature_num[0])
        for j in range(example_num):
            new_feature_list.iloc[j, i] = (float(feature_list.iloc[j, i]) - feature_min) / (feature_max - feature_min)
    for i in range(example_num):
        label = int(label_list.iloc[i])
        new_label_list.iloc[i, label] = 1
    return new_feature_list, new_label_list


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(y_hat):
    exp_sum = 0
    exp_list = []
    softmax_list = []
    if isinstance(y_hat, list):
        for i in range(len(y_hat)):
            exp_list.append(np.exp(y_hat[i]))
            exp_sum += np.exp(y_hat[i])
        for i in range(len(y_hat)):
            softmax_list.append((exp_list[i] / exp_sum))
    else:
        for i in range(y_hat.shape[0]):
            exp_list.append(np.exp(y_hat.iloc[i]))
            exp_sum += np.exp(y_hat.iloc[i])
        for i in range(y_hat.shape[0]):
            softmax_list.append((exp_list[i] / exp_sum))
    return softmax_list


def train(feature_list: pd.DataFrame, label_list: pd.DataFrame, hidden_layer: int, learning_rate: float,
          num_epochs: int):
    example_num = feature_list.shape[0]
    df_bias = pd.DataFrame(-1, index=range(example_num), columns=[-1])
    feature_list_bias = pd.concat([df_bias, feature_list], axis=1)
    in_feature = feature_list_bias.shape[1]
    out_num = label_list.shape[1]
    w1 = np.random.randn(in_feature, hidden_layer)
    w2 = np.random.randn(hidden_layer, out_num)
    for epoch in range(num_epochs):
        


if __name__ == '__main__':
    train_path = './data/Iris-train.txt'
    data_list = []
    with open(train_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data_list.append(line.strip().split(' '))
    data_df = pd.DataFrame(data_list)
    feature_df = data_df.iloc[:, :-1]
    label_df = data_df.iloc[:, -1]
    feature_list_, label_list_ = load_data(feature_df, label_df)
