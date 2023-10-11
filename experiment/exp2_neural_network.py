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


def softmax(y_hat: np.ndarray):
    exp_sum = 0
    exp_list = []
    softmax_list = np.zeros_like(y_hat)
    example_num = y_hat.shape[0]
    label_num = y_hat.shape[1]
    for i in range(example_num):
        for j in range(label_num):
            exp_sum += np.exp(y_hat[i, j])
            exp_list.append(np.exp(y_hat[i, j]))
        for j in range(label_num):
            softmax_list[i, j] = exp_list[j] / exp_sum
        exp_sum = 0
        exp_list = []
    return softmax_list


def train_and_inference(feature_list: pd.DataFrame, label_list: pd.DataFrame, hidden_layer: int, learning_rate: float,
                        num_epochs: int, test_feature_list: pd.DataFrame, test_label_list: pd.DataFrame):
    example_num = feature_list.shape[0]
    df_bias = pd.DataFrame(-1, index=range(example_num), columns=[-1])
    feature_list_bias = pd.concat([df_bias, feature_list], axis=1)  # (75, 5)
    in_feature = feature_list_bias.shape[1]
    feature_list_bias = np.asarray(feature_list_bias)
    out_num = label_list.shape[1]
    label_list = np.asarray(label_list)
    label_list_num = label_list.argmax(axis=1)
    w1 = np.random.randn(in_feature, hidden_layer)  # (5, 10)
    w2 = np.random.randn(hidden_layer, out_num)  # (10, 3)
    best_acc = 0
    for _ in range(num_epochs):
        h = np.dot(feature_list_bias, w1)  # (75, 10)
        h_activate = sigmoid(h)
        y = np.dot(h_activate, w2)  # (75, 3)
        y_hat = softmax(y)
        d1 = (y_hat - label_list) * (1 - y_hat) * y_hat
        d2 = np.dot(d1, w2.T)  # (75, 10)
        d3 = d2 * h_activate * (1 - h_activate)
        d4 = np.dot(feature_list_bias.T, d3)  # for w1
        d5 = np.dot(h_activate.T, d1)  # for w2
        w1 -= d4 * learning_rate
        w2 -= d5 * learning_rate
        loss = ((y_hat - label_list) ** 2) / 2
        # y_hat = y_hat.argmax(axis=1)
        # correct = 0
        # for i in range(example_num):
        #     if y_hat[i] == label_list_num[i]:
        #         correct += 1
        # acc = correct / example_num
        # if acc > best_acc:
        #     best_acc = acc
    test_feature_list_bias = pd.concat([df_bias, test_feature_list], axis=1)
    test_example_num = test_feature_list_bias.shape[0]
    test_feature_list_bias = np.asarray(test_feature_list_bias)
    test_label_list = np.asarray(test_label_list)
    test_label_list_num = test_label_list.argmax(axis=1)
    h_test = np.dot(test_feature_list_bias, w1)
    h_activate_test = sigmoid(h_test)
    y_test = np.dot(h_activate_test, w2)
    y_hat_test = sigmoid(y_test)
    y_hat_test = y_hat_test.argmax(axis=1)
    correct = 0
    for i in range(test_example_num):
        if y_hat_test[i] == test_label_list_num[i]:
            correct += 1
    acc = correct / test_example_num
    return round(acc, 3)


if __name__ == '__main__':
    train_path = './data/Iris-train.txt'
    test_path = './data/Iris-test.txt'
    data_list = []
    test_list = []
    with open(train_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data_list.append(line.strip().split(' '))
    data_df = pd.DataFrame(data_list)
    feature_df = data_df.iloc[:, :-1]
    label_df = data_df.iloc[:, -1]
    feature_list_, label_list_ = load_data(feature_df, label_df)

    with open(test_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            test_list.append(line.strip().split(' '))
    test_df = pd.DataFrame(test_list)
    test_feature_df = test_df.iloc[:, :-1]
    test_label_df = test_df.iloc[:, -1]
    test_feature_list_, test_label_list_ = load_data(test_feature_df, test_label_df)
    acc_list = []
    for _ in range(10):
        acc_list.append(train_and_inference(feature_list_, label_list_, 10, 0.01, 1000,
                                            test_feature_list_, test_label_list_))
    print(acc_list)
    acc_mean = float(np.mean(acc_list))
    print(round(acc_mean, 3))
