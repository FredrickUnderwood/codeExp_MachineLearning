import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data（数据的归一化处理）
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
                        num_epochs: int, test_feature_list: pd.DataFrame, test_label_list: pd.DataFrame,
                        batch_size: int):
    example_num = feature_list.shape[0]  # 求得训练集样本数
    batch_num = int(example_num / batch_size)  # 求得batch的数量
    df_bias = pd.DataFrame(-1, index=range(example_num), columns=[-1])
    feature_list_bias = pd.concat([df_bias, feature_list], axis=1)  # (75, 5)，构造带有bias的数据集，4列->5列
    in_feature = feature_list_bias.shape[1]  # 求得feature数
    feature_list_bias = np.asarray(feature_list_bias)  # 将feature_list转为array，方便后续计算
    out_num = label_list.shape[1]  # 求得label有几种
    label_list = np.asarray(label_list)  # 将label_list转为array，方便后续计算
    w1 = np.random.randn(in_feature, hidden_layer)  # (5, 10)，初始化第一层的参数
    w2 = np.random.randn(hidden_layer, out_num)  # (10, 3)，初始化隐藏层的参数
    err_list = []  # 用来记录每个epoch的loss值

    # 开始训练
    for _ in range(num_epochs):
        loss_sum = 0  # 用于计算每一轮的损失
        for batch_idx in range(batch_num):
            feature_list_bias_tmp = feature_list_bias[
                                    batch_idx * batch_size:((batch_idx + 1) * batch_size)]  # 取得需要的batch对应的feature_list
            label_list_tmp = label_list[
                             batch_idx * batch_size:((batch_idx + 1) * batch_size)]  # 取得需要的batch对应的label_list
            # 数据的正向传播
            h = np.dot(feature_list_bias_tmp, w1)  # (75, 10)，隐藏层输出值
            h_activate = sigmoid(h)  # 隐藏层的激活值
            y = np.dot(h_activate, w2)  # (75, 3)，最后一层输出值
            y_hat = softmax(y)  # 最后一层的激活值
            loss = (((y_hat - label_list_tmp) ** 2) / 2).sum()  # 损失
            loss_sum += loss
            # 误差的反向传播
            d1 = (y_hat - label_list_tmp) * (1 - y_hat) * y_hat
            d2 = np.dot(d1, w2.T)  # (75, 10)
            d3 = d2 * h_activate * (1 - h_activate)
            d4 = np.dot(feature_list_bias_tmp.T, d3)  # w1的梯度
            d5 = np.dot(h_activate.T, d1)  # w2的梯度
            w1 -= d4 * learning_rate  # 执行梯度下降
            w2 -= d5 * learning_rate  # 执行梯度下降
        err_list.append(loss_sum / batch_num)  # 统计该轮的平均损失

    # inference部分
    test_feature_list_bias = pd.concat([df_bias, test_feature_list], axis=1)  # 生成带有bias列的测试集feature_list
    test_example_num = test_feature_list_bias.shape[0]  # 计算测试集样本数
    test_feature_list_bias = np.asarray(test_feature_list_bias)  # 转为array方便后续计算
    test_label_list = np.asarray(test_label_list)  # 转为array方便后续计算
    test_label_list_num = test_label_list.argmax(axis=1)  # 将独热编码转为一般编码，方便后续计算

    # 网络正向传播，进行inference
    h_test = np.dot(test_feature_list_bias, w1)
    h_activate_test = sigmoid(h_test)
    y_test = np.dot(h_activate_test, w2)
    y_hat_test = sigmoid(y_test)
    y_hat_test = y_hat_test.argmax(axis=1)
    # 统计正确率
    correct = 0
    for i in range(test_example_num):
        if y_hat_test[i] == test_label_list_num[i]:
            correct += 1
    acc = correct / test_example_num
    # 绘制损失函数
    x_test = range(num_epochs)
    plt.plot(x_test, err_list)
    # 返回inference正确率
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
    for i in range(10):
        acc_list.append(train_and_inference(feature_list_, label_list_, 10, 0.02, 1000,
                                            test_feature_list_, test_label_list_, batch_size=25))
    plt.show()
    print(acc_list)
    acc_mean = float(np.mean(acc_list))
    print(round(acc_mean, 3))