import numpy as np
import pandas as pd

featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}


def generate_data_set():
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]
    ]
    feature_list = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']
    dataSet = np.array(dataSet)
    data_set = dataSet[:, :-1]
    label_set = dataSet[:, -1]
    # 转为DataFrame
    df = pd.DataFrame(data_set)
    df.columns = feature_list
    # 将文字数据转为可处理的数字
    color = pd.get_dummies(df.色泽, prefix='色泽')
    root = pd.get_dummies(df.根蒂, prefix='根蒂')
    knock = pd.get_dummies(df.敲声, prefix='敲声')
    texture = pd.get_dummies(df.纹理, prefix='纹理')
    touch = pd.get_dummies(df.触感, prefix='触感')
    density_sugar = pd.DataFrame()
    density_sugar["密度"] = df.密度
    density_sugar["含糖量"] = df.含糖量
    # 数据融合
    train_data = pd.concat([color, root, knock, texture, touch, density_sugar], axis=1)  # 融合
    train_features_list = list(
        train_data.columns)  # ['色泽_乌黑', '色泽_浅白', '色泽_青绿', '根蒂_硬挺', '根蒂_稍蜷', '根蒂_蜷缩', '敲声_沉闷', '敲声_浊响', '敲声_清脆',
    # '纹理_模糊', '纹理_清晰', '纹理_稍糊', '触感_硬滑', '触感_软粘', '密度', '含糖量']
    train_data = np.asarray(train_data, dtype=float)
    label_set = np.asarray(label_set, dtype=int).reshape(-1, 1)
    # 融合
    train_set = np.concatenate((train_data, label_set), axis=1)

    return train_set, train_features_list


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def calc_accuracy(data_set, V, theta1, W, theta2):
    right_count = 0
    test_iter, label_iter = data_set[:, :-1], data_set[:, -1]
    for data, label in zip(test_iter, label_iter):
        b = sigmoid(np.dot(data, V) - theta1)
        y_hat = sigmoid(np.dot(b, W)[0] - theta2)
        if label == 1:
            if y_hat >= 0.5:
                right_count += 1
        else:
            if y_hat < 0.5:
                right_count += 1
    return right_count / test_iter.shape[0]


def train(train_set, learning_rate, hidden_layer, num_epochs):
    train_acc = []
    y = train_set[:, -1]  # row_num*col_num
    X = train_set[:, :-1]
    row_num, col_num = X.shape  # _,num_inputs
    V = np.random.randn(col_num, hidden_layer)  # col_num*hidden_layer
    theta1 = np.random.randn(1, hidden_layer)
    W = np.random.randn(hidden_layer, 1)
    theta2 = np.random.randn(1)
    for epoch in range(num_epochs):
        for i in range(row_num):
            b = sigmoid(np.dot(X[i], V) - theta1)  # 隐藏层输出值 1*hidden_layer
            y_hat = sigmoid(np.dot(b, W)[0] - theta2)  # 输出层输出值 1
            g = y_hat * (1 - y_hat) * (y_hat - y[i])  # 1
            e = b * (1 - b) * g * W.T.sum()  # 1*hidden_layer
            W -= learning_rate * b.T * g
            V -= learning_rate * np.dot(X[i].reshape((col_num, -1)), e)
            theta1 += learning_rate * e
            theta2 += learning_rate * g
            train_acc.append(calc_accuracy(train_set, V, theta1, W, theta2))

    return train_acc


if __name__ == '__main__':
    train_set, train_features_list = generate_data_set()
    for train_acc in train(train_set, 0.1, 17, 100):
        print(train_acc)
