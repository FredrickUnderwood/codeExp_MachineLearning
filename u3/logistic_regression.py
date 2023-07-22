# import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def get_data_set():
    with open('watermelon_dataset_3.txt', 'r', encoding='utf-8') as file1:
        data_list = []
        lines = file1.readlines()
        for line in lines:
            data_list.append(line.strip().split(','))
        for data in data_list:
            data.insert(2, 1)
    return np.array(data_list, dtype=float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid+cross_entropy
def train(data_list, train_list, learning_rate, epoch_nums):
    m, n = data_list.shape
    beta = np.ones((n - 1, 1))  # n-1行1列
    train_set = data_list[train_list]
    feature_list = train_set[:, :-1]
    label_list = np.array(train_set[:, -1]).reshape(-1, 1)
    for epoch in range(epoch_nums):
        y_hat = sigmoid(np.dot(feature_list, beta))  # y_hat是一个16行1列的向量
        loss_grad = -feature_list * (label_list - y_hat)  # loss_grad是一个16行3列的向量
        l_grad_sum = np.sum(loss_grad, axis=0, keepdims=True)  # 变为16行1列
        beta -= learning_rate * l_grad_sum.T  # beta也是16行1列
        print(f'epoch{epoch} correct rate={test(data_list, beta)}')
    return beta


def test(data_list, beta):
    feature_list = data_list[:, :-1]
    label_list = data_list[:, -1]
    right_count = 0
    false_count = 0
    for data, label in zip(feature_list, label_list):
        y_hat = sigmoid(np.dot(data, beta))
        if label == 1:
            if y_hat >= 0.5:
                right_count += 1
            else:
                false_count += 1
        else:
            if y_hat < 0.5:
                right_count += 1
            else:
                false_count += 1
    return right_count / (right_count + false_count)


if __name__ == '__main__':
    data_list = get_data_set()
    train_list = [x for x in range(16)]
    lr = 0.001
    epoch_nums = 1000
    beta_grad = train(data_list, train_list, lr, epoch_nums)

    pos_node = []
    neg_node = []
    for data in data_list:
        if data[-1] == 1:
            pos_node.append(list(data))
        else:
            neg_node.append(list(data))
    plt.figure()
    col0 = [col[0] for col in pos_node]
    col1 = [col[1] for col in pos_node]
    col2 = [col[0] for col in neg_node]
    col3 = [col[1] for col in neg_node]

    plt.scatter(col0, col1)
    plt.scatter(col2, col3)
    x1 = np.linspace(0, 1, 100)
    x2grad = -(x1 * beta_grad[0] + beta_grad[2]) / beta_grad[1]
    plt.plot(x1, x2grad, label="gradient descent")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
