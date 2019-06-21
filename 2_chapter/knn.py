# k-近邻算法
import os
import numpy as np
import operator
from sklearn.neighbors import KNeighborsClassifier as KNN


def classify(input_test, data_sets, labels, k):
    """
    KNN 分类器
    :param input_test: 测试集
    :param data_sets: 训练数据集
    :param labels: 分类标签
    :param k: kNN算法参数,选择距离最小的k个点
    :return:
    """
    # 对应坐标差值的平方
    diff = (input_test - data_sets) ** 2
    # input_test 到 data_sets中每个点的欧式距离
    distance = diff.sum(axis=1) ** 0.5

    # 返回distance中元素从小到大排序后的索引值
    sorted_index = distance.argsort()
    # 记录类别次数的字典
    class_count = {}
    for i in range(k):
        # 前k个元素的类别
        vote_label = labels[sorted_index[i]]
        # dict.get(key, default=None), 字典的get()方法, 返回指定键的值, 如果值不在字典中返回默认值。
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # key=operator.itemgetter(1)根据字典的值进行排序
    # 0 表示根据字典的键进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sorted_class_count[0][0]


# 案例一的数据集
def create_data_set():
    """
    创建数据集
    :return: data_sets: 数据集
            labels: 分类标签
    """
    # 四组二维特征
    data_sets = np.array([[1.0, 0.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 四组特征的标签
    labels = ['A', 'A', 'B', 'B']
    return data_sets, labels


# 案例二数据集
def file2matrix(file_name):
    """
    文件转换成矩阵类型
    :param file_name: 文件名
    :return: return_mat: 特征矩阵
            class_label_vector: 标签向量
    """
    with open(file_name) as fr:
        array_lines = fr.readlines()
        number_lines = len(array_lines)
        return_mat = np.zeros((number_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_lines:
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]
            class_label_vector.append((list_from_line[-1]))
            index += 1

        return return_mat, class_label_vector


# 归一化
def auto_norm(data_sets):
    """
    归一化
    :param data_sets: 数据集
    :return: norm_data_sets: 归一化后的数据集
             ranges: 数据范围
             min_values： 数据最小值
    """
    # min_values所有列中元素的最小值
    min_values = data_sets.min(0)
    # max_values所有列中元素的最大值
    max_values = data_sets.max(0)
    ranges = max_values - min_values
    norm_data_sets = (data_sets - min_values) / ranges
    return norm_data_sets, ranges, min_values


def dating_class_test():
    """
    分类器测试函数
    :return:
    """
    file_name = 'datingTestSet.txt'
    dating_data_mat, labels = file2matrix(file_name)
    norm_data_sets, ranges, min_values = auto_norm(dating_data_mat)
    ratio = 0.1
    m = norm_data_sets.shape[0]
    number_test_vector = int(m * ratio)
    error_count = 0
    for i in range(number_test_vector):
        # 前10% 数据为测试数据
        classfier_result = classify(norm_data_sets[i, :], norm_data_sets[number_test_vector:m, :],
                                    labels[number_test_vector: m], 3)
        print("分类结果:{}, 真实类别: {}".format(classfier_result, labels[i]))
        if classfier_result != labels[i]:
            error_count += 1
    print("错误率:{}".format(error_count / float(number_test_vector)))


def classify_person(miles, percent, ice_cream):
    """
    根据输入的特征判断喜好类型
    :param miles:
    :param percent:
    :param ice_cream:
    :return:
    """
    result_dict = {'didntLike': '讨厌',
                   'smallDoses': '喜欢',
                   'largeDoses': '非常喜欢'}
    # 打开的文件名
    filename = "datingTestSet.txt"
    dating_data_mat, labels = file2matrix(filename)
    # 训练集归一化
    norm_data_sets, ranges, min_values = auto_norm(dating_data_mat)
    # 测试集
    test_array = np.array([miles, percent, ice_cream])
    # 测试集归一化
    norm_test_array = (test_array - min_values) / ranges
    # 返回分类结果
    classifier_result = classify(norm_test_array, norm_data_sets, labels, 3)
    print("你可能{}这个人".format(result_dict[classifier_result]))


def img2vector(file_name):
    """
    32x32的二进制图像转换为1x1024的向量
    :param file_name:
    :return:
    """
    return_vector = np.zeros((1, 32*32))
    with open(file_name) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                return_vector[0, 32*i+j] = int(line[j])

    return return_vector


def hand_writing_class_test():
    """
    手写数字识别
    :return:
    """
    training_file_list = os.listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 32*32))
    labels = []
    for i in range(m):
        file_name = training_file_list[i]
        # 获取分类的数字
        number = int(file_name.split('.')[0].split('_')[0])
        labels.append(number)
        training_mat[i, :] = img2vector('trainingDigits/' + file_name)

    test_file_list = os.listdir('testDigits')
    m_test = len(test_file_list)
    error_count = 0
    for i in range(m_test):
        file_name = test_file_list[i]
        number = int(file_name.split('.')[0].split('_')[0])
        test_vector = img2vector('testDigits/' + file_name)
        classifier_result = classify(test_vector, training_mat, labels, 3)

        # print("分类返回结果为:  {}, 真实结果为: {}".format(classifier_result, number))
        if classifier_result != number:
            error_count += 1
    print("总共错了{}个数据, 错误率为{}".format(error_count, error_count / m_test))


# 直接调用sklearn中的KNeighborsClassifier
def hand_writing_class_test_2():
    """

    :return:
    """
    training_file_list = os.listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 32*32))
    labels = []
    for i in range(m):
        file_name = training_file_list[i]
        # 获取分类的数字
        number = int(file_name.split('.')[0].split('_')[0])
        labels.append(number)
        training_mat[i, :] = img2vector('trainingDigits/' + file_name)

    # 构建KNN分类器
    model = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型
    model.fit(training_mat, labels)

    test_file_list = os.listdir('testDigits')
    m_test = len(test_file_list)
    error_count = 0
    for i in range(m_test):
        file_name = test_file_list[i]
        number = int(file_name.split('.')[0].split('_')[0])
        test_vector = img2vector('testDigits/' + file_name)
        # 利用分类器做预测
        classifier_result = model.predict(test_vector)

        # print("分类返回结果为:  {}, 真实结果为: {}".format(classifier_result, number))
        if classifier_result != number:
            error_count += 1
    print("总共错了{}个数据, 错误率为{}".format(error_count, error_count / m_test))


if __name__ == '__main__':
    # 创建数据集和对应的标签
    data_sets, labels = create_data_set()
    # 测试集
    test = np.array([0, 0])
    k = 3
    test_class = classify(test, data_sets, labels, k)
    print('***********案例一************')
    print(test_class)

    # 案例二
    # 三维特征用户输入
    print('***********案例二************')
    # miles = float(input("每年获得的飞行常客里程数:"))
    # percent = float(input("玩视频游戏所耗时间百分比:"))
    # ice_cream = float(input("每周消费的冰激淋公升数:"))
    miles, percent, ice_cream = 20000, 10, 0.5
    classify_person(miles, percent, ice_cream)

    # 案例三
    print('***********案例三************')
    hand_writing_class_test()
    # hand_writing_class_test_2()

