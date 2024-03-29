import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier as KNN

def img2vector(filename):
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1x1024向量
    return returnVect

def handwritingClassTest():
    # 定义测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = os.listdir('E:\\pycharm_date\\pytorch\\emprical\\kNN_hand_writing\\trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,训练集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector(
            os.path.join('E:\\pycharm_date\\pytorch\\emprical\\kNN_hand_writing\\trainingDigits', fileNameStr))
    return trainingMat, hwLabels


# 构建kNN分类器
neigh = KNN(n_neighbors=3,weights='distance') #weights='distance' 类似加权平均，以距离的倒数作为权重的"加权众数"。
# 拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
trainingMat, hwLabels = handwritingClassTest()
neigh.fit(trainingMat, hwLabels)
# 返回testDigits目录下的文件列表
testFileList = os.listdir('E:\\pycharm_date\\pytorch\\emprical\\kNN_hand_writing\\testDigits')
# 错误检测计数
errorCount = 0.0
# 测试数据的数量
mTest = len(testFileList)
# 从文件中解析出测试集的类别并进行 分类测试
for i in range(mTest):
    # 获得文件的名字
    fileNameStr = testFileList[i]
    # 获得分类的数字
    classNumber = int(fileNameStr.split('_')[0])
    # 获得测试集的1x1024向量,用于训练
    vectorUnderTest = img2vector(os.path.join('E:\\pycharm_date\\pytorch\\emprical\\kNN_hand_writing\\testDigits', fileNameStr))
    # 获得预测结果
    classifierResult = neigh.predict(vectorUnderTest)

    print(fileNameStr + "分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
    if classifierResult != classNumber:
        errorCount += 1.0
print(f"总共错了{errorCount}个数据\n错误率为{errorCount / mTest}")
