import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing
import time
start_time=time.time()

SSE_list=[]
SSD_list=[]
upper=[]
lower=[]
Info=[]  # Summarize all the information
K_MAX=12
runtimes=25    # For this program, the max runtimes is 25, since the seed data set is limited, it is not real random

# Importation of Data

raw_data = pd.read_csv('Scripts/segment.csv')
df = raw_data.iloc[:, :-1]
df = np.array(df)
#  Implement of Z-score
df = preprocessing.scale(df)

Seed_index = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 105, 261, 212,
              1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812, 214,
              53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711, 1997, 1378, 827, 1875, 424,
              1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 672, 483,
              65, 92, 400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679,
              832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908,
              1964, 1260, 784, 520, 1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923,
              1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846,
              1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266,
              339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641,
              661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209,
              1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 1485,
              945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427,
              1434, 953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240,
              524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 1250, 613, 152,
              1440, 473, 1834, 1387, 1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611,
              1776, 123, 979, 1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686,
              854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582, 816, 1770, 663, 737,
              1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 1472]


def EuclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))


# Here, seed equals to the times the k-means have run, start with 0
def initCentroids(dataSet, k, seed):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    index = Seed_index[seed * k:seed * k + k]  # get the index set from index_set according to the k and times you run
    for i, j in zip(range(k), index):  # for each index you gain, assign an empty point to it forming a centroid
        centroids[i, :] = dataSet[j, :]  #  I use the orgin index of the seed data set
    return centroids


def kmeans(dataSet, k, seed, MAX_iteration=50):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True
    for i in range(numSamples):
        clusterAssment[i, 0] = -1
    ## step 1: init centroids
    centroids = initCentroids(dataSet, k, seed)

    iteration = 0
    while clusterChanged:
        iteration += 1
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 10000000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest then update its cluster
            for j in range(k):
                distance = EuclDistance(centroids[j, :], dataSet[i, :])  # 比较不同质心和自己的距离
                if distance < minDist:
                    minDist = distance
                    minIndex = j        # 分类给最近的质心
                    clusterAssment[i, 0] = minIndex  # 进行记录
        ## step 3: update centroids
        old_temp = centroids.copy()  # 矩阵的复制是假复制
        for j in range(k):  # 遍历所有分类
            pointsInCluster = []   #  初始化新质心
            for i in range(numSamples):  #遍历所有属于分类j的样本
                #print(iteration)
                if clusterAssment[i,0]==j:
                    pointsInCluster.append(dataSet[i])   # 将所选出的样本加入集合
            centroids[j, :] = np.mean(pointsInCluster, axis=0)   # 算出新质心
        if not (old_temp==centroids).all():   #  如果质心改变，再次循环
            #print('123')
            clusterChanged = True
        if iteration >= MAX_iteration:
            #print('Reached Max Iteration times, K_means halts')
            break
    #print(iteration)
    #print('Congratulations, cluster complete!')
    for i in range(numSamples):
        clusterAssment[i, 1] = np.linalg.norm(dataSet[i]-centroids[int(clusterAssment[i, 0])])**2
    return centroids, clusterAssment


for k in range(1,K_MAX+1):   # for each K
    Mean_SSE = 0  #  Reset Mean_SSE
    SSD_temp = []  # Sample standard deviation
    for i in range(0, runtimes):  # for each seed
        a, b = kmeans(df, k, i)
        # print(np.sum(b[:, 1]))
        Mean_SSE += np.sum(b[:, 1])  # record sum of SSE for every seed
        SSD_temp.append(np.sum(b[:, 1]))  # record each square error
    Mean_SSE = Mean_SSE / runtimes
    single_ssd = np.std(SSD_temp, ddof=1)
    upper_temp=Mean_SSE + 2 * single_ssd
    lower_temp=Mean_SSE - 2 * single_ssd
    SSD_list.append(2*single_ssd)
    SSE_list.append(Mean_SSE)   # Record each SSE, upper, lower for each k
    upper.append(upper_temp)
    lower.append(lower_temp)
    Info.append([k,Mean_SSE,single_ssd,upper_temp,lower_temp])

Info = pd.DataFrame(Info)
print("###########Summary##############")
Info.columns=['k','SSE','σk','upper','lower']
Info.to_csv('Summary.csv',index=False)
print(Info)


plt.errorbar(list(range(1,K_MAX+1)), SSE_list, yerr=SSD_list, color='blue')
plt.grid()
plt.xlabel("num of K")
plt.ylabel("SSE")
plt.title("Curve of SSE")
plt.show()

#  Record Runtime
print('Runtime: ',time.time()-start_time,' seconds')

#np.savetxt('',delimiter=',')


'''
#Using package to check the answer
k = np.average(df[:, :])
print(k)
for i in range(1, 13):
    k_means = KMeans(n_clusters=i, random_state=10, max_iter=50)
    k_means.fit(df)
    print(k_means.inertia_)
x = input()
'''