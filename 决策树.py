
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error  #平均绝对误差，用于评估预测结果和真实数据集的接近程度的程度其其值越小说明拟合效果越好。
from sklearn.metrics import mean_squared_error  #均方差，该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
import matplotlib.pyplot as plt
from matplotlib.pyplot import FuncFormatter
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import StandardScaler
all_data = pd.read_csv('D:\\Pythondata\\wcr.csv',header=0,index_col=0)   #读取文件 第一行当表头和第一列当索引


from sklearn.tree import export_graphviz

 

# data_test=all_data[424:448]   #利用索引选取第一个文件做测试集
# data_train=all_data.drop(index=[424,448],inplace=True) #选取后面所有文件做训练集

# scaler = StandardScaler()
data_test=all_data[all_data.index <=  19]   #利用索引选取第一个文件做测试集
data_train=all_data[all_data.index > 19]  #选取后面所有文件做训练集
norm=data_test
# data_test=scaler.fit_transform(data_test)
# data_train=scaler.fit_transform((data_train)
# mm = MinMaxScaler()
# # # 归一化
# # data_train= mm.fit_transform(data_train)
# # data_test= mm.fit_transform(data_test)
data_train=pd.DataFrame(normalize(data_train),columns = norm.columns)
data_test=pd.DataFrame(normalize(data_test),columns = norm.columns)

# data_train=pd.DataFrame(data_train,columns = norm.columns)   #列表格式转化为dataframe格式
# data_test=pd.DataFrame(data_test,columns = norm.columns)
X_train=data_train.iloc[:, 0:10]   #训练集中选取前10列
y_train=data_train.iloc[:, [10]]   #训练集中选取第11列

clf_tree = tree.DecisionTreeRegressor(random_state=(3))#实例化，建立评估模型对象，实例化用到的参数
clf_tree.fit(X_train, y_train)#模型接口，训练集   训练集数据训练模型

feature_names=['width','height','span','covering' ,'num_longitude', 'diameter_longitude','num_hoop','diameter_hoop','concrete_stress', 'deflection']
plt.figure(figsize=(17, 7))
plt.bar(feature_names, clf_tree.feature_importances_,width=0.5)
# plt.xticks(, cols, rotation=-45, fontsize=14)
plt.title('Feature importance', fontsize=15)
plt.show()


X_test=data_test.iloc[:, 0:10]    #测试集中选取前10列
y_test=data_test.iloc[:, [10]]    #测试集中选取第11列


y_test_pred = clf_tree.predict(X_test)  #测试集预测 

# path='D:/Pythondata/picture.dot'
# export_graphviz(clf_tree,out_file=path,feature_names=['width','height','span','covering' ,'num_longitude', 'diameter_longitude','num_hoop','diameter_hoop','concrete_stress', 'deflection'])

print(y_test_pred)
print(y_test)
print(mean_absolute_error(y_test, y_test_pred))
print(mean_squared_error(y_test, y_test_pred))
print(clf_tree.score(X_test, y_test))
#画图
deflection=data_test.iloc[:,[9]]    #测试集第10列作为x轴
resistance=data_test.iloc[:, [10]]  #测试集第11列作为y轴
pred_resistance=y_test_pred   #训练结果作为y轴进行对比
plt.plot(deflection,resistance,"bs-",label="Original data")
plt.plot(deflection,pred_resistance,"rs-",label="Predicted data")
plt.ylim(0,1)
# plt.xlim(0,0.00025)
# def formatnum(x, pos):
#     return '$%.1f$x$10^{-4}$' % (x*10000)
# formatter = FuncFormatter(formatnum)
# plt.gca().xaxis.set_major_formatter(formatter)
# 
plt.title("Decision Tree Regression")
plt.xlabel("deflection")
plt.ylabel("resistance")
plt.legend()
plt.grid(color='black',linestyle='--')
plt.show()
















