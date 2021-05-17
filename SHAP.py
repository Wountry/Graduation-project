# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:40:15 2021

@author: Wountry
"""

import pandas as pd
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
all_data = pd.read_csv('D:\\Pythondata\\wcr.csv',header=0,index_col=0)
norm=all_data
row_num=len(all_data)
col_num=all_data.columns.size
all_data= normalize(all_data) 
all_data=pd.DataFrame(all_data,columns = norm.columns) 
column_name = list(all_data.columns)
X=all_data.iloc[:,0:10]
y=all_data.iloc[:,[10]]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf_tree = tree.DecisionTreeRegressor(random_state=(3))
clf_tree.fit(X_train, y_train)

feature_names=['width','height','span','covering' ,'num_longitude', 'diameter_longitude','num_hoop','diameter_hoop','concrete_stress', 'deflection']
plt.figure(figsize=(17, 7))
plt.title('Feature importance', fontsize=18)
plt.bar(feature_names, clf_tree.feature_importances_,width=0.5)
# plt.xticks(, cols, rotation=-45, fontsize=14)
plt.xlabel("features",fontsize=18)
plt.ylabel("importances", fontsize=18)
plt.show()

explainer = shap.TreeExplainer(clf_tree)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train,plot_type='violin')
plt.show()
shap.summary_plot(shap_values, X_train, plot_type='bar')
plt.show()
shap.dependence_plot('span', shap_values, X_train, interaction_index='deflection')
shap.dependence_plot('deflection', shap_values, X_train, interaction_index='span')
shap.dependence_plot('span', shap_values, X_train, interaction_index='width')
shap.dependence_plot('width', shap_values, X_train, interaction_index='span')
shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[55], X_train.iloc[0],show=True, matplotlib=True)
plt.show()

# shap.plots.scatter(shap_values[:,"span"])


# shap.plots.heatmap(shap_values[:100])
shap.plots.bar(shap_values)
plt.show()
y_test_tree = clf_tree.predict(X_test)
mae_tree=mean_absolute_error(y_test, y_test_tree)
mse_tree=mean_squared_error(y_test, y_test_tree)
print('mae_tree is',mae_tree)
print('mse_tree is',mse_tree)

regression=LinearRegression()
regression.fit(X_train,y_train)
y_test_linear = regression.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_test_linear)
mse_linear = mean_squared_error(y_test, y_test_linear)
print('mae_linear is',mae_linear)
print('mse_linear is',mse_linear)



























# bp = MLPClassifier(hidden_layer_sizes=(500,), activation='relu',
#                     solver='lbfgs', alpha=0.0001, batch_size='auto',
#                     learning_rate='constant')
# print(ravel(y_train))
# bp.fit(X_train,y_train)


# forest=RandomForestRegressor()
# forest.fit(X_train,y_train)
# y_test_forest = forest.predict(X_test)
# mae_forest = mean_absolute_error(y_test, y_test_linear)
# mse_forest = mean_squared_error(y_test, y_test_linear)
# print('mae_forest is',mae_forest)
# print('mse_forest is',mse_forest)

# scaler = StandardScaler()  # 标准化转换
# scaler.fit(X_test)  # 训练标准化对象
# x_test_Standard = scaler.transform(X_test)  # 转换数据集
# scaler.fit(X_train)  # 训练标准化对象
# x_train_Standard = scaler.transform(X_train)  # 转换数据集
# bp = MLP Regressor(hidden_layer_sizes=(500,), activation='relu',
#                     solver='lbfgs', alpha=0.0001, batch_size='auto',
#                     learning_rate='constant')
# bp.fit(x_train_Standard, y_train.values.ravel())
# y_predict = bp.predict(x_test_Standard)
# model = MLPRegressor(hidden_layer_sizes=(10,), random_state=10, learning_rate_init=0.1)  # BP神经网络回归模型
# model.fit(X_train,y_train )  # 训练模型
# pre = model.predict(X_test)  # 模型预测
# np.abs(y_test- pre).mean()  # 模型评价
